"""Validated inputs for rebuilding Session 5 lateralized decoder models.

This module deliberately begins with file/event validation only. Filtering,
epoching, xDAWN, balancing, pruning, and model fitting are added in later
steps after the manifest has been reviewed.
"""

from pathlib import Path
import os
import gc
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.special import expit

from .analysis import load_training_analysis_file
from .bci import (
    BCI_SESSION_EXCEPTIONS,
    EXPECTED_REAL_RUNS_BY_SESSION,
    load_decoding_analysis_file,
)
from .config import EXPECTED_SUBJECTS, PROJECT_ROOT, get_subject_group
from .eeg import (
    DEFAULT_EPOCH_TMAX,
    DEFAULT_EPOCH_TMIN,
    EXPECTED_TRAINING_RUNS_BY_SESSION,
    _get_training_run_gdf_files_for_session,
    load_filter_epoch_baseline_correct_training_run,
    select_analysis_eeg_channels,
)
from .triggers import FS, STIMULUS_CODES, TRAINING_TRIALS, load_training_trigger_file


SESSION5_TRAINING_RUNS = EXPECTED_TRAINING_RUNS_BY_SESSION[5]
SESSION1_TRAINING_RUNS = EXPECTED_TRAINING_RUNS_BY_SESSION[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_TASK_CODES = {4, *STIMULUS_CODES, 64}
_CONDITION_BY_TRIGGER = dict(STIMULUS_CODES)
CONVENTIONAL_PPO_PAIRS = (
    ("P2", "P1"),
    ("P4", "P3"),
    ("P6", "P5"),
    ("P8", "P7"),
    ("PO4", "PO3"),
    ("PO6", "PO5"),
    ("PO8", "PO7"),
)


def select_poststimulus_feature_window(time, start_s=0.2, stop_s=None):
    """Return inclusive time indices for the post-stimulus xDAWN/feature window."""
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or time.size == 0:
        raise ValueError(f"time must be a non-empty 1D array, got {time.shape}.")
    if stop_s is None:
        stop_s = float(time[-1])
    if start_s < time[0] or stop_s > time[-1] or start_s > stop_s:
        raise ValueError(
            f"Requested window {start_s:g}-{stop_s:g} s is outside available "
            f"time axis {time[0]:g}-{time[-1]:g} s."
        )
    indices = np.flatnonzero((time >= start_s) & (time <= stop_s))
    if indices.size < 2:
        raise ValueError(
            f"Window {start_s:g}-{stop_s:g} s contains fewer than two samples."
        )
    return indices


def fit_xdawn_fold(epochs_time_channels_trials, labels, epoch_sample_indices, n_components=2):
    """Fit original-style xDAWN on one fold's training trials only.

    Parameters use the repository's ``time x channels x trials`` convention.
    This is the simple class-average branch of the original ``xdawn.m``:
    pooled full-epoch covariance, class-average evoked covariance within the
    requested window, generalized eigendecomposition, then two filters per
    class. The returned ``positive_class_filters`` are solely the filters for
    binary class 1, as in ``processFeatures``.
    """
    epochs = np.asarray(epochs_time_channels_trials, dtype=float)
    labels = np.asarray(labels).reshape(-1)
    sample_indices = np.asarray(epoch_sample_indices, dtype=int).reshape(-1)
    if epochs.ndim != 3:
        raise ValueError(
            "epochs_time_channels_trials must be time x channels x trials, "
            f"got {epochs.shape}."
        )
    n_times, n_channels, n_trials = epochs.shape
    if labels.size != n_trials:
        raise ValueError(f"labels has {labels.size} entries, but epochs has {n_trials} trials.")
    if not np.isfinite(epochs).all() or not np.isfinite(labels).all():
        raise ValueError("xDAWN input epochs and labels must be finite.")
    classes = np.unique(labels)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"xDAWN requires binary labels [0, 1], got {classes.tolist()}.")
    if n_components < 1 or n_components > n_channels:
        raise ValueError(
            f"n_components must be between 1 and {n_channels}, got {n_components}."
        )
    if sample_indices.size < 2 or sample_indices.min() < 0 or sample_indices.max() >= n_times:
        raise ValueError(
            f"Invalid xDAWN epoch_sample_indices for {n_times} time samples: "
            f"{sample_indices.tolist()}."
        )
    if not np.array_equal(sample_indices, np.unique(sample_indices)):
        raise ValueError("xDAWN epoch_sample_indices must be unique and sorted.")

    # MATLAB: epochs_data = permute(epochs_data, [3 2 1]); then
    # cov(reshape(epochs_data, n_epochs*n_times, n_channels)). In NumPy, the
    # equivalent row layout is trial-major then time-major.
    trial_channel_time = np.transpose(epochs, (2, 1, 0))
    signal_rows = np.transpose(trial_channel_time, (0, 2, 1)).reshape(
        n_trials * n_times, n_channels
    )
    signal_cov = np.cov(signal_rows, rowvar=False, ddof=1)
    if np.linalg.matrix_rank(signal_cov) < n_channels:
        raise ValueError(
            "Pooled xDAWN signal covariance is rank deficient; cannot reproduce "
            "the unregularized source generalized eigendecomposition."
        )

    all_filters = []
    all_patterns = []
    evokeds = {}
    eigenvalues_by_class = {}
    for class_label in classes:
        prototype = trial_channel_time[labels == class_label].mean(axis=0)
        evoked_window = prototype[:, sample_indices]
        evoked_cov = np.cov(evoked_window.T, rowvar=False, ddof=1)
        eigenvalues, eigenvectors = linalg.eigh(evoked_cov, signal_cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.real_if_close(eigenvalues[order])
        eigenvectors = np.real_if_close(eigenvectors[:, order])
        eigenvectors = eigenvectors / np.sqrt(np.sum(eigenvectors ** 2, axis=0))
        filters = eigenvectors[:, :n_components].T
        all_filters.append(filters)
        all_patterns.append(np.linalg.pinv(filters))
        evokeds[int(class_label)] = prototype
        eigenvalues_by_class[int(class_label)] = eigenvalues

    all_filters = np.concatenate(all_filters, axis=0)
    all_patterns = np.concatenate(all_patterns, axis=0)
    positive_start = int(np.flatnonzero(classes == 1)[0]) * n_components
    positive_filters = all_filters[positive_start:positive_start + n_components]
    if positive_filters.shape != (n_components, n_channels):
        raise RuntimeError("Unexpected positive-class xDAWN filter shape.")
    print("Fold-local xDAWN fit passed.")
    print(
        f"  Training input: {n_trials} trials, {n_channels} channels, {n_times} time samples."
    )
    print(
        f"  Class counts: no={int((labels == 0).sum())}, "
        f"distractor={int((labels == 1).sum())}."
    )
    print(
        f"  Evoked covariance window: {sample_indices.size} samples "
        f"(indices {sample_indices[0]}-{sample_indices[-1]})."
    )
    print(f"  Returned class-1 xDAWN filters: {positive_filters.shape}.")
    return {
        "all_filters_components_by_channels": all_filters,
        "all_patterns_channels_by_components": all_patterns,
        "positive_class_filters_components_by_channels": positive_filters,
        "signal_covariance": signal_cov,
        "evokeds_channels_by_time": evokeds,
        "eigenvalues_by_class": eigenvalues_by_class,
        "classes": classes.astype(int),
        "n_components": int(n_components),
        "epoch_sample_indices": sample_indices,
    }


def apply_xdawn_filters(epochs_time_channels_trials, filters_components_by_channels):
    """Apply fitted xDAWN filters without refitting any transform."""
    epochs = np.asarray(epochs_time_channels_trials, dtype=float)
    filters = np.asarray(filters_components_by_channels, dtype=float)
    if epochs.ndim != 3:
        raise ValueError(f"epochs must be time x channels x trials, got {epochs.shape}.")
    if filters.ndim != 2 or filters.shape[1] != epochs.shape[1]:
        raise ValueError(
            "filters must be components x matching channels; got "
            f"filters {filters.shape}, epochs {epochs.shape}."
        )
    projected = np.einsum("tcn,kc->tkn", epochs, filters, optimize=True)
    if not np.isfinite(projected).all():
        raise ValueError("xDAWN projection produced non-finite values.")
    return projected


def balance_binary_trials_within_run(
    trial_table,
    labels,
    active_mask=None,
    training_run_ids=None,
    random_seed=0,
    allow_empty_runs=False,
):
    """Select equal numbers of binary classes independently within each run.

    This mirrors the role of the source ``balanceRuns`` function. ``active_mask``
    represents the currently retained trials (and will later be the cumulative
    pruning mask). ``training_run_ids`` permits a leave-one-run-out caller to
    balance only its training runs; no held-out run is selected.
    """
    if not isinstance(trial_table, pd.DataFrame):
        raise TypeError("trial_table must be a pandas DataFrame.")
    if "run_id" not in trial_table.columns:
        raise ValueError("trial_table must contain a 'run_id' column.")
    labels = np.asarray(labels).reshape(-1)
    n_trials = len(trial_table)
    if labels.size != n_trials:
        raise ValueError(f"labels has {labels.size} entries but trial_table has {n_trials} rows.")
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError(f"Balancing requires labels [0, 1], got {np.unique(labels).tolist()}.")
    if active_mask is None:
        active_mask = np.ones(n_trials, dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if active_mask.size != n_trials:
            raise ValueError(
                f"active_mask has {active_mask.size} entries but expected {n_trials}."
            )
    all_run_ids = trial_table["run_id"].to_numpy()
    if training_run_ids is None:
        training_run_ids = sorted(pd.unique(all_run_ids).tolist())
    else:
        training_run_ids = sorted({int(run_id) for run_id in training_run_ids})
    if not training_run_ids:
        raise ValueError("training_run_ids is empty.")
    unknown_runs = sorted(set(training_run_ids) - set(pd.unique(all_run_ids)))
    if unknown_runs:
        raise ValueError(f"training_run_ids not present in trial_table: {unknown_runs}.")

    rng = np.random.default_rng(random_seed)
    selected_mask = np.zeros(n_trials, dtype=bool)
    audit_rows = []
    for run_id in training_run_ids:
        in_run = all_run_ids == run_id
        active_indices = np.flatnonzero(in_run & active_mask)
        class_zero = active_indices[labels[active_indices] == 0]
        class_one = active_indices[labels[active_indices] == 1]
        if (len(class_zero) == 0 or len(class_one) == 0) and not allow_empty_runs:
            raise ValueError(
                f"Run {run_id} cannot be balanced after pruning: "
                f"no={len(class_zero)}, distractor={len(class_one)}."
            )
        n_keep_per_class = min(len(class_zero), len(class_one))
        chosen_zero = (
            rng.choice(class_zero, size=n_keep_per_class, replace=False)
            if n_keep_per_class else np.array([], dtype=int)
        )
        chosen_one = (
            rng.choice(class_one, size=n_keep_per_class, replace=False)
            if n_keep_per_class else np.array([], dtype=int)
        )
        selected_mask[chosen_zero] = True
        selected_mask[chosen_one] = True
        audit_rows.append({
            "run_id": int(run_id),
            "n_active_no": int(len(class_zero)),
            "n_active_distractor": int(len(class_one)),
            "n_selected_no": int(len(chosen_zero)),
            "n_selected_distractor": int(len(chosen_one)),
            "n_selected_total": int(2 * n_keep_per_class),
        })

    if np.any(selected_mask & ~active_mask):
        raise RuntimeError("Balancing selected a trial outside active_mask.")
    selected_labels = labels[selected_mask]
    if selected_labels.size == 0 or not np.array_equal(np.unique(selected_labels), np.array([0, 1])):
        raise RuntimeError("Balanced selection does not contain both classes across all runs.")
    audit = pd.DataFrame(audit_rows)
    if not (audit["n_selected_no"] == audit["n_selected_distractor"]).all():
        raise RuntimeError("Within-run balancing did not yield equal class counts.")
    print("Within-run binary balancing passed.")
    print(f"  Random seed: {random_seed}")
    print(f"  Training runs balanced: {training_run_ids}")
    print(f"  Active trials available: {int(active_mask.sum())}")
    print(
        f"  Selected trials: {int(selected_mask.sum())}; "
        f"no={int((selected_labels == 0).sum())}, "
        f"distractor={int((selected_labels == 1).sum())}."
    )
    return {
        "selected_mask": selected_mask,
        "audit": audit,
        "random_seed": int(random_seed),
        "training_run_ids": training_run_ids,
    }


def _stride_resample_and_flatten(projected_epochs, window_indices, ratio):
    """Match MATLAB time-within-component feature flattening."""
    projected_epochs = np.asarray(projected_epochs, dtype=float)
    if projected_epochs.ndim != 3:
        raise ValueError(
            "projected_epochs must be time x components x trials, "
            f"got {projected_epochs.shape}."
        )
    if not isinstance(ratio, (int, np.integer)) or ratio < 1:
        raise ValueError(f"ratio must be a positive integer, got {ratio!r}.")
    samples = np.asarray(window_indices, dtype=int).reshape(-1)
    if samples.size == 0:
        raise ValueError("window_indices is empty.")
    resampled_indices = samples[::ratio]
    resampled = projected_epochs[resampled_indices, :, :]
    # MATLAB reshape(time x component x trial, [], n_trials) advances time
    # first, then component. For NumPy this is explicit Fortran ordering.
    features = np.reshape(
        resampled,
        (resampled.shape[0] * resampled.shape[1], resampled.shape[2]),
        order="F",
    )
    if not np.isfinite(features).all():
        raise ValueError("Resampled features contain non-finite values.")
    return features, resampled_indices


def _fit_zscore_training_features(features):
    """Fit MATLAB-compatible sample-SD feature z-scoring on training data."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] < 2:
        raise ValueError(
            "features must be feature x >=2 training trials, got "
            f"{features.shape}."
        )
    means = features.mean(axis=1, keepdims=True)
    stds = features.std(axis=1, ddof=1, keepdims=True)
    zero_variance = np.flatnonzero(stds[:, 0] <= np.finfo(float).eps)
    if zero_variance.size:
        raise ValueError(
            "Cannot z-score zero-variance training feature(s): "
            f"{zero_variance.tolist()}."
        )
    return means, stds


def _apply_zscore_features(features, means, stds):
    """Apply a previously fitted per-feature z-score transform."""
    features = np.asarray(features, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    if features.ndim != 2 or means.shape != (features.shape[0], 1) or stds.shape != means.shape:
        raise ValueError(
            "Feature/z-score dimensions do not align: "
            f"features={features.shape}, means={means.shape}, stds={stds.shape}."
        )
    normalized = (features - means) / stds
    if not np.isfinite(normalized).all():
        raise ValueError("Z-score normalization produced non-finite values.")
    return normalized


def compute_binary_feature_r2(features, labels):
    """Compute source-equivalent squared Pearson r2 for every feature."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if features.ndim != 2 or features.shape[1] != labels.size:
        raise ValueError(
            "features must be feature x trial and align with labels; got "
            f"features={features.shape}, labels={labels.shape}."
        )
    if not np.array_equal(np.unique(labels), np.array([0.0, 1.0])):
        raise ValueError(f"r2 requires binary labels [0, 1], got {np.unique(labels).tolist()}.")
    centered_features = features - features.mean(axis=1, keepdims=True)
    centered_labels = labels - labels.mean()
    denominator = np.sqrt(
        np.sum(centered_features ** 2, axis=1) * np.sum(centered_labels ** 2)
    )
    if np.any(denominator <= np.finfo(float).eps):
        bad = np.flatnonzero(denominator <= np.finfo(float).eps)
        raise ValueError(f"Cannot calculate r2 for zero-variance feature(s): {bad.tolist()}.")
    correlations = centered_features @ centered_labels / denominator
    return correlations ** 2


def fit_fold_feature_pipeline(
    training_epochs_time_channels_trials,
    training_labels,
    heldout_epochs_time_channels_trials,
    time,
    feature_start_s=0.2,
    feature_stop_s=None,
    resample_ratio=8,
    n_xdawn_components=2,
    n_selected_features=30,
):
    """Fit and apply all fold-local transforms through r2 feature selection.

    xDAWN, z-scoring, and r2 ranking are fitted only on ``training_*``. The
    held-out epochs are projected, resampled, normalized, and feature-selected
    only with those fitted quantities. Classification is intentionally outside
    this function and will be added after this stage is inspected.
    """
    training_epochs = np.asarray(training_epochs_time_channels_trials, dtype=float)
    heldout_epochs = np.asarray(heldout_epochs_time_channels_trials, dtype=float)
    training_labels = np.asarray(training_labels).reshape(-1)
    time = np.asarray(time, dtype=float)
    if training_epochs.ndim != 3 or heldout_epochs.ndim != 3:
        raise ValueError("Training and held-out epochs must both be 3D time x channels x trials.")
    if training_epochs.shape[:2] != heldout_epochs.shape[:2]:
        raise ValueError(
            "Training and held-out time/channel dimensions differ: "
            f"training={training_epochs.shape}, held-out={heldout_epochs.shape}."
        )
    if training_epochs.shape[2] != training_labels.size:
        raise ValueError("Training epoch trial count does not match training_labels.")
    if time.size != training_epochs.shape[0]:
        raise ValueError("time length does not match epoch time dimension.")

    window_indices = select_poststimulus_feature_window(time, feature_start_s, feature_stop_s)
    xdawn_fit = fit_xdawn_fold(
        training_epochs,
        training_labels,
        window_indices,
        n_components=n_xdawn_components,
    )
    filters = xdawn_fit["positive_class_filters_components_by_channels"]
    training_projected = apply_xdawn_filters(training_epochs, filters)
    heldout_projected = apply_xdawn_filters(heldout_epochs, filters)
    training_features, resampled_indices = _stride_resample_and_flatten(
        training_projected, window_indices, resample_ratio
    )
    heldout_features, heldout_resampled_indices = _stride_resample_and_flatten(
        heldout_projected, window_indices, resample_ratio
    )
    if not np.array_equal(resampled_indices, heldout_resampled_indices):
        raise RuntimeError("Training and held-out resampling indices differ.")
    means, stds = _fit_zscore_training_features(training_features)
    training_normalized = _apply_zscore_features(training_features, means, stds)
    heldout_normalized = _apply_zscore_features(heldout_features, means, stds)
    r2 = compute_binary_feature_r2(training_normalized, training_labels)
    if n_selected_features < 1 or n_selected_features > len(r2):
        raise ValueError(
            f"n_selected_features must be 1..{len(r2)}, got {n_selected_features}."
        )
    selected_indices = np.argsort(-r2, kind="stable")[:n_selected_features]
    n_resampled_time = len(resampled_indices)
    selected_coordinates = pd.DataFrame({
        "feature_index_zero_based": selected_indices,
        "component": (selected_indices // n_resampled_time) + 1,
        "time_index": resampled_indices[selected_indices % n_resampled_time],
        "time_s": time[resampled_indices[selected_indices % n_resampled_time]],
        "r2": r2[selected_indices],
    })
    print("Fold-local feature pipeline passed.")
    print(
        f"  Candidate features: {training_features.shape[0]} "
        f"({n_resampled_time} timepoints x {n_xdawn_components} components)."
    )
    print(
        f"  Training/held-out matrices after selection: "
        f"{training_normalized[selected_indices].shape} / "
        f"{heldout_normalized[selected_indices].shape}."
    )
    return {
        "xdawn_fit": xdawn_fit,
        "feature_window_indices": window_indices,
        "resampled_indices": resampled_indices,
        "resampled_time_s": time[resampled_indices],
        "normalization_means": means,
        "normalization_stds": stds,
        "training_features_normalized": training_normalized,
        "heldout_features_normalized": heldout_normalized,
        "r2": r2,
        "selected_indices_zero_based": selected_indices,
        "selected_coordinates": selected_coordinates,
        "training_selected_features": training_normalized[selected_indices, :],
        "heldout_selected_features": heldout_normalized[selected_indices, :],
    }


def fit_regularized_linear_lda(features, labels, gamma=0.05):
    """Fit the source model's uniform-prior regularized linear LDA.

    ``features`` follows the project convention (features x trials). The
    pooled within-class empirical covariance uses the MATLAB denominator
    ``N - K``. Gamma then applies the documented `fitcdiscr` regularization:
    ``(1-gamma) * Sigma + gamma * diag(diag(Sigma))``.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels).reshape(-1)
    if features.ndim != 2 or features.shape[1] != labels.size:
        raise ValueError(
            "features must be feature x trial and align with labels; got "
            f"features={features.shape}, labels={labels.shape}."
        )
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError(f"LDA requires binary labels [0, 1], got {np.unique(labels).tolist()}.")
    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
    class_zero = features[:, labels == 0].T
    class_one = features[:, labels == 1].T
    if len(class_zero) < 2 or len(class_one) < 2:
        raise ValueError("Each LDA class requires at least two training trials.")
    mean_zero = class_zero.mean(axis=0)
    mean_one = class_one.mean(axis=0)
    centered = np.vstack((class_zero - mean_zero, class_one - mean_one))
    pooled_covariance = centered.T @ centered / (features.shape[1] - 2)
    regularized_covariance = (
        (1.0 - gamma) * pooled_covariance
        + gamma * np.diag(np.diag(pooled_covariance))
    )
    if np.linalg.matrix_rank(regularized_covariance) < features.shape[0]:
        raise ValueError(
            "Regularized LDA covariance is rank deficient; unable to reproduce "
            "the source linear inversion."
        )
    linear_weights = linalg.solve(regularized_covariance, mean_one - mean_zero)
    # `Prior`, 'uniform' gives log(0.5 / 0.5) = 0. This equals the binary
    # class-1-versus-class-0 linear discriminant stored as Coeffs(2,1).
    intercept = -0.5 * (mean_one + mean_zero) @ linear_weights
    training_distances = features.T @ linear_weights + intercept
    if not np.isfinite(training_distances).all():
        raise ValueError("LDA training distances are non-finite.")
    print("Fold-local regularized linear LDA fit passed.")
    print(
        f"  Features/trials: {features.shape[0]} x {features.shape[1]}; "
        f"Gamma={gamma:.3g}; uniform priors."
    )
    return {
        "linear_weights": linear_weights,
        "intercept": float(intercept),
        "mean_class_zero": mean_zero,
        "mean_class_one": mean_one,
        "pooled_covariance": pooled_covariance,
        "regularized_covariance": regularized_covariance,
        "gamma": float(gamma),
        "training_distances": training_distances,
    }


def fit_source_sigmoid_calibration(training_distances):
    """Fit the original decoder's quantile-derived sigmoid slope."""
    distances = np.asarray(training_distances, dtype=float).reshape(-1)
    if distances.size < 2 or not np.isfinite(distances).all():
        raise ValueError("training_distances must contain at least two finite values.")
    lower_probability = 0.025
    upper_probability = 1.0 - lower_probability
    lower_distance, upper_distance = np.percentile(
        distances, [100 * lower_probability, 100 * upper_probability]
    )
    if lower_distance == 0 or upper_distance == 0:
        raise ValueError(
            "Cannot fit source sigmoid: the 2.5th or 97.5th distance quantile is zero."
        )
    lower_slope = -np.log((1 - lower_probability) / lower_probability) / lower_distance
    upper_slope = -np.log((1 - upper_probability) / upper_probability) / upper_distance
    slope = float((lower_slope + upper_slope) / 2.0)
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError(
            "Source sigmoid slope must be finite and positive; got "
            f"{slope} from quantiles ({lower_distance}, {upper_distance})."
        )
    return {
        "slope": slope,
        "lower_distance_quantile": float(lower_distance),
        "upper_distance_quantile": float(upper_distance),
        "lower_slope": float(lower_slope),
        "upper_slope": float(upper_slope),
    }


def predict_source_lda_posterior(features, lda_fit, calibration):
    """Return custom calibrated class-1 probabilities without refitting."""
    features = np.asarray(features, dtype=float)
    weights = np.asarray(lda_fit["linear_weights"], dtype=float).reshape(-1)
    if features.ndim != 2 or features.shape[0] != weights.size:
        raise ValueError(
            "features must be feature x trial and align with LDA weights; got "
            f"features={features.shape}, weights={weights.shape}."
        )
    distances = features.T @ weights + float(lda_fit["intercept"])
    posterior = expit(float(calibration["slope"]) * distances)
    if not np.isfinite(posterior).all() or np.any((posterior < 0) | (posterior > 1)):
        raise ValueError("LDA posterior prediction is not finite within [0, 1].")
    return {"distances": distances, "posterior": posterior}


def fit_fold_lda_and_calibrate(feature_pipeline, training_labels, gamma=0.05):
    """Fit LDA/calibration on a fold feature pipeline and score its held-out data."""
    if not isinstance(feature_pipeline, dict):
        raise TypeError("feature_pipeline must be returned by fit_fold_feature_pipeline.")
    lda_fit = fit_regularized_linear_lda(
        feature_pipeline["training_selected_features"], training_labels, gamma=gamma
    )
    calibration = fit_source_sigmoid_calibration(lda_fit["training_distances"])
    training_prediction = predict_source_lda_posterior(
        feature_pipeline["training_selected_features"], lda_fit, calibration
    )
    heldout_prediction = predict_source_lda_posterior(
        feature_pipeline["heldout_selected_features"], lda_fit, calibration
    )
    print("Fold-local LDA posterior calibration passed.")
    print(
        f"  Sigmoid slope={calibration['slope']:.6g}; held-out posterior range "
        f"{heldout_prediction['posterior'].min():.4f}-"
        f"{heldout_prediction['posterior'].max():.4f}."
    )
    return {
        "lda_fit": lda_fit,
        "calibration": calibration,
        "training_prediction": training_prediction,
        "heldout_prediction": heldout_prediction,
    }


def uniform_prior_precision_recall_auc(labels, scores):
    """Match ``perfcurve(..., Prior='uniform', reca, prec)`` for binary data.

    At every descending unique score threshold, recall is TPR and precision is
    recalculated using equal class priors: ``TPR / (TPR + FPR)``. MATLAB's
    returned AUC is trapezoidal integration of this precision-recall curve.
    """
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if labels.size != scores.size or labels.size == 0:
        raise ValueError(f"labels/scores must be same nonzero length, got {labels.size}/{scores.size}.")
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError(f"AUPRC requires binary labels [0, 1], got {np.unique(labels).tolist()}.")
    if not np.isfinite(scores).all():
        raise ValueError("AUPRC scores must be finite.")
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    thresholds = np.sort(np.unique(scores))[::-1]
    recall = [0.0]
    precision = [np.nan]
    output_thresholds = [float(thresholds[0])]
    for threshold in thresholds:
        predicted_positive = scores >= threshold
        true_positive_rate = float(np.sum(predicted_positive & (labels == 1)) / n_positive)
        false_positive_rate = float(np.sum(predicted_positive & (labels == 0)) / n_negative)
        denominator = true_positive_rate + false_positive_rate
        recall.append(true_positive_rate)
        precision.append(
            true_positive_rate / denominator if denominator > 0 else np.nan
        )
        output_thresholds.append(float(threshold))
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    output_thresholds = np.asarray(output_thresholds, dtype=float)
    valid = np.isfinite(precision)
    auc = float(np.trapezoid(precision[valid], recall[valid]))
    return {
        "auc": auc,
        "recall": recall,
        "precision": precision,
        "thresholds": output_thresholds,
    }


def run_leave_one_run_out_feature_cv(
    model_input,
    time,
    active_mask=None,
    random_seed=20260812,
    feature_start_s=0.2,
    feature_stop_s=None,
    resample_ratio=8,
    n_xdawn_components=2,
    n_selected_features=30,
    gamma=0.05,
):
    """Generate pooled held-out posteriors with fully fold-local transforms.

    This is one evaluation pass for one participant and one decoder side. It
    does not prune trials: ``active_mask`` only controls training eligibility.
    Each held-out run is scored in full, including trials excluded by
    ``active_mask``, consistent with the source iterative-pruning evaluation.
    """
    required = {"epochs_time_channels_trials", "labels", "trial_table"}
    if not isinstance(model_input, dict) or not required.issubset(model_input):
        raise ValueError(f"model_input must contain {sorted(required)}.")
    epochs = np.asarray(model_input["epochs_time_channels_trials"], dtype=float)
    labels = np.asarray(model_input["labels"]).reshape(-1).astype(int)
    trials = model_input["trial_table"].reset_index(drop=True).copy()
    if epochs.ndim != 3 or epochs.shape[2] != len(labels) or len(trials) != len(labels):
        raise ValueError(
            "Model input epochs, labels, and trial table do not align: "
            f"epochs={epochs.shape}, labels={len(labels)}, trials={len(trials)}."
        )
    if "run_id" not in trials:
        raise ValueError("model_input trial table is missing 'run_id'.")
    if active_mask is None:
        active_mask = np.ones(len(labels), dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if len(active_mask) != len(labels):
            raise ValueError("active_mask must align with model-input trials.")
    run_ids = sorted(pd.unique(trials["run_id"]).tolist())
    if len(run_ids) < 2:
        raise ValueError("Leave-one-run-out CV requires at least two runs.")

    posterior = np.full(len(labels), np.nan, dtype=float)
    # Source behavior: balance all active runs once per pruning iteration, then
    # reuse that shared balanced pool for every leave-one-run-out fold.
    balanced_pool = balance_binary_trials_within_run(
        trials,
        labels,
        active_mask=active_mask,
        training_run_ids=run_ids,
        random_seed=int(random_seed),
        allow_empty_runs=True,
    )
    balanced_pool_mask = balanced_pool["selected_mask"]
    fold_rows = []
    selected_feature_rows = []
    for fold_number, heldout_run_id in enumerate(run_ids, start=1):
        heldout_mask = trials["run_id"].to_numpy() == heldout_run_id
        training_mask = balanced_pool_mask & ~heldout_mask
        if np.any(training_mask & heldout_mask):
            raise RuntimeError(f"Fold {fold_number}: balancing selected held-out trials.")
        training_classes = np.unique(labels[training_mask])
        training_class_counts = np.bincount(labels[training_mask], minlength=2)
        if (
            not np.array_equal(training_classes, np.array([0, 1]))
            or np.any(training_class_counts < 2)
        ):
            raise ValueError(
                f"Fold {fold_number} (held-out run {heldout_run_id}) is untrainable "
                f"after balancing: n_training={int(training_mask.sum())}, "
                f"classes={training_classes.tolist()}, "
                f"class_counts={training_class_counts.tolist()}; each LDA class requires "
                "at least two training trials."
            )
        feature_pipeline = fit_fold_feature_pipeline(
            epochs[:, :, training_mask],
            labels[training_mask],
            epochs[:, :, heldout_mask],
            time,
            feature_start_s=feature_start_s,
            feature_stop_s=feature_stop_s,
            resample_ratio=resample_ratio,
            n_xdawn_components=n_xdawn_components,
            n_selected_features=n_selected_features,
        )
        classifier = fit_fold_lda_and_calibrate(
            feature_pipeline, labels[training_mask], gamma=gamma
        )
        heldout_posterior = classifier["heldout_prediction"]["posterior"]
        posterior[heldout_mask] = heldout_posterior
        fold_rows.append({
            "fold": fold_number,
            "heldout_run_id": int(heldout_run_id),
            "n_heldout": int(heldout_mask.sum()),
            "n_heldout_no": int((labels[heldout_mask] == 0).sum()),
            "n_heldout_distractor": int((labels[heldout_mask] == 1).sum()),
            "n_active_training": int((active_mask & ~heldout_mask).sum()),
            "n_balanced_training": int(training_mask.sum()),
            "sigmoid_slope": float(classifier["calibration"]["slope"]),
            "heldout_posterior_min": float(heldout_posterior.min()),
            "heldout_posterior_max": float(heldout_posterior.max()),
        })
        coordinates = feature_pipeline["selected_coordinates"].copy()
        coordinates.insert(0, "heldout_run_id", int(heldout_run_id))
        coordinates.insert(0, "fold", fold_number)
        selected_feature_rows.append(coordinates)

    if not np.isfinite(posterior).all():
        missing = np.flatnonzero(~np.isfinite(posterior))
        raise RuntimeError(
            "LOO CV did not produce exactly one finite posterior per trial; "
            f"missing indices: {missing.tolist()}."
        )
    auprc = uniform_prior_precision_recall_auc(labels, posterior)
    fold_audit = pd.DataFrame(fold_rows)
    selected_features = pd.concat(selected_feature_rows, ignore_index=True)
    print("Leave-one-run-out feature CV passed.")
    print(
        f"  Runs/folds: {len(run_ids)}; pooled held-out trials: {len(labels)}; "
        f"uniform-prior PR-AUC={auprc['auc']:.6f}."
    )
    return {
        "posterior": posterior,
        "labels": labels,
        "active_mask": active_mask,
        "fold_audit": fold_audit,
        "selected_features_by_fold": selected_features,
        "uniform_prior_pr": auprc,
        "balanced_pool_mask": balanced_pool_mask,
        "balanced_pool_audit": balanced_pool["audit"],
    }


def compute_source_auxiliary_threshold_metrics(labels, posterior):
    """Reproduce the source's 0.2--0.8 operating-threshold summary."""
    labels = np.asarray(labels).reshape(-1).astype(int)
    posterior = np.asarray(posterior, dtype=float).reshape(-1)
    if labels.size != posterior.size or not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError("Binary labels and finite aligned posterior values are required.")
    if not np.isfinite(posterior).all():
        raise ValueError("posterior must be finite.")
    thresholds = np.linspace(0.2, 0.8, 121)
    tpr = np.empty(len(thresholds))
    fpr = np.empty(len(thresholds))
    for index, threshold in enumerate(thresholds):
        predicted = posterior >= threshold
        tpr[index] = np.mean(predicted[labels == 1])
        fpr[index] = np.mean(predicted[labels == 0])
    tnr = 1.0 - fpr
    difference = np.abs(tpr - tnr)
    candidate_indices = np.flatnonzero(difference <= difference.min() + 1e-12)
    balanced_accuracy = 0.5 * (tpr + tnr)
    best_index = candidate_indices[np.argmax(balanced_accuracy[candidate_indices])]
    threshold = float(thresholds[best_index])
    predicted = posterior >= threshold
    tpr_selected = float(np.mean(predicted[labels == 1]))
    tnr_selected = float(np.mean(~predicted[labels == 0]))
    accuracy = float(np.mean(predicted == labels))
    return {
        "threshold": threshold,
        "tpr": tpr_selected,
        "tnr": tnr_selected,
        "accuracy": accuracy,
    }


def prune_trials_mask(labels, posterior, threshold=0.5, pct_remove=0.05, active_mask=None):
    """Apply the source's cumulative posterior-based pruning rule once."""
    labels = np.asarray(labels).reshape(-1).astype(int)
    posterior = np.asarray(posterior, dtype=float).reshape(-1)
    if labels.size != posterior.size or not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError("Binary labels and aligned posterior are required for pruning.")
    if not np.isfinite(posterior).all():
        raise ValueError("Pruning posterior values must be finite.")
    if not 0 < pct_remove <= 1:
        raise ValueError(f"pct_remove must be in (0, 1], got {pct_remove}.")
    if active_mask is None:
        active_mask = np.ones(len(labels), dtype=bool)
    else:
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if active_mask.size != labels.size:
            raise ValueError("active_mask must align with labels.")

    prune_mask = np.ones(len(labels), dtype=bool)
    audit_rows = []
    for class_label in (0, 1):
        class_indices = np.flatnonzero(active_mask & (labels == class_label))
        n_class = len(class_indices)
        if n_class == 0:
            continue
        n_mis = int(np.ceil(pct_remove * n_class))
        n_threshold = int(np.ceil(pct_remove * n_class))
        class_posterior = posterior[class_indices]
        if class_label == 1:
            misclassified = class_indices[class_posterior < 0.2]
        else:
            misclassified = class_indices[class_posterior > 0.8]
        if len(misclassified):
            confidence = np.abs(posterior[misclassified] - threshold)
            order = np.argsort(-confidence, kind="stable")
            high_confidence_drop = misclassified[order[:min(n_mis, len(order))]]
        else:
            high_confidence_drop = np.array([], dtype=int)
        distance_to_threshold = np.abs(class_posterior - threshold)
        near_order = np.argsort(distance_to_threshold, kind="stable")
        near_threshold_drop = class_indices[near_order[:min(n_threshold, len(near_order))]]
        drop_indices = np.unique(np.concatenate((high_confidence_drop, near_threshold_drop)))
        prune_mask[drop_indices] = False
        audit_rows.append({
            "class_label": class_label,
            "n_active": n_class,
            "n_high_confidence_error_candidates": int(len(misclassified)),
            "n_high_confidence_error_removed": int(len(high_confidence_drop)),
            "n_near_threshold_removed": int(len(near_threshold_drop)),
            "n_unique_removed": int(len(drop_indices)),
        })
    next_active_mask = active_mask & prune_mask
    if next_active_mask.sum() >= active_mask.sum():
        raise RuntimeError("Pruning did not remove any active trials.")
    return {
        "prune_mask": prune_mask,
        "next_active_mask": next_active_mask,
        "audit": pd.DataFrame(audit_rows),
    }


def run_iterative_pruning_feature_cv(
    model_input,
    time,
    n_iterations=20,
    random_seed=20260812,
    stop_on_untrainable=True,
    **cv_kwargs,
):
    """Run source-style cumulative pruning around fold-local feature CV."""
    if not isinstance(n_iterations, (int, np.integer)) or n_iterations < 1:
        raise ValueError(f"n_iterations must be a positive integer, got {n_iterations!r}.")
    labels = np.asarray(model_input["labels"]).reshape(-1).astype(int)
    active_mask = np.ones(len(labels), dtype=bool)
    iteration_masks = []
    iteration_posteriors = []
    history_rows = []
    pruning_audits = []
    stop_reason = None
    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} of {n_iterations} ---")
        iteration_masks.append(active_mask.copy())
        try:
            cv_result = run_leave_one_run_out_feature_cv(
                model_input,
                time,
                active_mask=active_mask,
                random_seed=int(random_seed + iteration - 1),
                **cv_kwargs,
            )
        except ValueError as exc:
            if not stop_on_untrainable or "is untrainable after balancing" not in str(exc):
                raise
            stop_reason = str(exc)
            iteration_masks.pop()
            print(f"Stopping before iteration {iteration}: {stop_reason}")
            break
        auxiliary = compute_source_auxiliary_threshold_metrics(
            labels, cv_result["posterior"]
        )
        pruning = prune_trials_mask(
            labels,
            cv_result["posterior"],
            threshold=0.5,
            pct_remove=0.05,
            active_mask=active_mask,
        )
        history_rows.append({
            "iteration": iteration,
            "n_active_trials": int(active_mask.sum()),
            "auprc": float(cv_result["uniform_prior_pr"]["auc"]),
            **auxiliary,
            "n_removed": int(active_mask.sum() - pruning["next_active_mask"].sum()),
            "n_remaining_after_prune": int(pruning["next_active_mask"].sum()),
        })
        audit = pruning["audit"].copy()
        audit.insert(0, "iteration", iteration)
        pruning_audits.append(audit)
        iteration_posteriors.append(cv_result["posterior"].copy())
        active_mask = pruning["next_active_mask"]
    history = pd.DataFrame(history_rows)
    if history.empty:
        raise RuntimeError("No valid pruning iteration completed before the model became untrainable.")
    best_index = int(np.argmax(history["auprc"].to_numpy()))
    best_iteration = best_index + 1
    masks = np.column_stack(iteration_masks)
    best_mask = masks[:, best_index]
    print("Iterative pruning feature CV passed.")
    print(
        f"  Best iteration: {best_iteration}; AUPRC={history.loc[best_index, 'auprc']:.6f}; "
        f"clean trials={int(best_mask.sum())}."
    )
    if stop_reason is not None:
        print(
            f"  Completed {len(history)} of requested {n_iterations} iterations; "
            "stopped because a later fold had no balanced training data."
        )
    return {
        "history": history,
        "iteration_masks": masks,
        "iteration_posteriors": np.column_stack(iteration_posteriors),
        "pruning_audit": pd.concat(pruning_audits, ignore_index=True),
        "best_iteration": best_iteration,
        "best_mask": best_mask,
        "best_posterior": iteration_posteriors[best_index],
        "best_trial_table": model_input["trial_table"].loc[best_mask].reset_index(drop=True),
        "requested_iterations": int(n_iterations),
        "completed_iterations": int(len(history)),
        "stop_reason": stop_reason,
    }


def fit_final_clean_feature_reference(
    clean_epochs_time_channels_trials,
    clean_labels,
    time,
    feature_start_s=0.2,
    feature_stop_s=None,
    resample_ratio=8,
    n_xdawn_components=2,
    n_selected_features=30,
):
    """Fit the frozen Session 5 feature reference on a selected clean dataset.

    This intentionally follows the original final-refit convention: all trials
    retained by the best pruning mask are used, with no additional final
    balancing. It fits xDAWN, z-scoring, r2 ranking, and top feature indices;
    it does not fit a final classifier because this post-hoc workflow tracks
    features rather than deploys a new online decoder.
    """
    clean_epochs = np.asarray(clean_epochs_time_channels_trials, dtype=float)
    clean_labels = np.asarray(clean_labels).reshape(-1).astype(int)
    time = np.asarray(time, dtype=float)
    if clean_epochs.ndim != 3 or clean_epochs.shape[2] != clean_labels.size:
        raise ValueError(
            "Clean epochs must be time x channels x trials and align with labels; "
            f"got epochs={clean_epochs.shape}, labels={clean_labels.shape}."
        )
    if time.size != clean_epochs.shape[0]:
        raise ValueError("time length does not match clean epoch time dimension.")
    if not np.array_equal(np.unique(clean_labels), np.array([0, 1])):
        raise ValueError("Clean final reference requires both binary classes.")
    window_indices = select_poststimulus_feature_window(time, feature_start_s, feature_stop_s)
    xdawn_fit = fit_xdawn_fold(
        clean_epochs,
        clean_labels,
        window_indices,
        n_components=n_xdawn_components,
    )
    filters = xdawn_fit["positive_class_filters_components_by_channels"]
    projected = apply_xdawn_filters(clean_epochs, filters)
    features, resampled_indices = _stride_resample_and_flatten(
        projected, window_indices, resample_ratio
    )
    means, stds = _fit_zscore_training_features(features)
    normalized_features = _apply_zscore_features(features, means, stds)
    r2 = compute_binary_feature_r2(normalized_features, clean_labels)
    if n_selected_features < 1 or n_selected_features > len(r2):
        raise ValueError(
            f"n_selected_features must be 1..{len(r2)}, got {n_selected_features}."
        )
    selected_indices = np.argsort(-r2, kind="stable")[:n_selected_features]
    n_resampled_time = len(resampled_indices)
    selected_coordinates = pd.DataFrame({
        "rank": np.arange(1, n_selected_features + 1),
        "feature_index_zero_based": selected_indices,
        "component": (selected_indices // n_resampled_time) + 1,
        "time_index": resampled_indices[selected_indices % n_resampled_time],
        "time_s": time[resampled_indices[selected_indices % n_resampled_time]],
        "r2_clean_training": r2[selected_indices],
    })
    class_counts = np.bincount(clean_labels, minlength=2)
    print("Final clean-trial Session 5 feature reference passed.")
    print(
        f"  Clean refit trials: {clean_epochs.shape[2]}; "
        f"no={class_counts[0]}, distractor={class_counts[1]} (no final balancing)."
    )
    print(
        f"  Candidate features: {len(r2)}; frozen top features: "
        f"{len(selected_indices)}."
    )
    return {
        "xdawn_filters_components_by_channels": filters,
        "xdawn_fit": xdawn_fit,
        "feature_window_indices": window_indices,
        "resampled_indices": resampled_indices,
        "resampled_time_s": time[resampled_indices],
        "normalization_means": means,
        "normalization_stds": stds,
        "r2_clean_training": r2,
        "selected_indices_zero_based": selected_indices,
        "selected_coordinates": selected_coordinates,
        "clean_features_normalized": normalized_features,
        "clean_labels": clean_labels,
        "settings": {
            "feature_start_s": float(feature_start_s),
            "feature_stop_s": float(time[window_indices[-1]]),
            "resample_ratio": int(resample_ratio),
            "n_xdawn_components": int(n_xdawn_components),
            "n_selected_features": int(n_selected_features),
            "final_balancing": "none",
        },
    }


def apply_frozen_feature_reference_and_compute_r2(
    epochs_time_channels_trials,
    labels,
    feature_reference,
):
    """Apply a final Session 5 reference without refitting any transform.

    Returns r2 for both all candidate features and the fixed selected features.
    This is appropriate for the full unpruned Session 5 descriptive reference
    and, later, for independently collected decoding-session data.
    """
    required = {
        "xdawn_filters_components_by_channels",
        "resampled_indices",
        "normalization_means",
        "normalization_stds",
        "selected_indices_zero_based",
        "selected_coordinates",
    }
    if not isinstance(feature_reference, dict) or not required.issubset(feature_reference):
        raise ValueError(f"feature_reference must contain {sorted(required)}.")
    epochs = np.asarray(epochs_time_channels_trials, dtype=float)
    labels = np.asarray(labels).reshape(-1).astype(int)
    if epochs.ndim != 3 or epochs.shape[2] != labels.size:
        raise ValueError(
            "epochs must be time x channels x trials and align with labels; "
            f"got epochs={epochs.shape}, labels={labels.shape}."
        )
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError("Evaluation r2 requires both binary classes.")
    filters = np.asarray(feature_reference["xdawn_filters_components_by_channels"], dtype=float)
    resampled_indices = np.asarray(feature_reference["resampled_indices"], dtype=int)
    selected_indices = np.asarray(feature_reference["selected_indices_zero_based"], dtype=int)
    if resampled_indices.min() < 0 or resampled_indices.max() >= epochs.shape[0]:
        raise ValueError("Frozen resampled indices are outside the evaluation epoch time axis.")
    projected = apply_xdawn_filters(epochs, filters)
    resampled = projected[resampled_indices, :, :]
    features = np.reshape(
        resampled,
        (resampled.shape[0] * resampled.shape[1], resampled.shape[2]),
        order="F",
    )
    normalized_features = _apply_zscore_features(
        features,
        feature_reference["normalization_means"],
        feature_reference["normalization_stds"],
    )
    if selected_indices.min() < 0 or selected_indices.max() >= normalized_features.shape[0]:
        raise ValueError("Frozen selected feature indices are outside evaluation features.")
    candidate_r2 = compute_binary_feature_r2(normalized_features, labels)
    selected_r2 = candidate_r2[selected_indices]
    feature_r2_table = feature_reference["selected_coordinates"].copy()
    feature_r2_table["r2_evaluation"] = selected_r2
    class_counts = np.bincount(labels, minlength=2)
    print("Frozen Session 5 feature reference application passed.")
    print(
        f"  Evaluation trials: {len(labels)}; no={class_counts[0]}, "
        f"distractor={class_counts[1]}; no learned transform refit."
    )
    print(
        f"  Applied features: {normalized_features.shape[0]} candidates; "
        f"reported fixed features: {len(selected_indices)}."
    )
    return {
        "features_normalized": normalized_features,
        "candidate_r2": candidate_r2,
        "selected_r2": selected_r2,
        "feature_r2_table": feature_r2_table,
        "labels": labels,
    }


def _expected_evaluation_run_count(subject_id, session_id, evaluation_task):
    """Return documented expected non-practice run count for one evaluation cell."""
    if evaluation_task == "session1_training_pre":
        if int(session_id) != 1:
            raise ValueError("session1_training_pre is defined only for Session 1.")
        return 7 if str(subject_id).lower() == "e30" else SESSION1_TRAINING_RUNS
    if evaluation_task == "decoding":
        exception = BCI_SESSION_EXCEPTIONS.get((str(subject_id).lower(), int(session_id)))
        if exception is not None and "real_runs" in exception:
            return int(exception["real_runs"])
        return int(EXPECTED_REAL_RUNS_BY_SESSION[int(session_id)])
    raise ValueError(f"Unsupported evaluation_task: {evaluation_task!r}.")


def _get_nonpractice_task_run_files(subject_id, session_id, task, project_root):
    """Resolve complete non-practice run inputs and retain incomplete-run issues."""
    subject_id = str(subject_id).lower().strip()
    subject_dir = Path(project_root) / subject_id
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
    session_dirs = sorted(
        path for path in subject_dir.iterdir()
        if path.is_dir() and path.name.startswith(f"{subject_id}_")
    )
    if len(session_dirs) != 5:
        raise ValueError(f"{subject_id}: expected 5 session folders, found {len(session_dirs)}.")
    session_dir = session_dirs[int(session_id) - 1]
    run_dirs = sorted(
        path for path in session_dir.iterdir()
        if path.is_dir() and path.name.endswith(f"_{task}") and not path.name.endswith(f"_{task}_practice")
    )
    files, issues = [], []
    for run_id, run_dir in enumerate(run_dirs, start=1):
        gdf_matches = sorted(run_dir.glob("*.gdf"))
        trigger_matches = sorted(run_dir.glob("*.triggers.txt"))
        analysis_matches = sorted(run_dir.glob("*.analysis.txt"))
        missing = []
        if len(gdf_matches) != 1:
            missing.append(f"GDF={len(gdf_matches)}")
        if len(trigger_matches) != 1:
            missing.append(f"trigger={len(trigger_matches)}")
        if len(analysis_matches) != 1:
            missing.append(f"analysis={len(analysis_matches)}")
        if missing:
            issues.append({
                "run_id": run_id,
                "run_dir": str(run_dir),
                "issue": (
                    f"incomplete run files: expected exactly one GDF, trigger, and analysis "
                    f"file; {', '.join(missing)}"
                ),
            })
            continue
        files.append((run_id, gdf_matches[0]))
    return files, issues


def _build_evaluation_run_trial_table(subject_id, session_id, run_id, gdf_path, evaluation_task):
    """Validate labels/events for one pre-training or decoding evaluation run."""
    trigger_path = gdf_path.with_suffix(".triggers.txt")
    analysis_path = gdf_path.with_suffix(".analysis.txt")
    # EEG epochs are anchored to GDF Status events.  A response timestamp equal
    # to the preceding stimulus is retained here as an explicit metadata flag,
    # because it cannot alter condition labels or EEG epoch timing.
    trigger_df = load_training_trigger_file(
        trigger_path, allow_zero_latency_response=True
    )
    if evaluation_task == "session1_training_pre":
        analysis_df = load_training_analysis_file(analysis_path)
    elif evaluation_task == "decoding":
        analysis_df = load_decoding_analysis_file(analysis_path)
    else:
        raise ValueError(f"Unsupported evaluation_task: {evaluation_task!r}.")
    stimulus_rows = trigger_df.groupby("trial", sort=True).nth(1).reset_index()
    stimulus_rows = stimulus_rows.rename(
        columns={"trigger": "stimulus_trigger", "time": "stimulus_sample"}
    )[["trial", "stimulus_trigger", "stimulus_sample"]]
    trial_table = analysis_df.rename(columns={"trial_index": "trial"}).merge(
        stimulus_rows, on="trial", how="left", validate="one_to_one"
    )
    if len(trial_table) != TRAINING_TRIALS or trial_table["stimulus_trigger"].isna().any():
        raise ValueError(f"{gdf_path}: analysis/trigger merge did not yield 60 complete trials.")
    trial_table["stimulus_trigger"] = trial_table["stimulus_trigger"].astype(int)
    trial_table["condition"] = trial_table["stimulus_trigger"].map(_CONDITION_BY_TRIGGER)
    expected_task = (trial_table["stimulus_trigger"] != 8).astype(int)
    if not np.array_equal(trial_table["task"].to_numpy(dtype=int), expected_task.to_numpy()):
        bad_trials = trial_table.loc[
            trial_table["task"].to_numpy(dtype=int) != expected_task.to_numpy(), "trial"
        ].tolist()
        raise ValueError(f"{gdf_path}: analysis task disagrees with trigger at trial(s) {bad_trials}.")
    trial_table.insert(0, "subject_id", str(subject_id).lower())
    trial_table.insert(1, "group", get_subject_group(subject_id))
    trial_table.insert(2, "evaluation_task", evaluation_task)
    trial_table.insert(3, "session_id", int(session_id))
    trial_table.insert(4, "run_id", int(run_id))
    for side in ("right", "left"):
        trial_table[f"{side}_model_include"] = trial_table["condition"].isin(
            [f"distractor_{side}", "no_distractor"]
        )
        trial_table[f"{side}_model_label"] = np.where(
            trial_table["condition"] == f"distractor_{side}", 1,
            np.where(trial_table["condition"] == "no_distractor", 0, np.nan),
        )
    status_summary = _load_and_validate_status_stimuli(gdf_path, stimulus_rows)
    trigger_time_matrix = trigger_df["time"].to_numpy(dtype=int).reshape(
        TRAINING_TRIALS, 3
    )
    zero_latency_trials = np.flatnonzero(
        trigger_time_matrix[:, 1] == trigger_time_matrix[:, 2]
    ) + 1
    counts = trial_table["condition"].value_counts()
    return trial_table, {
        "subject_id": str(subject_id).lower(),
        "group": get_subject_group(subject_id),
        "evaluation_task": evaluation_task,
        "session_id": int(session_id),
        "run_id": int(run_id),
        "gdf_path": str(gdf_path),
        "trigger_path": str(trigger_path),
        "analysis_path": str(analysis_path),
        "n_trials": int(len(trial_table)),
        "n_no_distractor": int(counts.get("no_distractor", 0)),
        "n_distractor_right": int(counts.get("distractor_right", 0)),
        "n_distractor_left": int(counts.get("distractor_left", 0)),
        "n_right_model_trials": int(trial_table["right_model_include"].sum()),
        "n_left_model_trials": int(trial_table["left_model_include"].sum()),
        "zero_latency_response_trials": tuple(zero_latency_trials.tolist()),
        **status_summary,
    }


def build_longitudinal_evaluation_manifest(subject_ids=None, project_root=PROJECT_ROOT):
    """Validate all target inputs before applying a frozen Session 5 reference.

    Target datasets are eight Session 1 training runs (pre-intervention) plus
    every non-practice decoding run in Sessions 1--5. This function performs
    no EEG filtering, epoching, or feature evaluation.
    """
    project_root = Path(project_root)
    subjects = list(EXPECTED_SUBJECTS if subject_ids is None else subject_ids)
    if not subjects:
        raise ValueError("subject_ids is empty.")
    print("LONGITUDINAL FEATURE-EVALUATION INPUT VALIDATION")
    print("Targets: Session 1 pre-intervention training + decoding Sessions 1-5.")
    manifest_rows, trial_tables, issues, skipped_runs = [], [], [], []
    task_specs = [("session1_training_pre", 1, "training")]
    task_specs.extend(("decoding", session_id, "decoding") for session_id in range(1, 6))
    for subject_id in subjects:
        subject_id = str(subject_id).lower().strip()
        for evaluation_task, session_id, folder_task in task_specs:
            run_files, file_issues = _get_nonpractice_task_run_files(
                subject_id, session_id, folder_task, project_root
            )
            expected_runs = _expected_evaluation_run_count(
                subject_id, session_id, evaluation_task
            )
            for file_issue in file_issues:
                issues.append({
                    "subject_id": subject_id,
                    "evaluation_task": evaluation_task,
                    "session_id": session_id,
                    "run_id": file_issue["run_id"],
                    "run_dir": file_issue["run_dir"],
                    "issue": file_issue["issue"],
                })
                print(
                    f"WARNING: {subject_id} {evaluation_task} Session {session_id} "
                    f"run {file_issue['run_id']}: {file_issue['issue']}."
                )
            if evaluation_task == "decoding" and len(run_files) == expected_runs + 1:
                skipped_run_id, skipped = run_files[0]
                run_files = run_files[1:]
                issues.append({
                    "subject_id": subject_id,
                    "evaluation_task": evaluation_task,
                    "session_id": session_id,
                    "expected_runs": expected_runs,
                    "found_runs": expected_runs + 1,
                    "issue": "skipped first extra non-practice-labeled run as practice-like",
                    "gdf_path": str(skipped),
                })
                print(
                    f"WARNING: {subject_id} decoding Session {session_id}: skipped "
                    f"first of {expected_runs + 1} non-practice-labeled runs as practice-like: "
                    f"{skipped.name}"
                )
            if len(run_files) != expected_runs:
                issues.append({
                    "subject_id": subject_id,
                    "evaluation_task": evaluation_task,
                    "session_id": session_id,
                    "expected_runs": expected_runs,
                    "found_runs": len(run_files),
                    "issue": "non-practice run count mismatch",
                })
                print(
                    f"WARNING: {subject_id} {evaluation_task} Session {session_id}: "
                    f"found {len(run_files)}, expected {expected_runs}."
                )
            for run_id, gdf_path in run_files:
                try:
                    trials, row = _build_evaluation_run_trial_table(
                        subject_id, session_id, run_id, gdf_path, evaluation_task
                    )
                except Exception as exc:
                    issues.append({
                        "subject_id": subject_id,
                        "evaluation_task": evaluation_task,
                        "session_id": session_id,
                        "run_id": run_id,
                        "gdf_path": str(gdf_path),
                        "issue": f"run validation failed: {exc}",
                    })
                    print(
                        f"WARNING: {subject_id} {evaluation_task} Session {session_id} "
                        f"run {run_id} failed validation: {exc}"
                    )
                    continue
                trial_tables.append(trials)
                manifest_rows.append(row)
    if not manifest_rows:
        raise RuntimeError("No longitudinal evaluation runs passed validation.")
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["group", "subject_id", "evaluation_task", "session_id", "run_id"], kind="stable"
    ).reset_index(drop=True)
    trials = pd.concat(trial_tables, ignore_index=True)
    if not (manifest["n_trials"] == TRAINING_TRIALS).all():
        raise RuntimeError("A longitudinal evaluation run does not contain 60 trials.")
    if not (manifest["status_event_alignment"] == "pass").all():
        raise RuntimeError("A longitudinal evaluation run failed GDF Status-event alignment.")
    summary = manifest.groupby(["evaluation_task", "session_id", "group"], sort=True).agg(
        n_subjects=("subject_id", "nunique"),
        n_runs=("run_id", "size"),
        n_trials=("n_trials", "sum"),
        no_distractor=("n_no_distractor", "sum"),
        distractor_right=("n_distractor_right", "sum"),
        distractor_left=("n_distractor_left", "sum"),
    ).reset_index()
    print(f"Validated evaluation runs: {len(manifest)}; trials: {len(trials)}.")
    if issues:
        print(f"Run-count issues retained explicitly: {len(issues)}.")
    else:
        print("All documented evaluation run-count expectations passed.")
    return {
        "manifest": manifest,
        "trials": trials,
        "summary": summary,
        "issues": pd.DataFrame(issues),
    }


def _load_and_validate_status_stimuli(gdf_path, trigger_stimuli):
    """Validate GDF stimulus events against a paired trigger file."""
    # MNE otherwise attempts to create a user-home configuration lock when it
    # is first imported, which is unsuitable for this read-only validation.
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - depends on analysis environment
        raise ImportError(
            "MNE is required to validate Session 5 GDF Status events. "
            "Use the Pd_AI kernel/environment."
        ) from exc

    raw = mne.io.read_raw_gdf(gdf_path, preload=False, verbose="ERROR")
    if not np.isclose(float(raw.info["sfreq"]), FS):
        raise ValueError(
            f"{gdf_path}: expected {FS} Hz, found {raw.info['sfreq']} Hz."
        )
    if "Status" not in raw.ch_names:
        raise ValueError(f"{gdf_path}: missing required Status channel.")

    events = mne.find_events(
        raw, stim_channel="Status", shortest_event=1, verbose="ERROR"
    )
    task_events = events[np.isin(events[:, 2], sorted(_CORE_TASK_CODES))]
    stimulus_events = task_events[np.isin(task_events[:, 2], list(STIMULUS_CODES))]
    if len(stimulus_events) != TRAINING_TRIALS:
        raise ValueError(
            f"{gdf_path}: expected {TRAINING_TRIALS} Status stimulus events, "
            f"found {len(stimulus_events)}."
        )

    status_codes = stimulus_events[:, 2].astype(int)
    trigger_codes = trigger_stimuli["stimulus_trigger"].to_numpy(dtype=int)
    if not np.array_equal(status_codes, trigger_codes):
        mismatch = np.flatnonzero(status_codes != trigger_codes)[0]
        raise ValueError(
            f"{gdf_path}: Status/trigger condition mismatch at trial {mismatch + 1}: "
            f"Status={status_codes[mismatch]}, trigger file={trigger_codes[mismatch]}."
        )

    status_samples = stimulus_events[:, 0].astype(float)
    trigger_samples = trigger_stimuli["stimulus_sample"].to_numpy(dtype=float)
    time_slope, time_intercept = np.polyfit(status_samples, trigger_samples, deg=1)
    time_residuals = trigger_samples - (time_slope * status_samples + time_intercept)
    max_time_residual = float(np.max(np.abs(time_residuals)))
    # Status events, not trigger-text timestamps, define EEG epoch onsets.
    # The text clock is retained only as a cross-check.  We allow <=10 samples
    # (10 ms in that clock) of residual jitter after requiring an exact 60-trial
    # condition-order match above; larger disagreement indicates a broken map.
    if time_slope <= 0 or max_time_residual > 10.0:
        raise ValueError(
            f"{gdf_path}: Status and trigger-file stimulus times are not aligned "
            "by a stable positive linear clock conversion; "
            f"slope={time_slope:.8g}, max residual={max_time_residual:.3f} samples."
        )

    pre_samples = int(round(abs(DEFAULT_EPOCH_TMIN) * FS))
    post_samples = int(round(DEFAULT_EPOCH_TMAX * FS))
    epoch_start = status_samples - pre_samples
    epoch_stop = status_samples + post_samples
    first_sample = int(raw.first_samp)
    last_sample = first_sample + int(raw.n_times) - 1
    if (epoch_start < first_sample).any() or (epoch_stop > last_sample).any():
        bad_trials = np.flatnonzero(
            (epoch_start < first_sample) | (epoch_stop > last_sample)
        ) + 1
        raise ValueError(
            f"{gdf_path}: requested {DEFAULT_EPOCH_TMIN:g} to "
            f"{DEFAULT_EPOCH_TMAX:g} s epochs exceed recording bounds for "
            f"trial(s) {bad_trials.tolist()}."
        )

    return {
        "n_status_stimuli": int(len(stimulus_events)),
        "status_to_trigger_time_slope": float(time_slope),
        "status_to_trigger_time_intercept": float(time_intercept),
        "max_trigger_time_residual_samples": max_time_residual,
        "status_event_alignment": "pass",
        "sfreq_hz": float(raw.info["sfreq"]),
        "n_raw_samples": int(raw.n_times),
        "n_raw_channels": int(len(raw.ch_names)),
        "channel_names": tuple(raw.ch_names),
    }


def _make_run_trial_table(subject_id, session_id, run_id, gdf_path):
    """Load one validated Session 5 training run and construct binary labels."""
    trigger_path = gdf_path.with_suffix(".triggers.txt")
    analysis_path = gdf_path.with_suffix(".analysis.txt")
    trigger_df = load_training_trigger_file(trigger_path)
    analysis_df = load_training_analysis_file(analysis_path)

    stimulus_rows = trigger_df.groupby("trial", sort=True).nth(1).reset_index()
    stimulus_rows = stimulus_rows.rename(
        columns={"trigger": "stimulus_trigger", "time": "stimulus_sample"}
    )[["trial", "stimulus_trigger", "stimulus_sample"]]
    if stimulus_rows["trial"].tolist() != list(range(1, TRAINING_TRIALS + 1)):
        raise ValueError(f"{trigger_path}: stimulus trials are not consecutive 1..60.")

    trial_table = analysis_df.rename(columns={"trial_index": "trial"}).merge(
        stimulus_rows, on="trial", how="left", validate="one_to_one"
    )
    if len(trial_table) != TRAINING_TRIALS:
        raise ValueError(
            f"{gdf_path}: analysis/trigger merge returned {len(trial_table)} trials, "
            f"expected {TRAINING_TRIALS}."
        )

    trial_table["condition"] = trial_table["stimulus_trigger"].map(_CONDITION_BY_TRIGGER)
    if trial_table["condition"].isna().any():
        bad = trial_table.loc[trial_table["condition"].isna(), "stimulus_trigger"].unique()
        raise ValueError(f"{gdf_path}: unsupported stimulus trigger(s): {bad.tolist()}.")

    expected_task = (trial_table["stimulus_trigger"] != 8).astype(int)
    if not np.array_equal(trial_table["task"].to_numpy(dtype=int), expected_task.to_numpy()):
        bad = trial_table.index[trial_table["task"].to_numpy(dtype=int) != expected_task.to_numpy()] + 1
        raise ValueError(
            f"{gdf_path}: analysis task column disagrees with stimulus triggers "
            f"at trial(s) {bad.tolist()}."
        )

    trial_table.insert(0, "subject_id", str(subject_id).lower())
    trial_table.insert(1, "group", get_subject_group(subject_id))
    trial_table.insert(2, "session_id", int(session_id))
    trial_table.insert(3, "run_id", int(run_id))
    trial_table["right_model_include"] = trial_table["condition"].isin(
        ["distractor_right", "no_distractor"]
    )
    trial_table["right_model_label"] = np.where(
        trial_table["condition"] == "distractor_right", 1,
        np.where(trial_table["condition"] == "no_distractor", 0, np.nan),
    )
    trial_table["left_model_include"] = trial_table["condition"].isin(
        ["distractor_left", "no_distractor"]
    )
    trial_table["left_model_label"] = np.where(
        trial_table["condition"] == "distractor_left", 1,
        np.where(trial_table["condition"] == "no_distractor", 0, np.nan),
    )

    status_summary = _load_and_validate_status_stimuli(gdf_path, stimulus_rows)
    condition_counts = trial_table["condition"].value_counts()
    manifest_row = {
        "subject_id": str(subject_id).lower(),
        "group": get_subject_group(subject_id),
        "session_id": int(session_id),
        "run_id": int(run_id),
        "gdf_path": str(gdf_path),
        "trigger_path": str(trigger_path),
        "analysis_path": str(analysis_path),
        "n_trials": int(len(trial_table)),
        "n_no_distractor": int(condition_counts.get("no_distractor", 0)),
        "n_distractor_right": int(condition_counts.get("distractor_right", 0)),
        "n_distractor_left": int(condition_counts.get("distractor_left", 0)),
        "n_right_model_trials": int(trial_table["right_model_include"].sum()),
        "n_left_model_trials": int(trial_table["left_model_include"].sum()),
        **status_summary,
    }
    return trial_table, manifest_row


def build_session5_training_model_manifest(
    subject_ids=None,
    project_root=PROJECT_ROOT,
):
    """Build validated Session 5 training inputs for new lateralized models.

    Returns a run-level manifest, one trial-level table, and separate binary
    right/no and left/no trial tables. This function performs no filtering or
    model fitting.
    """
    project_root = Path(project_root)
    subjects = list(EXPECTED_SUBJECTS if subject_ids is None else subject_ids)
    if not subjects:
        raise ValueError("subject_ids is empty; at least one participant is required.")

    print("SESSION 5 LATERALIZED MODEL INPUT VALIDATION")
    print("No preprocessing or model fitting is performed in this step.")
    print(f"Participants requested: {len(subjects)}")
    print(f"Expected non-practice Session 5 training runs/person: {SESSION5_TRAINING_RUNS}")

    trial_tables = []
    manifest_rows = []
    for subject_id in subjects:
        subject_id = str(subject_id).lower().strip()
        gdf_files = _get_training_run_gdf_files_for_session(
            subject_id, session=5, project_root=project_root, allow_incomplete=False
        )
        if len(gdf_files) != SESSION5_TRAINING_RUNS:
            raise ValueError(
                f"{subject_id}: expected {SESSION5_TRAINING_RUNS} Session 5 training "
                f"runs, got {len(gdf_files)}."
            )
        for run_id, gdf_path in enumerate(gdf_files, start=1):
            run_trials, manifest_row = _make_run_trial_table(
                subject_id, session_id=5, run_id=run_id, gdf_path=gdf_path
            )
            trial_tables.append(run_trials)
            manifest_rows.append(manifest_row)

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["group", "subject_id", "run_id"], kind="stable"
    ).reset_index(drop=True)
    trials = pd.concat(trial_tables, ignore_index=True)
    right_trials = trials.loc[trials["right_model_include"]].copy()
    left_trials = trials.loc[trials["left_model_include"]].copy()

    expected_rows = len(subjects) * SESSION5_TRAINING_RUNS
    if len(manifest) != expected_rows:
        raise RuntimeError(
            f"Manifest has {len(manifest)} runs; expected {expected_rows}."
        )
    if not (manifest["n_trials"] == TRAINING_TRIALS).all():
        raise RuntimeError("Manifest contains a run that does not have 60 validated trials.")
    if not (manifest["status_event_alignment"] == "pass").all():
        raise RuntimeError("At least one run failed GDF Status/trigger alignment.")

    summary = manifest.groupby("group", sort=True).agg(
        n_subjects=("subject_id", "nunique"),
        n_runs=("run_id", "size"),
        n_trials=("n_trials", "sum"),
        no_distractor=("n_no_distractor", "sum"),
        distractor_right=("n_distractor_right", "sum"),
        distractor_left=("n_distractor_left", "sum"),
    )
    print("Validation passed: all requested Session 5 training runs have aligned files and Status events.")
    print("\nGroup-level condition counts:")
    print(summary.to_string())
    print(
        f"\nBinary tables: right/no={len(right_trials)} trials; "
        f"left/no={len(left_trials)} trials."
    )
    return {
        "manifest": manifest,
        "trials": trials,
        "right_trials": right_trials,
        "left_trials": left_trials,
        "group_summary": summary.reset_index(),
    }


def build_session5_training_post_evaluation_manifest(subject_ids=None, project_root=PROJECT_ROOT):
    """Return validated full Session-5 training runs as a post-training evaluation set.

    This reuses the strict four-run Session-5 training manifest used to build
    the frozen references, but labels the data as an evaluation target.  It
    does not retain or apply the clean-trial mask: all valid trials are kept.
    """
    inputs = build_session5_training_model_manifest(
        subject_ids=subject_ids, project_root=project_root
    )
    manifest = inputs["manifest"].copy()
    trials = inputs["trials"].copy()
    manifest.insert(2, "evaluation_task", "session5_training_post")
    trials.insert(2, "evaluation_task", "session5_training_post")
    summary = manifest.groupby(["evaluation_task", "session_id", "group"], sort=True).agg(
        n_subjects=("subject_id", "nunique"),
        n_runs=("run_id", "size"),
        n_trials=("n_trials", "sum"),
        no_distractor=("n_no_distractor", "sum"),
        distractor_right=("n_distractor_right", "sum"),
        distractor_left=("n_distractor_left", "sum"),
    ).reset_index()
    if not manifest["evaluation_task"].eq("session5_training_post").all():
        raise RuntimeError("Session-5 post-training manifest task labeling failed.")
    if not trials["evaluation_task"].eq("session5_training_post").all():
        raise RuntimeError("Session-5 post-training trial-table task labeling failed.")
    print("Session-5 post-training evaluation manifest passed.")
    print("  Full valid trials retained; clean-trial selection is not applied during evaluation.")
    return {"manifest": manifest, "trials": trials, "summary": summary, "issues": pd.DataFrame()}


def build_session5_decoding_model_manifest(subject_ids=None, project_root=PROJECT_ROOT):
    """Validate complete Session-5 decoding inputs for new exploratory references.

    This is deliberately distinct from the Session-5 training reference
    manifest. It uses only full, non-practice decoding runs and their decoding
    task labels. Each participant must have every documented Session-5 run;
    partial source data are not silently used to select a new feature set.
    """
    project_root = Path(project_root)
    subjects = [str(subject).lower().strip() for subject in (
        EXPECTED_SUBJECTS if subject_ids is None else subject_ids
    )]
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("subject_ids must contain one or more unique participant IDs.")
    unknown = sorted(set(subjects).difference(EXPECTED_SUBJECTS))
    if unknown:
        raise ValueError(f"Unknown participant ID(s): {unknown}.")
    manifest_rows, trial_tables, issues, skipped_runs = [], [], [], []
    print("SESSION-5 DECODING REFERENCE INPUT VALIDATION")
    print("Source data: complete, non-practice Session-5 decoding runs only.")
    for subject_id in subjects:
        run_files, file_issues = _get_nonpractice_task_run_files(
            subject_id, session_id=5, task="decoding", project_root=project_root
        )
        expected_runs = _expected_evaluation_run_count(subject_id, 5, "decoding")
        if len(run_files) == expected_runs + 1:
            skipped_run_id, skipped = run_files[0]
            run_files = run_files[1:]
            skipped_runs.append({
                "subject_id": subject_id, "run_id": skipped_run_id,
                "issue": "skipped first extra non-practice-labeled run as practice-like",
                "gdf_path": str(skipped),
            })
        for issue in file_issues:
            issues.append({"subject_id": subject_id, **issue})
        if len(run_files) != expected_runs:
            issues.append({
                "subject_id": subject_id, "run_id": pd.NA,
                "issue": "non-practice run count mismatch",
                "expected_runs": expected_runs, "found_runs": len(run_files),
            })
            continue
        for run_id, gdf_path in run_files:
            try:
                trials, row = _build_evaluation_run_trial_table(
                    subject_id, session_id=5, run_id=run_id, gdf_path=gdf_path,
                    evaluation_task="decoding",
                )
                row["reference_source"] = "session5_decoding"
                trials["reference_source"] = "session5_decoding"
                manifest_rows.append(row)
                trial_tables.append(trials)
            except Exception as exc:
                issues.append({
                    "subject_id": subject_id, "run_id": run_id, "gdf_path": str(gdf_path),
                    "issue": f"run validation failed: {exc}",
                })
    if issues:
        issue_table = pd.DataFrame(issues)
        raise ValueError(
            "Session-5 decoding reference manifest is incomplete; no model selection may proceed. "
            f"Issues: {issue_table.to_dict('records')}"
        )
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["group", "subject_id", "run_id"], kind="stable"
    ).reset_index(drop=True)
    trials = pd.concat(trial_tables, ignore_index=True)
    expected_total_runs = sum(
        _expected_evaluation_run_count(subject_id, 5, "decoding") for subject_id in subjects
    )
    if len(manifest) != expected_total_runs or not manifest["n_trials"].eq(TRAINING_TRIALS).all():
        raise RuntimeError("Session-5 decoding manifest has an invalid run or trial count.")
    if not manifest["status_event_alignment"].eq("pass").all():
        raise RuntimeError("A Session-5 decoding source run failed Status-event alignment.")
    for side in ("right", "left"):
        include = trials[f"{side}_model_include"]
        labels = trials.loc[include, f"{side}_model_label"]
        if not set(labels.unique()).issubset({0, 1}) or set(labels.unique()) != {0, 1}:
            raise RuntimeError(f"{side}: Session-5 decoding manifest lacks both binary classes.")
    summary = manifest.groupby("group", sort=True).agg(
        n_subjects=("subject_id", "nunique"), n_runs=("run_id", "size"),
        n_trials=("n_trials", "sum"), no_distractor=("n_no_distractor", "sum"),
        distractor_right=("n_distractor_right", "sum"),
        distractor_left=("n_distractor_left", "sum"),
    ).reset_index()
    print("Session-5 decoding reference manifest passed.")
    print(f"  Participants: {len(subjects)}; validated runs: {len(manifest)}; trials: {len(trials)}.")
    if skipped_runs:
        print(f"  Documented practice-like extra runs excluded: {len(skipped_runs)}.")
    print(summary.to_string(index=False))
    return {
        "manifest": manifest,
        "trials": trials,
        "right_trials": trials.loc[trials["right_model_include"]].copy(),
        "left_trials": trials.loc[trials["left_model_include"]].copy(),
        "group_summary": summary,
        "issues": pd.DataFrame(),
        "skipped_runs": pd.DataFrame(skipped_runs),
    }


def preprocess_session5_decoding_subject(subject_id, session5_decoding_inputs):
    """Preprocess one validated Session-5 decoding source dataset for new models.

    Applies scalp-channel selection, zero-phase 0.1--20 Hz filtering,
    Status-anchored stimulus epochs, and per-trial -200--0 ms baseline
    correction. It does not fit xDAWN, normalization, feature ranking,
    balancing, pruning, or a classifier.
    """
    if not isinstance(session5_decoding_inputs, dict) or not {"manifest", "trials"}.issubset(session5_decoding_inputs):
        raise ValueError("session5_decoding_inputs must contain validated 'manifest' and 'trials' tables.")
    subject_id = str(subject_id).lower().strip()
    manifest = session5_decoding_inputs["manifest"]
    trials = session5_decoding_inputs["trials"]
    subject_manifest = manifest.loc[manifest["subject_id"].astype(str).str.lower().eq(subject_id)]
    subject_trials = trials.loc[trials["subject_id"].astype(str).str.lower().eq(subject_id)]
    if len(subject_manifest) != _expected_evaluation_run_count(subject_id, 5, "decoding"):
        raise ValueError(f"{subject_id}: Session-5 decoding manifest does not contain its complete expected run set.")
    if subject_trials.empty or not subject_manifest["evaluation_task"].eq("decoding").all():
        raise ValueError(f"{subject_id}: source manifest must contain Session-5 decoding runs only.")
    if not subject_manifest["session_id"].eq(5).all() or not subject_trials["session_id"].eq(5).all():
        raise ValueError(f"{subject_id}: Session-5 decoding preprocessing received another session.")
    if not subject_trials["evaluation_task"].eq("decoding").all():
        raise ValueError(f"{subject_id}: trial table contains a non-decoding task.")
    print(f"SESSION-5 DECODING MODEL PREPROCESSING: {subject_id}")
    result = preprocess_longitudinal_evaluation_subject(subject_id, session5_decoding_inputs)
    expected_trials = len(subject_manifest) * TRAINING_TRIALS
    if result["epochs_time_channels_trials"].shape[2] != expected_trials:
        raise RuntimeError(
            f"{subject_id}: expected {expected_trials} full decoding epochs, "
            f"got {result['epochs_time_channels_trials'].shape[2]}."
        )
    if not result["run_summary"]["session_id"].eq(5).all() or not result["run_summary"]["evaluation_task"].eq("decoding").all():
        raise RuntimeError(f"{subject_id}: preprocessing output task/session labels changed unexpectedly.")
    print(
        f"{subject_id}: Session-5 decoding preprocessing passed: "
        f"{len(result['run_summary'])} runs, {expected_trials} epochs, "
        f"{len(result['eeg_labels'])} scalp EEG channels."
    )
    return result


def validate_session5_analysis_channel_layout(manifest):
    """Verify one stable, label-derived scalp-EEG layout across Session 5 runs."""
    required = {"gdf_path", "channel_names"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            "Session 5 manifest is missing columns required for channel validation: "
            f"{sorted(missing)}. Rebuild the manifest first."
        )
    if manifest.empty:
        raise ValueError("Session 5 manifest is empty.")

    raw_layouts = manifest["channel_names"].map(tuple)
    reference_raw_layout = raw_layouts.iloc[0]
    mismatch = manifest.loc[raw_layouts != reference_raw_layout, "gdf_path"]
    if not mismatch.empty:
        raise ValueError(
            "Raw GDF channel label/order differs from the first Session 5 run. "
            f"First mismatching file: {mismatch.iloc[0]}"
        )
    analysis_labels, excluded, status_labels = select_analysis_eeg_channels(
        list(reference_raw_layout)
    )
    if len(analysis_labels) != 61:
        raise ValueError(
            "Expected 61 scalp EEG channels after excluding M1, M2, EOG, sens7, "
            f"sens8, and Status; found {len(analysis_labels)}: {analysis_labels}."
        )
    expected_excluded = {"M1", "M2", "EOG", "sens7", "sens8"}
    if set(excluded) != expected_excluded or status_labels != ["Status"]:
        raise ValueError(
            "Unexpected Session 5 non-analysis channel labels. Found excluded "
            f"{excluded}; Status labels {status_labels}."
        )
    print("Session 5 channel-layout validation passed.")
    print(f"  Raw layout: {len(reference_raw_layout)} channels, identical in {len(manifest)} runs.")
    print(f"  Analysis layout: {len(analysis_labels)} scalp EEG channels.")
    print(f"  Excluded: {excluded + status_labels}")
    return {
        "raw_channel_labels": list(reference_raw_layout),
        "analysis_eeg_labels": analysis_labels,
        "excluded_channels": excluded + status_labels,
    }


def preprocess_session5_training_subject(
    subject_id,
    session5_model_inputs,
    l_freq=0.1,
    h_freq=20.0,
    tmin=DEFAULT_EPOCH_TMIN,
    tmax=DEFAULT_EPOCH_TMAX,
    baseline_tmin=-0.2,
    baseline_tmax=0.0,
):
    """Preprocess one participant's validated Session 5 training runs.

    This step applies label-derived scalp-channel selection, zero-phase FIR
    filtering, Status-anchored epoching, and per-trial baseline correction. It
    deliberately does not perform lateralization, xDAWN, balancing, pruning,
    feature selection, or classifier fitting.
    """
    if not isinstance(session5_model_inputs, dict):
        raise TypeError("session5_model_inputs must be the manifest-result dictionary.")
    if "manifest" not in session5_model_inputs or "trials" not in session5_model_inputs:
        raise ValueError("session5_model_inputs must contain 'manifest' and 'trials'.")
    subject_id = str(subject_id).lower().strip()
    manifest = session5_model_inputs["manifest"].copy()
    trials = session5_model_inputs["trials"].copy()
    subject_manifest = manifest.loc[manifest["subject_id"] == subject_id].sort_values("run_id")
    subject_trials = trials.loc[trials["subject_id"] == subject_id].sort_values(
        ["run_id", "trial"]
    )
    if len(subject_manifest) != SESSION5_TRAINING_RUNS:
        raise ValueError(
            f"{subject_id}: expected {SESSION5_TRAINING_RUNS} validated Session 5 runs, "
            f"found {len(subject_manifest)}."
        )
    if len(subject_trials) != SESSION5_TRAINING_RUNS * TRAINING_TRIALS:
        raise ValueError(
            f"{subject_id}: expected {SESSION5_TRAINING_RUNS * TRAINING_TRIALS} "
            f"validated trials, found {len(subject_trials)}."
        )
    layout = validate_session5_analysis_channel_layout(subject_manifest)
    print(f"\nSESSION 5 PREPROCESSING: {subject_id}")
    print("Pipeline: scalp channels -> zero-phase 0.1-20 Hz FIR -> Status epochs -> baseline.")

    run_epochs = []
    run_summaries = []
    reference_time = None
    for run_row in subject_manifest.itertuples(index=False):
        run_id = int(run_row.run_id)
        run_trials = subject_trials.loc[subject_trials["run_id"] == run_id]
        result = load_filter_epoch_baseline_correct_training_run(
            gdf_path=Path(run_row.gdf_path),
            l_freq=l_freq,
            h_freq=h_freq,
            tmin=tmin,
            tmax=tmax,
            baseline_tmin=baseline_tmin,
            baseline_tmax=baseline_tmax,
        )
        if result["eeg_labels"] != layout["analysis_eeg_labels"]:
            raise ValueError(f"{subject_id} run {run_id}: analysis channel order changed after loading.")
        event_codes = result["stimulus_events"][:, 2].astype(int)
        expected_codes = run_trials["stimulus_trigger"].to_numpy(dtype=int)
        if not np.array_equal(event_codes, expected_codes):
            raise ValueError(
                f"{subject_id} run {run_id}: preprocessed Status-event labels do not "
                "align with the validated trial table."
            )
        epoch_data = result["baseline_corrected_epochs"]
        if epoch_data.shape[2] != len(run_trials):
            raise ValueError(
                f"{subject_id} run {run_id}: {epoch_data.shape[2]} epochs but "
                f"{len(run_trials)} validated trials."
            )
        time = np.asarray(result["time"], dtype=float)
        if reference_time is None:
            reference_time = time
        elif not np.array_equal(time, reference_time):
            raise ValueError(f"{subject_id} run {run_id}: epoch time axis differs from prior runs.")
        run_epochs.append(np.transpose(epoch_data, (1, 0, 2)))
        run_summaries.append({
            "subject_id": subject_id,
            "run_id": run_id,
            "n_epochs": int(epoch_data.shape[2]),
            "n_channels": int(epoch_data.shape[0]),
            "n_samples": int(epoch_data.shape[1]),
            "max_abs_residual_baseline_mean": float(
                result["max_abs_residual_baseline_mean"]
            ),
        })
        print(
            f"  Run {run_id}: {epoch_data.shape[2]} aligned epochs; "
            f"shape time x channels x trials = {run_epochs[-1].shape}."
        )

    epochs = np.concatenate(run_epochs, axis=2)
    if epochs.shape != (len(reference_time), len(layout["analysis_eeg_labels"]), len(subject_trials)):
        raise RuntimeError(
            f"{subject_id}: unexpected concatenated epoch shape {epochs.shape}."
        )
    if not np.isfinite(epochs).all():
        raise ValueError(f"{subject_id}: preprocessed epochs contain non-finite values.")
    run_summary = pd.DataFrame(run_summaries)
    print(
        f"Preprocessing passed: {epochs.shape[2]} trials, {epochs.shape[1]} channels, "
        f"{epochs.shape[0]} time samples; no learned transform has been fit."
    )
    return {
        "subject_id": subject_id,
        "group": get_subject_group(subject_id),
        "epochs_time_channels_trials": epochs,
        "time": reference_time,
        "eeg_labels": layout["analysis_eeg_labels"],
        "trial_table": subject_trials.reset_index(drop=True),
        "run_summary": run_summary,
        "preprocessing": {
            "l_freq": float(l_freq),
            "h_freq": float(h_freq),
            "tmin": float(tmin),
            "tmax": float(tmax),
            "baseline_tmin": float(baseline_tmin),
            "baseline_tmax": float(baseline_tmax),
            "filter_phase": "zero",
        },
    }


def preprocess_longitudinal_evaluation_subject(
    subject_id,
    longitudinal_inputs,
    l_freq=0.1,
    h_freq=20.0,
    tmin=DEFAULT_EPOCH_TMIN,
    tmax=DEFAULT_EPOCH_TMAX,
    baseline_tmin=-0.2,
    baseline_tmax=0.0,
):
    """Preprocess one subject's pre-training and decoding evaluation runs.

    The input is the validated manifest from
    :func:`build_longitudinal_evaluation_manifest`: Session 1 training first,
    followed by non-practice decoding runs in Sessions 1--5.  All runs receive
    the same post-hoc preprocessing as the Session 5 reference data (scalp EEG
    selection, zero-phase 0.1--20 Hz FIR, Status-anchored epochs, and -200--0
    ms per-trial baseline correction).  This function never fits xDAWN,
    normalization, r2, or a classifier.
    """
    if not isinstance(longitudinal_inputs, dict):
        raise TypeError("longitudinal_inputs must be the longitudinal manifest-result dictionary.")
    required = {"manifest", "trials"}
    missing = required - set(longitudinal_inputs)
    if missing:
        raise ValueError(f"longitudinal_inputs is missing required key(s): {sorted(missing)}.")
    subject_id = str(subject_id).lower().strip()
    manifest = longitudinal_inputs["manifest"].copy()
    trials = longitudinal_inputs["trials"].copy()
    subject_manifest = manifest.loc[manifest["subject_id"] == subject_id].copy()
    subject_trials = trials.loc[trials["subject_id"] == subject_id].copy()
    if subject_manifest.empty or subject_trials.empty:
        raise ValueError(f"{subject_id}: no validated longitudinal evaluation data were found.")
    task_rank = {"session1_training_pre": 0, "decoding": 1, "session5_training_post": 2}
    subject_manifest["_task_rank"] = subject_manifest["evaluation_task"].map(task_rank)
    if subject_manifest["_task_rank"].isna().any():
        raise ValueError(f"{subject_id}: manifest contains an unknown evaluation task.")
    subject_manifest = subject_manifest.sort_values(
        ["_task_rank", "session_id", "run_id"], kind="stable"
    ).drop(columns="_task_rank")
    layout = validate_session5_analysis_channel_layout(subject_manifest)
    expected_trials = len(subject_manifest) * TRAINING_TRIALS
    if len(subject_trials) != expected_trials:
        raise ValueError(
            f"{subject_id}: expected {expected_trials} trials from {len(subject_manifest)} runs, "
            f"found {len(subject_trials)}."
        )
    print(f"\nLONGITUDINAL EVALUATION PREPROCESSING: {subject_id}")
    task_session_pairs = set(zip(subject_manifest["evaluation_task"], subject_manifest["session_id"]))
    if task_session_pairs == {("decoding", 5)}:
        source_order = "Session-5 decoding source runs only"
    else:
        source_order = "Session 1 training (pre) -> Session 1--5 decoding -> Session 5 training (post, if present)"
    print(
        f"Order: {source_order}; "
        "scalp channels -> zero-phase 0.1-20 Hz FIR -> Status epochs -> baseline."
    )
    run_epochs, ordered_trial_tables, run_summaries = [], [], []
    reference_time = None
    for run_row in subject_manifest.itertuples(index=False):
        selector = (
            (subject_trials["evaluation_task"] == run_row.evaluation_task)
            & (subject_trials["session_id"] == run_row.session_id)
            & (subject_trials["run_id"] == run_row.run_id)
        )
        run_trials = subject_trials.loc[selector].sort_values("trial", kind="stable")
        if len(run_trials) != TRAINING_TRIALS:
            raise ValueError(
                f"{subject_id} {run_row.evaluation_task} Session {run_row.session_id} "
                f"run {run_row.run_id}: expected 60 trial-table rows, found {len(run_trials)}."
            )
        result = load_filter_epoch_baseline_correct_training_run(
            gdf_path=Path(run_row.gdf_path), l_freq=l_freq, h_freq=h_freq,
            tmin=tmin, tmax=tmax, baseline_tmin=baseline_tmin,
            baseline_tmax=baseline_tmax, validate_paired_files=False,
        )
        if result["eeg_labels"] != layout["analysis_eeg_labels"]:
            raise ValueError(
                f"{subject_id} {run_row.evaluation_task} Session {run_row.session_id} "
                f"run {run_row.run_id}: analysis channel order changed after loading."
            )
        event_codes = result["stimulus_events"][:, 2].astype(int)
        expected_codes = run_trials["stimulus_trigger"].to_numpy(dtype=int)
        if not np.array_equal(event_codes, expected_codes):
            raise ValueError(
                f"{subject_id} {run_row.evaluation_task} Session {run_row.session_id} "
                f"run {run_row.run_id}: preprocessed Status labels do not align with the trial table."
            )
        epoch_data = result["baseline_corrected_epochs"]
        if epoch_data.shape[2] != TRAINING_TRIALS:
            raise ValueError(
                f"{subject_id} {run_row.evaluation_task} Session {run_row.session_id} "
                f"run {run_row.run_id}: expected 60 epochs, found {epoch_data.shape[2]}."
            )
        time = np.asarray(result["time"], dtype=float)
        if reference_time is None:
            reference_time = time
        elif not np.array_equal(time, reference_time):
            raise ValueError(
                f"{subject_id}: epoch time axis differs in {run_row.evaluation_task} "
                f"Session {run_row.session_id} run {run_row.run_id}."
            )
        run_epochs.append(np.transpose(epoch_data, (1, 0, 2)))
        ordered_trial_tables.append(run_trials)
        task_event_codes, task_event_counts = np.unique(
            result["task_events"][:, 2].astype(int), return_counts=True
        )
        task_event_count_map = dict(zip(task_event_codes.tolist(), task_event_counts.tolist()))
        run_summaries.append({
            "subject_id": subject_id,
            "evaluation_task": run_row.evaluation_task,
            "session_id": int(run_row.session_id),
            "run_id": int(run_row.run_id),
            "n_epochs": int(epoch_data.shape[2]),
            "n_channels": int(epoch_data.shape[0]),
            "n_samples": int(epoch_data.shape[1]),
            "n_status_fixations": int(task_event_count_map.get(4, 0)),
            "n_status_stimuli": int(len(result["stimulus_events"])),
            "n_status_responses": int(task_event_count_map.get(64, 0)),
            "max_abs_residual_baseline_mean": float(result["max_abs_residual_baseline_mean"]),
        })
    epochs = np.concatenate(run_epochs, axis=2)
    ordered_trials = pd.concat(ordered_trial_tables, ignore_index=True)
    expected_shape = (len(reference_time), len(layout["analysis_eeg_labels"]), len(ordered_trials))
    if epochs.shape != expected_shape:
        raise RuntimeError(f"{subject_id}: expected concatenated shape {expected_shape}, got {epochs.shape}.")
    if not np.isfinite(epochs).all():
        raise ValueError(f"{subject_id}: preprocessed longitudinal epochs contain non-finite values.")
    run_summary = pd.DataFrame(run_summaries)
    print(
        f"Preprocessing passed: {len(run_summary)} runs, {epochs.shape[2]} trials, "
        f"{epochs.shape[1]} channels, {epochs.shape[0]} time samples; no learned transform fit."
    )
    return {
        "subject_id": subject_id,
        "group": get_subject_group(subject_id),
        "epochs_time_channels_trials": epochs,
        "time": reference_time,
        "eeg_labels": layout["analysis_eeg_labels"],
        "trial_table": ordered_trials,
        "run_summary": run_summary,
        "preprocessing": {
            "l_freq": float(l_freq), "h_freq": float(h_freq),
            "tmin": float(tmin), "tmax": float(tmax),
            "baseline_tmin": float(baseline_tmin),
            "baseline_tmax": float(baseline_tmax), "filter_phase": "zero",
        },
    }


def run_longitudinal_preprocessing_qc(longitudinal_inputs, subject_ids=None):
    """Run preprocessing QC sequentially for every requested participant.

    Large epoch arrays are deliberately discarded after each participant. This
    is a validation pass only; it produces a compact cohort QC table and does
    not save data, fit any learned transform, or compute feature r2 values.
    """
    if not isinstance(longitudinal_inputs, dict) or "manifest" not in longitudinal_inputs:
        raise TypeError("longitudinal_inputs must be a longitudinal manifest-result dictionary.")
    manifest = longitudinal_inputs["manifest"]
    available_subjects = sorted(manifest["subject_id"].unique().tolist())
    requested_subjects = (
        available_subjects if subject_ids is None
        else [str(subject_id).lower().strip() for subject_id in subject_ids]
    )
    unknown_subjects = sorted(set(requested_subjects) - set(available_subjects))
    if unknown_subjects:
        raise ValueError(
            "Requested subject(s) have no validated longitudinal inputs: "
            f"{unknown_subjects}."
        )
    print("LONGITUDINAL PREPROCESSING QC")
    print(
        f"Participants: {len(requested_subjects)}; runs represented in manifest: "
        f"{len(manifest.loc[manifest['subject_id'].isin(requested_subjects)])}."
    )
    subject_rows, run_qc_tables, failures = [], [], []
    for subject_number, subject_id in enumerate(requested_subjects, start=1):
        print(f"\n[{subject_number}/{len(requested_subjects)}] {subject_id}")
        try:
            subject_manifest = manifest.loc[manifest["subject_id"] == subject_id].copy()
            task_rank = {"session1_training_pre": 0, "decoding": 1}
            subject_manifest["_task_rank"] = subject_manifest["evaluation_task"].map(task_rank)
            subject_manifest = subject_manifest.sort_values(
                ["_task_rank", "session_id", "run_id"], kind="stable"
            ).drop(columns="_task_rank")
            layout = validate_session5_analysis_channel_layout(subject_manifest)
            subject_trials = longitudinal_inputs["trials"].loc[
                longitudinal_inputs["trials"]["subject_id"] == subject_id
            ]
            reference_time = None
            run_rows = []
            for run_row in subject_manifest.itertuples(index=False):
                run_trials = subject_trials.loc[
                    (subject_trials["evaluation_task"] == run_row.evaluation_task)
                    & (subject_trials["session_id"] == run_row.session_id)
                    & (subject_trials["run_id"] == run_row.run_id)
                ].sort_values("trial", kind="stable")
                if len(run_trials) != TRAINING_TRIALS:
                    raise ValueError(
                        f"{subject_id} {run_row.evaluation_task} Session {run_row.session_id} "
                        f"run {run_row.run_id}: expected 60 validated trials, found {len(run_trials)}."
                    )
                with redirect_stdout(io.StringIO()):
                    result = load_filter_epoch_baseline_correct_training_run(
                        gdf_path=Path(run_row.gdf_path), validate_paired_files=False
                    )
                if result["eeg_labels"] != layout["analysis_eeg_labels"]:
                    raise ValueError(f"{subject_id} {run_row.gdf_path}: analysis channel order changed.")
                if not np.array_equal(
                    result["stimulus_events"][:, 2].astype(int),
                    run_trials["stimulus_trigger"].to_numpy(dtype=int),
                ):
                    raise ValueError(f"{subject_id} {run_row.gdf_path}: Status/trial labels disagree after preprocessing.")
                epoch_data = result["baseline_corrected_epochs"]
                if epoch_data.shape != (61, 768, TRAINING_TRIALS):
                    raise ValueError(
                        f"{subject_id} {run_row.gdf_path}: unexpected epoch shape {epoch_data.shape}."
                    )
                time = np.asarray(result["time"], dtype=float)
                if reference_time is None:
                    reference_time = time
                elif not np.array_equal(time, reference_time):
                    raise ValueError(f"{subject_id} {run_row.gdf_path}: epoch time axis changed.")
                codes, counts = np.unique(result["task_events"][:, 2], return_counts=True)
                event_counts = dict(zip(codes.astype(int).tolist(), counts.astype(int).tolist()))
                run_rows.append({
                    "subject_id": subject_id, "evaluation_task": run_row.evaluation_task,
                    "session_id": int(run_row.session_id), "run_id": int(run_row.run_id),
                    "n_epochs": int(epoch_data.shape[2]), "n_channels": int(epoch_data.shape[0]),
                    "n_samples": int(epoch_data.shape[1]),
                    "n_status_fixations": int(event_counts.get(4, 0)),
                    "n_status_stimuli": int(len(result["stimulus_events"])),
                    "n_status_responses": int(event_counts.get(64, 0)),
                    "max_abs_residual_baseline_mean": float(result["max_abs_residual_baseline_mean"]),
                })
                del result, epoch_data
                gc.collect()
            run_qc = pd.DataFrame(run_rows)
            run_qc_tables.append(run_qc)
            subject_rows.append({
                "subject_id": subject_id,
                "group": get_subject_group(subject_id),
                "qc_status": "pass",
                "n_runs": int(len(run_qc)),
                "n_trials": int(run_qc["n_epochs"].sum()),
                "n_channels": int(run_qc["n_channels"].iloc[0]),
                "n_samples": int(run_qc["n_samples"].iloc[0]),
                "max_abs_residual_baseline_mean": float(
                    run_qc["max_abs_residual_baseline_mean"].max()
                ),
                "n_runs_with_non60_responses": int(
                    (run_qc["n_status_responses"] != TRAINING_TRIALS).sum()
                ),
            })
            print(
                f"  PASS: {len(run_qc)} runs, {run_qc['n_epochs'].sum()} trials; "
                f"max baseline residual={run_qc['max_abs_residual_baseline_mean'].max():.3e}; "
                f"non-60 response-event runs={(run_qc['n_status_responses'] != TRAINING_TRIALS).sum()}."
            )
            gc.collect()
        except Exception as exc:
            failures.append({
                "subject_id": subject_id,
                "issue": str(exc),
            })
            subject_rows.append({
                "subject_id": subject_id,
                "group": get_subject_group(subject_id),
                "qc_status": "failed",
                "n_runs": np.nan,
                "n_trials": np.nan,
                "n_channels": np.nan,
                "n_samples": np.nan,
                "max_abs_residual_baseline_mean": np.nan,
                "n_runs_with_non60_responses": np.nan,
            })
            print(f"WARNING: preprocessing QC failed for {subject_id}: {exc}")
            gc.collect()
    subject_qc = pd.DataFrame(subject_rows)
    run_qc = (
        pd.concat(run_qc_tables, ignore_index=True)
        if run_qc_tables else pd.DataFrame()
    )
    failures_table = pd.DataFrame(failures)
    n_passed = int((subject_qc["qc_status"] == "pass").sum())
    print(f"\nCohort QC complete: {n_passed}/{len(subject_qc)} participant(s) passed.")
    if not failures_table.empty:
        print("Participants requiring follow-up:")
        print(failures_table.to_string(index=False))
    else:
        print("All requested participants passed preprocessing QC.")
    return {
        "subject_qc": subject_qc,
        "run_qc": run_qc,
        "failures": failures_table,
    }


def run_checkpointed_longitudinal_preprocessing_qc(
    subject_ids=None,
    project_root=PROJECT_ROOT,
    output_dir=None,
    resume=True,
    retry_failures=False,
):
    """Run longitudinal QC in fresh per-participant processes with checkpoints.

    This is the cohort-scale entry point. Each child process runs one
    participant's run-level QC and exits, preventing MNE memory accumulation.
    Three CSV files are rewritten after every completed participant so an
    interrupted run can resume without repeating participants already marked
    ``pass`` (or previously recorded failures unless ``retry_failures=True``).
    """
    project_root = Path(project_root)
    subjects = list(EXPECTED_SUBJECTS if subject_ids is None else subject_ids)
    subjects = [str(subject_id).lower().strip() for subject_id in subjects]
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("subject_ids must be a non-empty list of unique subject IDs.")
    unknown = sorted(set(subjects) - set(EXPECTED_SUBJECTS))
    if unknown:
        raise ValueError(f"Unknown subject ID(s): {unknown}.")
    output_dir = Path(output_dir) if output_dir is not None else REPO_ROOT / "analyses" / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_path = output_dir / "longitudinal_preprocessing_qc_subjects.csv"
    run_path = output_dir / "longitudinal_preprocessing_qc_runs.csv"
    failure_path = output_dir / "longitudinal_preprocessing_qc_failures.csv"
    def _read_checkpoint(path):
        if not resume or not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    existing_subject = _read_checkpoint(subject_path)
    existing_run = _read_checkpoint(run_path)
    existing_failures = _read_checkpoint(failure_path)
    completed = set()
    if not existing_subject.empty and {"subject_id", "qc_status"}.issubset(existing_subject.columns):
        completed_statuses = ["pass"] if retry_failures else ["pass", "failed"]
        completed = set(existing_subject.loc[
            existing_subject["qc_status"].isin(completed_statuses), "subject_id"
        ])
    source_dir = Path(__file__).resolve().parents[1]
    child_environment = os.environ.copy()
    child_environment["PYTHONPATH"] = (
        f"{source_dir}{os.pathsep}{child_environment.get('PYTHONPATH', '')}"
    )
    print("CHECKPOINTED LONGITUDINAL PREPROCESSING QC")
    print(f"Outputs: {output_dir}")
    print(f"Requested: {len(subjects)} participant(s); already passed: {len(completed & set(subjects))}.")
    for index, subject_id in enumerate(subjects, start=1):
        if subject_id in completed:
            print(f"[{index}/{len(subjects)}] {subject_id}: already passed; skipped.")
            continue
        command = [
            sys.executable, "-m", "posthoc_analysis.longitudinal_qc_worker",
            "--subject-id", subject_id,
            "--project-root", str(project_root),
        ]
        print(f"[{index}/{len(subjects)}] {subject_id}: running isolated QC worker...")
        result = subprocess.run(
            command, text=True, capture_output=True, env=child_environment, check=False
        )
        try:
            # Package configuration diagnostics can precede the worker's
            # captured output. Its final stdout line is the JSON payload.
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"worker exit code {result.returncode}")
            subject_result = pd.DataFrame(payload["subject_qc"])
            run_result = pd.DataFrame(payload["run_qc"])
            failure_result = pd.DataFrame(payload["failures"])
        except Exception as exc:
            subject_result = pd.DataFrame([{
                "subject_id": subject_id, "group": get_subject_group(subject_id),
                "qc_status": "failed", "worker_error": str(exc),
            }])
            run_result = pd.DataFrame()
            failure_result = pd.DataFrame([{
                "subject_id": subject_id,
                "issue": result.stderr.strip() or str(exc),
            }])
        if "subject_id" in existing_subject:
            existing_subject = existing_subject.loc[
                existing_subject["subject_id"] != subject_id
            ]
        existing_subject = pd.concat([existing_subject, subject_result], ignore_index=True)
        if not existing_run.empty and "subject_id" in existing_run:
            existing_run = existing_run.loc[existing_run["subject_id"] != subject_id]
        existing_run = pd.concat([existing_run, run_result], ignore_index=True)
        if not existing_failures.empty and "subject_id" in existing_failures:
            existing_failures = existing_failures.loc[existing_failures["subject_id"] != subject_id]
        existing_failures = pd.concat([existing_failures, failure_result], ignore_index=True)
        existing_subject.to_csv(subject_path, index=False)
        existing_run.to_csv(run_path, index=False)
        existing_failures.to_csv(failure_path, index=False)
        status = subject_result.iloc[0]["qc_status"]
        print(f"  {subject_id}: {status}; checkpoint saved.")
    return {
        "subject_qc": existing_subject.sort_values("subject_id", kind="stable").reset_index(drop=True),
        "run_qc": existing_run.sort_values(
            ["subject_id", "evaluation_task", "session_id", "run_id"], kind="stable"
        ).reset_index(drop=True) if not existing_run.empty else existing_run,
        "failures": existing_failures.reset_index(drop=True),
        "paths": {"subject_qc": subject_path, "run_qc": run_path, "failures": failure_path},
    }


def construct_session5_conventional_difference_inputs(preprocessed_subject):
    """Create conventional right-minus-left P/PO inputs for both binary models.

    This intentionally corrects a historical channel-ordering issue: the
    original code independently selected left/right label sets with
    ``find(ismember(...))``, which did not preserve the electrode-list order.
    New post-hoc models use explicit conventional pairs. No spatial filter is
    fit in this deterministic preprocessing step.
    """
    required = {"epochs_time_channels_trials", "eeg_labels", "trial_table", "time"}
    missing = required - set(preprocessed_subject)
    if missing:
        raise ValueError(
            f"Preprocessed subject input is missing required key(s): {sorted(missing)}."
        )
    epochs = np.asarray(preprocessed_subject["epochs_time_channels_trials"], dtype=float)
    labels = list(preprocessed_subject["eeg_labels"])
    trials = preprocessed_subject["trial_table"].reset_index(drop=True).copy()
    if epochs.ndim != 3:
        raise ValueError(
            "epochs_time_channels_trials must be time x channels x trials, "
            f"got {epochs.shape}."
        )
    if epochs.shape[1] != len(labels) or epochs.shape[2] != len(trials):
        raise ValueError(
            "Epoch dimensions do not align with labels/trials: "
            f"epochs={epochs.shape}, labels={len(labels)}, trials={len(trials)}."
        )
    label_to_index = {label: index for index, label in enumerate(labels)}
    requested_labels = [label for pair in CONVENTIONAL_PPO_PAIRS for label in pair]
    missing_labels = sorted(set(requested_labels) - set(label_to_index))
    if missing_labels:
        raise ValueError(
            "Missing conventional P/PO electrode label(s): "
            f"{missing_labels}. Available labels: {labels}."
        )
    right_labels = [right for right, _ in CONVENTIONAL_PPO_PAIRS]
    left_labels = [left for _, left in CONVENTIONAL_PPO_PAIRS]
    right_indices = [label_to_index[label] for label in right_labels]
    left_indices = [label_to_index[label] for label in left_labels]
    source_pairs = list(CONVENTIONAL_PPO_PAIRS)
    differences = epochs[:, right_indices, :] - epochs[:, left_indices, :]
    if not np.isfinite(differences).all():
        raise ValueError("Source-decoder P/PO difference epochs contain non-finite values.")

    binary_inputs = {}
    for side, include_column, label_column in (
        ("right", "right_model_include", "right_model_label"),
        ("left", "left_model_include", "left_model_label"),
    ):
        if include_column not in trials or label_column not in trials:
            raise ValueError(f"Trial table is missing {include_column!r} or {label_column!r}.")
        include = trials[include_column].to_numpy(dtype=bool)
        model_labels = trials.loc[include, label_column].to_numpy(dtype=float)
        if np.isnan(model_labels).any() or not set(np.unique(model_labels)).issubset({0.0, 1.0}):
            raise ValueError(f"{side} model labels must be finite binary values.")
        model_labels = model_labels.astype(int)
        expected_positive = int((trials["condition"] == f"distractor_{side}").sum())
        expected_negative = int((trials["condition"] == "no_distractor").sum())
        observed_positive = int((model_labels == 1).sum())
        observed_negative = int((model_labels == 0).sum())
        if (observed_positive, observed_negative) != (expected_positive, expected_negative):
            raise ValueError(
                f"{side} binary trial counts do not match condition labels: "
                f"observed (+/-)=({observed_positive}/{observed_negative}), "
                f"expected=({expected_positive}/{expected_negative})."
            )
        binary_inputs[side] = {
            "epochs_time_channels_trials": differences[:, :, include],
            "labels": model_labels,
            "trial_table": trials.loc[include].reset_index(drop=True),
        }

    print("SESSION 5 CONVENTIONAL P/PO DIFFERENCE INPUTS")
    print("Fixed orientation: conventional right electrode minus left homologue.")
    print("Ordered channel pairs used:")
    for right_label, left_label in source_pairs:
        print(f"  {right_label} - {left_label}")
    print(
        "NOTE: These explicit conventional pairs intentionally correct the "
        "historical independent-ordering behavior of find(ismember(...))."
    )
    for side, model_input in binary_inputs.items():
        counts = np.bincount(model_input["labels"], minlength=2)
        print(
            f"  {side.title()} model: {model_input['epochs_time_channels_trials'].shape}; "
            f"labels no={counts[0]}, distractor={counts[1]}."
        )
    return {
        "subject_id": preprocessed_subject.get("subject_id"),
        "group": preprocessed_subject.get("group"),
        "time": np.asarray(preprocessed_subject["time"], dtype=float),
        "difference_channel_labels": [f"{right}-{left}" for right, left in source_pairs],
        "right_source_labels": right_labels,
        "left_source_labels": left_labels,
        "conventional_pairs_right_minus_left": source_pairs,
        "all_difference_epochs_time_channels_trials": differences,
        "right": binary_inputs["right"],
        "left": binary_inputs["left"],
    }


def build_session5_top30_feature_references_for_subject(
    subject_id,
    session5_model_inputs=None,
    n_pruning_iterations=20,
    random_seed=20260812,
    preprocessed_subject=None,
):
    """Build frozen right/no and left/no Session-5 top-30 references.

    This reproducible orchestration function performs conventional-pair
    construction, iterative pruning with leave-one-run-out CV, then a final
    clean-trial xDAWN/z-score/r2 refit. By default it preprocesses the
    validated Session-5 training input. Advanced callers may instead provide
    an already validated/preprocessed Session-5 source dataset, which is used
    unchanged and must be documented by the caller.
    """
    subject_id = str(subject_id).lower().strip()
    if preprocessed_subject is None:
        if session5_model_inputs is None:
            raise ValueError("session5_model_inputs is required when preprocessed_subject is not supplied.")
        preprocessed = preprocess_session5_training_subject(subject_id, session5_model_inputs)
    else:
        preprocessed = preprocessed_subject
        if not isinstance(preprocessed, dict) or not {"subject_id", "trial_table"}.issubset(preprocessed):
            raise ValueError("preprocessed_subject must be a validated preprocessing-result dictionary.")
        if str(preprocessed["subject_id"]).lower().strip() != subject_id:
            raise ValueError("preprocessed_subject belongs to a different participant.")
    difference_inputs = construct_session5_conventional_difference_inputs(preprocessed)
    time = difference_inputs["time"]
    side_results = {}
    summary_rows, clean_trial_tables, feature_tables = [], [], []
    print(f"\nSESSION 5 TOP-30 FEATURE REFERENCES: {subject_id}")
    for side_index, side in enumerate(("right", "left")):
        model_input = difference_inputs[side]
        print(f"\nBuilding {side}/no reference.")
        pruning = run_iterative_pruning_feature_cv(
            model_input,
            time,
            n_iterations=n_pruning_iterations,
            random_seed=int(random_seed + side_index * 10_000),
            feature_start_s=0.2,
            feature_stop_s=None,
            resample_ratio=8,
            n_xdawn_components=2,
            n_selected_features=30,
        )
        clean_mask = np.asarray(pruning["best_mask"], dtype=bool)
        clean_epochs = model_input["epochs_time_channels_trials"][:, :, clean_mask]
        clean_labels = np.asarray(model_input["labels"])[clean_mask]
        reference = fit_final_clean_feature_reference(
            clean_epochs,
            clean_labels,
            time,
            feature_start_s=0.2,
            feature_stop_s=None,
            resample_ratio=8,
            n_xdawn_components=2,
            n_selected_features=30,
        )
        clean_trials = model_input["trial_table"].loc[clean_mask].copy().reset_index(drop=True)
        clean_trials.insert(0, "decoder_side", side)
        if "subject_id" not in clean_trials.columns:
            clean_trials.insert(0, "subject_id", subject_id)
        elif not clean_trials["subject_id"].astype(str).str.lower().eq(subject_id).all():
            raise ValueError(
                f"{subject_id} {side}: clean-trial subject IDs do not match the requested subject."
            )
        clean_trials["best_pruning_iteration"] = pruning["best_iteration"]
        features = reference["selected_coordinates"].copy()
        features.insert(0, "decoder_side", side)
        features.insert(0, "subject_id", subject_id)
        features["best_pruning_iteration"] = pruning["best_iteration"]
        history = pruning["history"]
        best_row = history.loc[history["iteration"] == pruning["best_iteration"]].iloc[0]
        summary_rows.append({
            "subject_id": subject_id,
            "group": get_subject_group(subject_id),
            "decoder_side": side,
            "best_pruning_iteration": int(pruning["best_iteration"]),
            "best_cv_auprc": float(best_row["auprc"]),
            "clean_trial_count": int(clean_mask.sum()),
            "clean_no_distractor_count": int((clean_labels == 0).sum()),
            "clean_distractor_count": int((clean_labels == 1).sum()),
            "completed_pruning_iterations": int(pruning["completed_iterations"]),
            "pruning_stop_reason": pruning["stop_reason"],
            "random_seed": int(random_seed + side_index * 10_000),
            "feature_start_s": 0.2,
            "feature_stop_s": float(reference["settings"]["feature_stop_s"]),
            "resample_ratio": 8,
            "n_xdawn_components": 2,
            "n_selected_features": 30,
        })
        clean_trial_tables.append(clean_trials)
        feature_tables.append(features)
        side_results[side] = {"pruning": pruning, "reference": reference}
    summary = pd.DataFrame(summary_rows)
    clean_trial_table = pd.concat(clean_trial_tables, ignore_index=True)
    top30_table = pd.concat(feature_tables, ignore_index=True)
    if len(top30_table) != 60 or top30_table.groupby("decoder_side").size().to_dict() != {"left": 30, "right": 30}:
        raise RuntimeError(f"{subject_id}: expected 30 frozen features per side.")
    print("Session-5 top-30 reference build passed.")
    print(summary[["decoder_side", "best_pruning_iteration", "best_cv_auprc", "clean_trial_count"]].to_string(index=False))
    return {
        "subject_id": subject_id,
        "preprocessed": preprocessed,
        "difference_inputs": difference_inputs,
        "side_results": side_results,
        "summary": summary,
        "clean_trials": clean_trial_table,
        "top30_features": top30_table,
    }


def save_session5_top30_feature_references(reference_result, output_dir=None):
    """Upsert a subject's lean, frozen Session-5 reference artifacts.

    The Parquet tables retain the cohort-searchable selection summary, clean
    trial identities, and final clean-training r2 values.  Per-subject/side
    ``.npz`` files retain the frozen final xDAWN filters, resampling indices,
    z-score parameters, and feature indices.  Evaluation datasets must apply
    those saved transforms unchanged; they must not refit them.
    """
    required = {"subject_id", "summary", "clean_trials", "top30_features", "side_results", "difference_inputs"}
    missing = required.difference(reference_result)
    if missing:
        raise ValueError(f"Reference result is missing required entries: {sorted(missing)}.")
    subject_id = str(reference_result["subject_id"]).lower().strip()
    summary = reference_result["summary"].copy()
    clean_trials = reference_result["clean_trials"].copy()
    top30 = reference_result["top30_features"].copy()
    expected_summary = {"subject_id", "decoder_side", "best_cv_auprc"}
    expected_clean_trials = {"subject_id", "decoder_side", "session_id", "run_id", "trial"}
    expected_top30 = {"subject_id", "decoder_side", "rank", "r2_clean_training"}
    for name, table, required_columns in (
        ("summary", summary, expected_summary),
        ("clean_trials", clean_trials, expected_clean_trials),
        ("top30_features", top30, expected_top30),
    ):
        missing_columns = required_columns.difference(table.columns)
        if missing_columns:
            raise ValueError(f"{subject_id}: {name} is missing columns {sorted(missing_columns)}.")
        if not table["subject_id"].astype(str).str.lower().eq(subject_id).all():
            raise ValueError(f"{subject_id}: {name} contains another subject's rows.")
    if summary["decoder_side"].value_counts().to_dict() != {"right": 1, "left": 1}:
        raise ValueError(f"{subject_id}: summary must contain exactly one row for each decoder side.")
    if top30.groupby("decoder_side").size().to_dict() != {"right": 30, "left": 30}:
        raise ValueError(f"{subject_id}: top30 table must contain 30 features for each side.")
    if clean_trials.duplicated(["subject_id", "decoder_side", "session_id", "run_id", "trial"]).any():
        raise ValueError(f"{subject_id}: clean-trial identities are not unique within decoder side.")

    output_dir = Path(output_dir) if output_dir is not None else REPO_ROOT / "analyses" / "session5_feature_references"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "session5_reference_summary.parquet",
        "clean_trials": output_dir / "session5_reference_clean_trials.parquet",
        "top30_features": output_dir / "session5_reference_top30.parquet",
        "settings": output_dir / "session5_reference_settings.json",
        "transforms_dir": output_dir / "transforms",
    }
    paths["transforms_dir"].mkdir(exist_ok=True)

    def _upsert(path, new_rows, identity_columns):
        if path.exists():
            existing = pd.read_parquet(path)
            missing_existing = set(new_rows.columns).difference(existing.columns)
            if missing_existing:
                raise ValueError(
                    f"Existing {path.name} lacks current schema columns {sorted(missing_existing)}; "
                    "migrate it explicitly rather than silently dropping fields."
                )
            existing = existing.loc[
                ~existing["subject_id"].astype(str).str.lower().eq(subject_id)
            ].copy()
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows.copy()
        if combined.duplicated(identity_columns).any():
            raise RuntimeError(f"{path.name}: upsert produced duplicate identity rows.")
        temporary_path = path.with_name(f".{path.stem}.tmp.parquet")
        combined.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
        return combined

    saved_summary = _upsert(paths["summary"], summary, ["subject_id", "decoder_side"])
    saved_clean = _upsert(
        paths["clean_trials"], clean_trials,
        ["subject_id", "decoder_side", "session_id", "run_id", "trial"],
    )
    saved_top30 = _upsert(paths["top30_features"], top30, ["subject_id", "decoder_side", "rank"])
    transform_paths = {}
    for decoder_side in ("right", "left"):
        reference = reference_result["side_results"].get(decoder_side, {}).get("reference")
        if reference is None:
            raise ValueError(f"{subject_id}: no final frozen {decoder_side} reference is available to save.")
        reference_required = {
            "xdawn_filters_components_by_channels", "feature_window_indices", "resampled_indices",
            "normalization_means", "normalization_stds", "selected_indices_zero_based", "settings",
        }
        reference_missing = reference_required.difference(reference)
        if reference_missing:
            raise ValueError(
                f"{subject_id} {decoder_side}: final reference lacks {sorted(reference_missing)}."
            )
        selected_rows = saved_top30.loc[
            (saved_top30["subject_id"].astype(str).str.lower() == subject_id)
            & (saved_top30["decoder_side"] == decoder_side)
        ].sort_values("rank")
        selected_indices = np.asarray(reference["selected_indices_zero_based"], dtype=int)
        if not np.array_equal(selected_rows["feature_index_zero_based"].to_numpy(dtype=int), selected_indices):
            raise RuntimeError(
                f"{subject_id} {decoder_side}: saved top-30 table does not match the frozen reference indices."
            )
        transform_path = paths["transforms_dir"] / f"{subject_id}_{decoder_side}_frozen_reference.npz"
        temporary_path = transform_path.with_name(f".{transform_path.stem}.tmp.npz")
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                artifact_version=np.asarray([1], dtype=int),
                subject_id=np.asarray([subject_id]),
                decoder_side=np.asarray([decoder_side]),
                epoch_time_s=np.asarray(reference_result["difference_inputs"]["time"], dtype=float),
                difference_channel_labels=np.asarray(reference_result["difference_inputs"]["difference_channel_labels"]),
                xdawn_filters_components_by_channels=np.asarray(
                    reference["xdawn_filters_components_by_channels"], dtype=float
                ),
                feature_window_indices=np.asarray(reference["feature_window_indices"], dtype=int),
                resampled_indices=np.asarray(reference["resampled_indices"], dtype=int),
                normalization_means=np.asarray(reference["normalization_means"], dtype=float),
                normalization_stds=np.asarray(reference["normalization_stds"], dtype=float),
                selected_indices_zero_based=selected_indices,
            )
        os.replace(temporary_path, transform_path)
        transform_paths[decoder_side] = transform_path
    settings = {
        "analysis": "session5_frozen_top30_feature_references",
        "source": "Session 5 training task, four non-practice runs",
        "decoder_models": {
            "right": "right distractor versus no distractor",
            "left": "left distractor versus no distractor",
        },
        "channel_construction": "seven conventional P/PO right-minus-left pairs",
        "feature_window_s": [0.2, "last_available_sample"],
        "resample_ratio": 8,
        "xdawn_components": 2,
        "feature_ranking": "binary r2 after final clean-trial xDAWN and z-scoring",
        "selection": "run-wise balancing, leave-one-run-out iterative pruning, best CV AUPRC iteration",
        "n_selected_features_per_decoder": 30,
        "frozen_transform_artifacts": "transforms/{subject_id}_{decoder_side}_frozen_reference.npz",
        "evaluation_rule": "apply the saved Session-5 xDAWN filters, resampling, z-score parameters, and selected indices unchanged; do not refit them on an evaluation dataset",
    }
    if paths["settings"].exists():
        with paths["settings"].open() as handle:
            existing_settings = json.load(handle)
        if existing_settings != settings:
            legacy_keys = {"frozen_transform_artifacts", "evaluation_rule"}
            can_upgrade_legacy_settings = (
                legacy_keys.isdisjoint(existing_settings)
                and all(existing_settings.get(key) == value for key, value in existing_settings.items())
            )
            if not can_upgrade_legacy_settings:
                raise ValueError("Existing session5 reference settings differ from the current workflow.")
            with paths["settings"].open("w") as handle:
                json.dump(settings, handle, indent=2)
                handle.write("\n")
            print("Upgraded legacy Session-5 reference settings to record frozen transforms.")
    else:
        with paths["settings"].open("w") as handle:
            json.dump(settings, handle, indent=2)
            handle.write("\n")
    print(
        f"Saved {subject_id} Session-5 frozen references: "
        f"{len(saved_summary)} summary rows, {len(saved_clean)} clean-trial rows, "
        f"{len(saved_top30)} top-30 feature rows, and two frozen transform files."
    )
    return {
        "paths": {**paths, "transforms": transform_paths},
        "summary": saved_summary,
        "clean_trials": saved_clean,
        "top30_features": saved_top30,
    }


def load_session5_frozen_feature_reference(subject_id, decoder_side, output_dir=None):
    """Load one saved Session-5 transform for fixed-feature evaluation."""
    subject_id = str(subject_id).lower().strip()
    decoder_side = str(decoder_side).lower().strip()
    if decoder_side not in {"right", "left"}:
        raise ValueError("decoder_side must be 'right' or 'left'.")
    output_dir = Path(output_dir) if output_dir is not None else REPO_ROOT / "analyses" / "session5_feature_references"
    transform_path = output_dir / "transforms" / f"{subject_id}_{decoder_side}_frozen_reference.npz"
    top30_path = output_dir / "session5_reference_top30.parquet"
    if not transform_path.exists() or not top30_path.exists():
        raise FileNotFoundError(
            f"Missing frozen reference for {subject_id} {decoder_side}: expected "
            f"{transform_path.name} and {top30_path.name}."
        )
    with np.load(transform_path, allow_pickle=False) as archive:
        required = {
            "artifact_version", "subject_id", "decoder_side", "epoch_time_s", "difference_channel_labels",
            "xdawn_filters_components_by_channels", "feature_window_indices", "resampled_indices",
            "normalization_means", "normalization_stds", "selected_indices_zero_based",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{transform_path.name} is missing fields {sorted(missing)}.")
        saved_subject = str(archive["subject_id"][0]).lower()
        saved_side = str(archive["decoder_side"][0]).lower()
        if saved_subject != subject_id or saved_side != decoder_side:
            raise ValueError(f"{transform_path.name} identity does not match the requested reference.")
        reference = {
            "xdawn_filters_components_by_channels": np.asarray(
                archive["xdawn_filters_components_by_channels"], dtype=float
            ),
            "feature_window_indices": np.asarray(archive["feature_window_indices"], dtype=int),
            "resampled_indices": np.asarray(archive["resampled_indices"], dtype=int),
            "normalization_means": np.asarray(archive["normalization_means"], dtype=float),
            "normalization_stds": np.asarray(archive["normalization_stds"], dtype=float),
            "selected_indices_zero_based": np.asarray(archive["selected_indices_zero_based"], dtype=int),
            "epoch_time_s": np.asarray(archive["epoch_time_s"], dtype=float),
            "difference_channel_labels": archive["difference_channel_labels"].astype(str).tolist(),
        }
    if reference["xdawn_filters_components_by_channels"].shape != (2, 7):
        raise ValueError(f"{transform_path.name} has unexpected xDAWN shape.")
    n_candidates = 2 * len(reference["resampled_indices"])
    if (
        reference["normalization_means"].shape != (n_candidates, 1)
        or reference["normalization_stds"].shape != (n_candidates, 1)
    ):
        raise ValueError(f"{transform_path.name} normalization dimensions do not match its resampling indices.")
    if np.any(reference["normalization_stds"] <= 0):
        raise ValueError(f"{transform_path.name} contains non-positive normalization standard deviations.")
    top30 = pd.read_parquet(top30_path)
    selected_coordinates = top30.loc[
        (top30["subject_id"].astype(str).str.lower() == subject_id)
        & (top30["decoder_side"] == decoder_side)
    ].sort_values("rank").reset_index(drop=True)
    if len(selected_coordinates) != 30 or not np.array_equal(
        selected_coordinates["feature_index_zero_based"].to_numpy(dtype=int),
        reference["selected_indices_zero_based"],
    ):
        raise ValueError(f"{subject_id} {decoder_side}: top-30 table and transform artifact disagree.")
    reference["selected_coordinates"] = selected_coordinates
    print(
        f"Loaded frozen Session-5 reference for {subject_id} {decoder_side}: "
        f"{len(selected_coordinates)} fixed features, xDAWN={reference['xdawn_filters_components_by_channels'].shape}."
    )
    return reference


def evaluate_frozen_session5_features_for_subject(
    subject_id,
    longitudinal_inputs,
    reference_dir=None,
):
    """Evaluate frozen Session-5 top-30 features for one participant.

    Produces one long table with two explicitly distinct estimates: ``run``
    rows use trials from one eligible run; ``session`` rows concatenate all
    eligible runs within an evaluation task and session *before* calculating
    r2.  It never refits xDAWN, normalization, feature selection, balancing,
    pruning, or a classifier.
    """
    subject_id = str(subject_id).lower().strip()
    if not isinstance(longitudinal_inputs, dict) or not {"manifest", "trials"}.issubset(longitudinal_inputs):
        raise ValueError("longitudinal_inputs must contain validated 'manifest' and 'trials' tables.")
    reference_dir = Path(reference_dir) if reference_dir is not None else REPO_ROOT / "analyses" / "session5_feature_references"
    preprocessed = preprocess_longitudinal_evaluation_subject(subject_id, longitudinal_inputs)
    difference_inputs = construct_session5_conventional_difference_inputs(preprocessed)
    expected_time = np.asarray(difference_inputs["time"], dtype=float)
    expected_labels = list(difference_inputs["difference_channel_labels"])
    result_tables = []
    frozen_references = {}

    def _evaluate_subset(side, epochs, labels, trial_rows, aggregation_level, run_id, n_contributing_runs):
        labels = np.asarray(labels, dtype=int)
        if not np.array_equal(np.unique(labels), np.array([0, 1])):
            raise ValueError(
                f"{subject_id} {side} {aggregation_level}: expected both binary classes, "
                f"found {np.unique(labels).tolist()}."
            )
        reference = frozen_references[side]
        frozen_result = apply_frozen_feature_reference_and_compute_r2(epochs, labels, reference)
        feature_rows = frozen_result["feature_r2_table"].copy()
        task_values = trial_rows["evaluation_task"].unique()
        session_values = trial_rows["session_id"].unique()
        if len(task_values) != 1 or len(session_values) != 1:
            raise ValueError(f"{subject_id} {side}: an r2 subset must belong to one task and session.")
        feature_rows = feature_rows.drop(
            columns=[column for column in ("subject_id", "decoder_side") if column in feature_rows],
        )
        feature_rows.insert(0, "subject_id", subject_id)
        feature_rows.insert(1, "group", get_subject_group(subject_id))
        feature_rows.insert(2, "decoder_side", side)
        feature_rows.insert(3, "aggregation_level", aggregation_level)
        feature_rows.insert(4, "evaluation_task", str(task_values[0]))
        feature_rows.insert(5, "session_id", int(session_values[0]))
        feature_rows.insert(6, "run_id", run_id)
        feature_rows.insert(7, "n_contributing_runs", int(n_contributing_runs))
        feature_rows.insert(8, "n_binary_trials", int(len(labels)))
        feature_rows.insert(9, "n_no_distractor_trials", int((labels == 0).sum()))
        feature_rows.insert(10, "n_relevant_distractor_trials", int((labels == 1).sum()))
        return feature_rows

    print(f"\nFROZEN SESSION-5 TOP-30 LONGITUDINAL EVALUATION: {subject_id}")
    print("No xDAWN, normalization, or feature-selection quantity is refit on evaluation data.")
    for side in ("right", "left"):
        frozen_reference = load_session5_frozen_feature_reference(
            subject_id, side, output_dir=reference_dir
        )
        if not np.array_equal(frozen_reference["epoch_time_s"], expected_time):
            raise ValueError(f"{subject_id} {side}: evaluation epoch time axis differs from frozen Session-5 reference.")
        if frozen_reference["difference_channel_labels"] != expected_labels:
            raise ValueError(f"{subject_id} {side}: conventional difference-channel order differs from frozen reference.")
        frozen_references[side] = frozen_reference

    for side in ("right", "left"):
        model_input = difference_inputs[side]
        epochs = np.asarray(model_input["epochs_time_channels_trials"], dtype=float)
        labels = np.asarray(model_input["labels"], dtype=int)
        trials = model_input["trial_table"].reset_index(drop=True).copy()
        if epochs.shape[2] != len(labels) or len(trials) != len(labels):
            raise RuntimeError(f"{subject_id} {side}: binary epochs, labels, and trials are misaligned.")
        task_rank = {"session1_training_pre": 0, "decoding": 1, "session5_training_post": 2}
        run_groups = trials.assign(_task_rank=trials["evaluation_task"].map(task_rank)).groupby(
            ["_task_rank", "evaluation_task", "session_id", "run_id"], sort=True, dropna=False
        )
        for (_, evaluation_task, session_id, run_id), run_trials in run_groups:
            indices = run_trials.index.to_numpy(dtype=int)
            if len(indices) == 0:
                continue
            result_tables.append(_evaluate_subset(
                side, epochs[:, :, indices], labels[indices], run_trials,
                aggregation_level="run", run_id=int(run_id), n_contributing_runs=1,
            ))
        session_groups = trials.assign(_task_rank=trials["evaluation_task"].map(task_rank)).groupby(
            ["_task_rank", "evaluation_task", "session_id"], sort=True, dropna=False
        )
        for (_, evaluation_task, session_id), session_trials in session_groups:
            indices = session_trials.index.to_numpy(dtype=int)
            n_runs = int(session_trials["run_id"].nunique())
            result_tables.append(_evaluate_subset(
                side, epochs[:, :, indices], labels[indices], session_trials,
                aggregation_level="session", run_id=pd.NA, n_contributing_runs=n_runs,
            ))
    r2_table = pd.concat(result_tables, ignore_index=True)
    r2_table["run_id"] = r2_table["run_id"].astype("Int64")
    identity = [
        "subject_id", "decoder_side", "aggregation_level", "evaluation_task",
        "session_id", "run_id", "rank",
    ]
    if r2_table.duplicated(identity).any():
        raise RuntimeError(f"{subject_id}: longitudinal r2 output has duplicate feature identities.")
    if not r2_table.groupby(["decoder_side", "aggregation_level", "evaluation_task", "session_id", "run_id"], dropna=False).size().eq(30).all():
        raise RuntimeError(f"{subject_id}: every run/session r2 subset must contain exactly 30 features.")
    run_rows = int((r2_table["aggregation_level"] == "run").sum() / 30)
    session_rows = int((r2_table["aggregation_level"] == "session").sum() / 30)
    print(
        f"{subject_id}: frozen-feature r2 passed for {run_rows} runs and {session_rows} pooled sessions "
        f"per decoder representation; output rows={len(r2_table)}."
    )
    return {
        "subject_id": subject_id,
        "r2_table": r2_table.sort_values(identity, kind="stable").reset_index(drop=True),
        "run_qc": preprocessed["run_summary"].copy(),
    }


def save_longitudinal_frozen_r2_for_subject(evaluation_result, output_dir=None):
    """Upsert one participant's run-wise and pooled-session frozen r2 rows."""
    if not isinstance(evaluation_result, dict) or not {"subject_id", "r2_table"}.issubset(evaluation_result):
        raise ValueError("evaluation_result must contain 'subject_id' and 'r2_table'.")
    subject_id = str(evaluation_result["subject_id"]).lower().strip()
    table = evaluation_result["r2_table"].copy()
    required = {
        "subject_id", "decoder_side", "aggregation_level", "evaluation_task", "session_id",
        "run_id", "n_contributing_runs", "n_binary_trials", "rank", "r2_evaluation",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{subject_id}: r2 table is missing required columns {sorted(missing)}.")
    if table.empty or not table["subject_id"].astype(str).str.lower().eq(subject_id).all():
        raise ValueError(f"{subject_id}: r2 table is empty or contains another participant's rows.")
    if set(table["aggregation_level"]) != {"run", "session"}:
        raise ValueError(f"{subject_id}: r2 table must contain both run and session aggregation levels.")
    output_dir = Path(output_dir) if output_dir is not None else REPO_ROOT / "analyses" / "session5_feature_references"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "longitudinal_top30_r2.parquet"
    identity = [
        "subject_id", "decoder_side", "aggregation_level", "evaluation_task",
        "session_id", "run_id", "rank",
    ]
    if table.duplicated(identity).any():
        raise ValueError(f"{subject_id}: r2 table has duplicate feature identities.")
    if path.exists():
        existing = pd.read_parquet(path)
        missing_existing = set(table.columns).difference(existing.columns)
        if missing_existing:
            raise ValueError(
                f"Existing {path.name} lacks current schema columns {sorted(missing_existing)}; migrate explicitly."
            )
        existing = existing.loc[~existing["subject_id"].astype(str).str.lower().eq(subject_id)]
        combined = pd.concat([existing, table], ignore_index=True)
    else:
        combined = table
    if combined.duplicated(identity).any():
        raise RuntimeError(f"{path.name}: upsert produced duplicate feature identities.")
    temporary_path = path.with_name(f".{path.stem}.tmp.parquet")
    combined.to_parquet(temporary_path, index=False)
    os.replace(temporary_path, path)
    print(
        f"Saved {subject_id} frozen-feature r2: {len(table)} rows for this participant; "
        f"cohort table rows={len(combined)}."
    )
    return {"path": path, "r2_table": combined}


def append_session5_training_post_frozen_r2(
    subject_ids=None,
    project_root=PROJECT_ROOT,
    reference_dir=None,
    persist=True,
):
    """Append full-dataset Session-5 training r² evaluations to the cohort table.

    The frozen Session-5 transform is applied unchanged to all valid trials in
    the four non-practice Session-5 training runs.  All participants are
    evaluated before the existing cohort parquet is changed.  When ``persist``
    is true, one atomic replacement appends the validated rows and a separate
    audit log; a failed participant leaves the prior parquet untouched.
    """
    subjects = [str(subject).lower().strip() for subject in (
        EXPECTED_SUBJECTS if subject_ids is None else subject_ids
    )]
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("subject_ids must contain one or more unique participant IDs.")
    unknown = sorted(set(subjects).difference(EXPECTED_SUBJECTS))
    if unknown:
        raise ValueError(f"Unknown participant ID(s): {unknown}.")
    reference_dir = (
        Path(reference_dir) if reference_dir is not None
        else REPO_ROOT / "analyses" / "session5_feature_references"
    )
    r2_path = reference_dir / "longitudinal_top30_r2.parquet"
    log_path = reference_dir / "longitudinal_top30_r2_session5_training_post_build_log.parquet"
    if not r2_path.exists():
        raise FileNotFoundError(f"Cannot append post-training r²: missing {r2_path}.")
    existing = pd.read_parquet(r2_path)
    identity = [
        "subject_id", "decoder_side", "aggregation_level", "evaluation_task",
        "session_id", "run_id", "rank",
    ]
    required_existing = set(identity).union({"r2_evaluation", "feature_index_zero_based", "group"})
    missing_existing = required_existing.difference(existing.columns)
    if missing_existing:
        raise ValueError(f"{r2_path.name} is missing required columns {sorted(missing_existing)}.")
    if existing.duplicated(identity).any():
        raise RuntimeError(f"{r2_path.name} has duplicate feature identities before append.")
    existing_post = existing.loc[existing["evaluation_task"].eq("session5_training_post")].copy()
    if not existing_post.empty:
        expected_existing_rows = len(subjects) * 2 * (SESSION5_TRAINING_RUNS + 1) * 30
        requested_post = existing_post.loc[
            existing_post["subject_id"].astype(str).str.lower().isin(subjects)
        ].copy()
        if (
            len(requested_post) != expected_existing_rows
            or requested_post.duplicated(identity).any()
            or not requested_post["r2_evaluation"].between(0, 1).all()
            or set(requested_post["subject_id"].astype(str).str.lower()) != set(subjects)
        ):
            raise ValueError(
                "Session-5 post-training rows already exist but do not form a complete, "
                "valid append for the requested participants. Refusing to overwrite them."
            )
        coverage = requested_post.groupby(
            ["subject_id", "decoder_side", "aggregation_level"], dropna=False
        ).size()
        expected_coverage = {120, 30}
        if set(coverage.to_numpy(dtype=int)) != expected_coverage:
            raise ValueError("Existing post-training rows have incomplete run/session feature coverage.")
        print("Session-5 post-training r² append already present and validated; no files changed.")
        existing_log = pd.read_parquet(log_path) if log_path.exists() else pd.DataFrame()
        return {
            "post_training_inputs": None,
            "appended_r2": requested_post.sort_values(identity, kind="stable").reset_index(drop=True),
            "append_log": existing_log,
            "r2_path": r2_path,
            "append_log_path": log_path,
            "persisted": False,
        }
    original = existing.copy()
    post_inputs = build_session5_training_post_evaluation_manifest(
        subject_ids=subjects, project_root=project_root
    )
    expected_manifest_runs = len(subjects) * SESSION5_TRAINING_RUNS
    if len(post_inputs["manifest"]) != expected_manifest_runs:
        raise RuntimeError(
            f"Post-training manifest has {len(post_inputs['manifest'])} runs; "
            f"expected {expected_manifest_runs}."
        )
    if not post_inputs["issues"].empty:
        raise RuntimeError("Post-training manifest contains run issues; refusing a partial append.")

    result_tables, log_rows = [], []
    print(f"\nSESSION-5 FULL-DATA POST-TRAINING R² APPEND: {len(subjects)} participant(s)")
    for subject_number, subject_id in enumerate(subjects, start=1):
        print(f"\n[{subject_number}/{len(subjects)}] {subject_id}")
        try:
            result = evaluate_frozen_session5_features_for_subject(
                subject_id, post_inputs, reference_dir=reference_dir
            )
            table = result["r2_table"].copy()
            expected_rows = 2 * (SESSION5_TRAINING_RUNS + 1) * 30
            if len(table) != expected_rows:
                raise RuntimeError(
                    f"Expected {expected_rows} post-training r² rows, got {len(table)}."
                )
            if set(table["evaluation_task"].unique()) != {"session5_training_post"}:
                raise RuntimeError("Evaluation output contains an unexpected task label.")
            if set(table["session_id"].unique()) != {5}:
                raise RuntimeError("Post-training evaluation output must be Session 5 only.")
            if table.duplicated(identity).any() or not table["r2_evaluation"].between(0, 1).all():
                raise RuntimeError("Post-training output has duplicate identities or invalid r² values.")
            for side in ("right", "left"):
                reference = load_session5_frozen_feature_reference(subject_id, side, output_dir=reference_dir)
                expected_features = reference["selected_coordinates"].sort_values("rank")
                side_rows = table.loc[table["decoder_side"] == side]
                for _, subset in side_rows.groupby(
                    ["aggregation_level", "run_id"], dropna=False, sort=False
                ):
                    observed = subset.sort_values("rank")
                    if not np.array_equal(
                        observed["feature_index_zero_based"].to_numpy(dtype=int),
                        expected_features["feature_index_zero_based"].to_numpy(dtype=int),
                    ):
                        raise RuntimeError(f"{subject_id} {side}: evaluated features differ from frozen top-30 reference.")
                run_rows = side_rows.loc[side_rows["aggregation_level"] == "run"]
                session_rows = side_rows.loc[side_rows["aggregation_level"] == "session"]
                if (
                    set(run_rows["run_id"].dropna().astype(int)) != set(range(1, SESSION5_TRAINING_RUNS + 1))
                    or len(run_rows) != SESSION5_TRAINING_RUNS * 30
                    or len(session_rows) != 30
                    or not session_rows["n_contributing_runs"].eq(SESSION5_TRAINING_RUNS).all()
                ):
                    raise RuntimeError(f"{subject_id} {side}: post-training run/session coverage is incomplete.")
            result_tables.append(table)
            log_rows.append({
                "subject_id": subject_id,
                "group": get_subject_group(subject_id),
                "append_status": "validated",
                "n_r2_rows": int(len(table)),
                "n_validated_runs": int(len(result["run_qc"])),
                "n_session_subsets_per_decoder": 1,
                "error_type": None,
                "error_message": None,
                "attempted_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            })
            print(f"{subject_id}: post-training rows validated; held in memory pending atomic cohort append.")
        except Exception as exc:
            log_rows.append({
                "subject_id": subject_id,
                "group": get_subject_group(subject_id),
                "append_status": "failed",
                "n_r2_rows": 0,
                "n_validated_runs": 0,
                "n_session_subsets_per_decoder": 0,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "attempted_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            })
            raise RuntimeError(
                f"{subject_id}: post-training append validation failed; existing r² parquet was not changed. {exc}"
            ) from exc

    appended = pd.concat(result_tables, ignore_index=True)
    expected_appended_rows = len(subjects) * 2 * (SESSION5_TRAINING_RUNS + 1) * 30
    if len(appended) != expected_appended_rows or appended.duplicated(identity).any():
        raise RuntimeError("Combined post-training append rows have an unexpected count or duplicate identities.")
    overlap = appended.merge(existing[identity], on=identity, how="inner")
    if not overlap.empty:
        raise RuntimeError("Post-training append would overlap an existing r² identity.")
    combined = pd.concat([existing, appended[existing.columns]], ignore_index=True)
    if combined.duplicated(identity).any():
        raise RuntimeError("Combined r² table has duplicate feature identities after append.")
    if len(combined) != len(existing) + len(appended):
        raise RuntimeError("Combined r² row count does not match the validated append.")
    audit_log = pd.DataFrame(log_rows)
    if persist:
        temporary_r2_path = r2_path.with_name(f".{r2_path.stem}.session5_post.tmp.parquet")
        combined.to_parquet(temporary_r2_path, index=False)
        reread = pd.read_parquet(temporary_r2_path)
        prior_reread = reread.loc[~reread["evaluation_task"].eq("session5_training_post")]
        pd.testing.assert_frame_equal(
            original.sort_values(identity, kind="stable").reset_index(drop=True),
            prior_reread.sort_values(identity, kind="stable").reset_index(drop=True),
            check_like=False,
        )
        if len(reread) != len(combined) or reread.duplicated(identity).any():
            raise RuntimeError("Temporary post-training r² parquet failed read-back validation.")
        os.replace(temporary_r2_path, r2_path)
        temporary_log_path = log_path.with_name(f".{log_path.stem}.tmp.parquet")
        audit_log.to_parquet(temporary_log_path, index=False)
        os.replace(temporary_log_path, log_path)
        print(f"Atomic append completed: {len(appended)} Session-5 post-training rows added.")
        print(f"Updated r² table: {r2_path}")
        print(f"Append audit log: {log_path}")
    else:
        print("Dry run passed: no files were changed.")
    return {
        "post_training_inputs": post_inputs,
        "appended_r2": appended.sort_values(identity, kind="stable").reset_index(drop=True),
        "append_log": audit_log,
        "r2_path": r2_path,
        "append_log_path": log_path,
        "persisted": bool(persist),
    }


def run_checkpointed_longitudinal_frozen_r2_evaluation(
    subject_ids=None,
    project_root=PROJECT_ROOT,
    reference_dir=None,
    resume=True,
):
    """Evaluate frozen Session-5 features across a cohort with checkpoints.

    The longitudinal manifest is validated once. Each participant's expensive
    filtering, epoching, frozen-transform application, and both r2 levels are
    then saved immediately. Existing participants are skipped only after their
    run- and session-level rows match the current validated manifest.
    """
    subjects = [str(subject).lower().strip() for subject in (
        EXPECTED_SUBJECTS if subject_ids is None else subject_ids
    )]
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("subject_ids must contain one or more unique participant IDs.")
    unknown = sorted(set(subjects).difference(EXPECTED_SUBJECTS))
    if unknown:
        raise ValueError(f"Unknown participant ID(s): {unknown}.")
    reference_dir = Path(reference_dir) if reference_dir is not None else REPO_ROOT / "analyses" / "session5_feature_references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    r2_path = reference_dir / "longitudinal_top30_r2.parquet"
    log_path = reference_dir / "longitudinal_top30_r2_build_log.parquet"
    longitudinal_inputs = build_longitudinal_evaluation_manifest(
        subject_ids=subjects, project_root=project_root
    )

    def _existing_complete_subject(subject_id):
        if not r2_path.exists():
            return False
        try:
            table = pd.read_parquet(r2_path)
            subject_rows = table.loc[table["subject_id"].astype(str).str.lower().eq(subject_id)].copy()
            manifest_rows = longitudinal_inputs["manifest"].loc[
                longitudinal_inputs["manifest"]["subject_id"].astype(str).str.lower().eq(subject_id)
            ]
            expected_run_keys = set(
                zip(
                    manifest_rows["evaluation_task"], manifest_rows["session_id"].astype(int),
                    manifest_rows["run_id"].astype(int),
                )
            )
            expected_session_keys = set(
                zip(manifest_rows["evaluation_task"], manifest_rows["session_id"].astype(int))
            )
            for side in ("right", "left"):
                side_rows = subject_rows.loc[subject_rows["decoder_side"] == side]
                run_rows = side_rows.loc[side_rows["aggregation_level"] == "run"]
                session_rows = side_rows.loc[side_rows["aggregation_level"] == "session"]
                observed_run_keys = set(
                    zip(run_rows["evaluation_task"], run_rows["session_id"].astype(int), run_rows["run_id"].astype(int))
                )
                observed_session_keys = set(
                    zip(session_rows["evaluation_task"], session_rows["session_id"].astype(int))
                )
                if (
                    observed_run_keys != expected_run_keys
                    or observed_session_keys != expected_session_keys
                    or len(run_rows) != 30 * len(expected_run_keys)
                    or len(session_rows) != 30 * len(expected_session_keys)
                    or not run_rows.groupby(["evaluation_task", "session_id", "run_id"], dropna=False).size().eq(30).all()
                    or not session_rows.groupby(["evaluation_task", "session_id"], dropna=False).size().eq(30).all()
                    or not side_rows["r2_evaluation"].between(0, 1).all()
                ):
                    return False
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            return False
        return True

    def _checkpoint_log(new_rows):
        new_log = pd.DataFrame(new_rows)
        if log_path.exists():
            existing = pd.read_parquet(log_path)
            existing = existing.loc[
                ~existing["subject_id"].astype(str).str.lower().isin(
                    new_log["subject_id"].astype(str).str.lower()
                )
            ]
            combined = pd.concat([existing, new_log], ignore_index=True)
        else:
            combined = new_log
        if combined.duplicated("subject_id").any():
            raise RuntimeError("Longitudinal r2 build log has duplicate participant rows.")
        temporary_path = log_path.with_name(f".{log_path.stem}.tmp.parquet")
        combined.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, log_path)
        return combined

    print(f"\nLONGITUDINAL FROZEN TOP-30 R2 COHORT EVALUATION: {len(subjects)} participant(s)")
    final_log = None
    for subject_number, subject_id in enumerate(subjects, start=1):
        print(f"\n[{subject_number}/{len(subjects)}] {subject_id}")
        if resume and _existing_complete_subject(subject_id):
            final_log = _checkpoint_log([{
                "subject_id": subject_id,
                "group": get_subject_group(subject_id),
                "evaluation_status": "skipped_existing",
                "n_r2_rows": int(pd.read_parquet(r2_path).query("subject_id == @subject_id").shape[0]),
                "error_type": None,
                "error_message": None,
                "attempted_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }])
            print(f"{subject_id}: existing complete run/session r2 rows verified; skipped.")
            continue
        try:
            subject_result = evaluate_frozen_session5_features_for_subject(
                subject_id, longitudinal_inputs, reference_dir=reference_dir
            )
            save_longitudinal_frozen_r2_for_subject(subject_result, output_dir=reference_dir)
            final_log = _checkpoint_log([{
                "subject_id": subject_id,
                "group": get_subject_group(subject_id),
                "evaluation_status": "completed",
                "n_r2_rows": int(len(subject_result["r2_table"])),
                "n_run_subsets_per_decoder": int(
                    (subject_result["r2_table"]["aggregation_level"] == "run").sum() / 60
                ),
                "n_session_subsets_per_decoder": int(
                    (subject_result["r2_table"]["aggregation_level"] == "session").sum() / 60
                ),
                "error_type": None,
                "error_message": None,
                "attempted_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }])
        except Exception as exc:
            final_log = _checkpoint_log([{
                "subject_id": subject_id,
                "group": get_subject_group(subject_id),
                "evaluation_status": "failed",
                "n_r2_rows": 0,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "attempted_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }])
            print(f"WARNING: {subject_id} frozen-feature evaluation failed; checkpointed and continuing. {exc}")
    if final_log is None:
        raise RuntimeError("No longitudinal frozen-r2 evaluation records were generated.")
    print("\nLongitudinal frozen top-30 r2 cohort evaluation finished.")
    print(final_log["evaluation_status"].value_counts().to_string())
    return {
        "longitudinal_inputs": longitudinal_inputs,
        "build_log": final_log.sort_values("subject_id", kind="stable").reset_index(drop=True),
        "r2_path": r2_path,
        "build_log_path": log_path,
    }


def summarize_session5_top30_selection_frequency(reference_dir=None):
    """Summarize how often each frozen component-time feature was selected.

    Frequencies are calculated separately by group and lateralized decoder,
    with each participant contributing at most once to a candidate coordinate.
    The complete candidate time grid comes from the saved transforms, so
    coordinates never selected by anyone are retained with 0% frequency.
    """
    reference_dir = Path(reference_dir) if reference_dir is not None else REPO_ROOT / "analyses" / "session5_feature_references"
    top30_path = reference_dir / "session5_reference_top30.parquet"
    transform_paths = sorted((reference_dir / "transforms").glob("*_frozen_reference.npz"))
    if not top30_path.exists() or not transform_paths:
        raise FileNotFoundError(
            "Expected frozen-reference top-30 table and transform files in "
            f"{reference_dir}."
        )
    top30 = pd.read_parquet(top30_path)
    required = {"subject_id", "decoder_side", "rank", "component", "time_index", "time_s"}
    missing = required.difference(top30.columns)
    if missing:
        raise ValueError(f"{top30_path.name} is missing required columns {sorted(missing)}.")
    if "group" not in top30.columns:
        top30["group"] = top30["subject_id"].map(get_subject_group)
    if top30["group"].isna().any():
        raise ValueError("Could not derive a group label for every top-30 feature row.")
    expected_subjects = set(EXPECTED_SUBJECTS)
    if set(top30["subject_id"].astype(str).str.lower()) != expected_subjects:
        raise ValueError("Top-30 feature table does not cover the full expected cohort.")
    expected_counts = {"bci": 16, "control": 16}
    observed_counts = top30.groupby("group")["subject_id"].nunique().to_dict()
    if observed_counts != expected_counts:
        raise ValueError(f"Unexpected group coverage in top-30 table: {observed_counts}.")
    if not top30.groupby(["subject_id", "decoder_side"]).size().eq(30).all():
        raise ValueError("Every participant/decoder reference must contain exactly 30 selected features.")
    selected_identity = ["subject_id", "decoder_side", "component", "time_index"]
    if top30.duplicated(selected_identity).any():
        raise ValueError("A participant selected the same component-time feature more than once.")

    with np.load(transform_paths[0], allow_pickle=False) as archive:
        epoch_time = np.asarray(archive["epoch_time_s"], dtype=float)
        resampled_indices = np.asarray(archive["resampled_indices"], dtype=int)
    if resampled_indices.ndim != 1 or len(resampled_indices) == 0:
        raise ValueError("Frozen transform has an invalid resampled time-index vector.")
    candidate_time_s = epoch_time[resampled_indices]
    for transform_path in transform_paths:
        with np.load(transform_path, allow_pickle=False) as archive:
            if not np.array_equal(np.asarray(archive["resampled_indices"], dtype=int), resampled_indices):
                raise ValueError(f"{transform_path.name} has a different resampling grid.")
            if not np.array_equal(np.asarray(archive["epoch_time_s"], dtype=float), epoch_time):
                raise ValueError(f"{transform_path.name} has a different epoch time axis.")

    grid = pd.MultiIndex.from_product(
        [["bci", "control"], ["right", "left"], [1, 2], resampled_indices],
        names=["group", "decoder_side", "component", "time_index"],
    ).to_frame(index=False)
    time_lookup = pd.DataFrame({"time_index": resampled_indices, "time_s": candidate_time_s})
    grid = grid.merge(time_lookup, on="time_index", how="left", validate="many_to_one")
    selected_counts = top30.groupby(
        ["group", "decoder_side", "component", "time_index"], as_index=False
    )["subject_id"].nunique().rename(columns={"subject_id": "n_selected_participants"})
    summary = grid.merge(
        selected_counts,
        on=["group", "decoder_side", "component", "time_index"], how="left",
        validate="one_to_one",
    )
    summary["n_selected_participants"] = summary["n_selected_participants"].fillna(0).astype(int)
    summary["n_group_participants"] = summary["group"].map(expected_counts).astype(int)
    summary["selection_frequency_pct"] = 100 * (
        summary["n_selected_participants"] / summary["n_group_participants"]
    )
    if not summary["selection_frequency_pct"].between(0, 100).all():
        raise RuntimeError("Selection frequencies must lie between 0% and 100%.")
    print("Session-5 top-30 selection-frequency summary passed.")
    print(
        f"  Participants: BCI={expected_counts['bci']}, mental rehearsal={expected_counts['control']}; "
        f"candidate times/component={len(resampled_indices)}."
    )
    return summary.sort_values(
        ["decoder_side", "group", "component", "time_index"], kind="stable"
    ).reset_index(drop=True)


def plot_session5_top30_selection_frequency(
    selection_frequency=None,
    reference_dir=None,
    output_dir=None,
    filename_stem="session5_top30_feature_selection_frequency",
):
    """Plot group-wise stability of selected Session-5 component-time features."""
    import matplotlib.pyplot as plt

    if selection_frequency is None:
        selection_frequency = summarize_session5_top30_selection_frequency(reference_dir)
    frequency = selection_frequency.copy()
    required = {
        "group", "decoder_side", "component", "time_s", "selection_frequency_pct",
        "n_group_participants",
    }
    missing = required.difference(frequency.columns)
    if missing:
        raise ValueError(f"selection_frequency is missing columns {sorted(missing)}.")
    expected_rows = 2 * 2 * 2 * frequency["time_s"].nunique()
    if len(frequency) != expected_rows:
        raise ValueError(
            f"Expected a complete 2 group x 2 decoder x 2 component grid; got {len(frequency)} rows."
        )
    if frequency.groupby(["group", "decoder_side", "component"]).size().nunique() != 1:
        raise ValueError("Each group/decoder/component panel must use the same candidate time grid.")
    times_ms = np.sort(frequency["time_s"].unique() * 1000)
    if len(times_ms) < 2:
        raise ValueError("At least two resampled time points are required for a selection-frequency plot.")
    spacing_ms = float(np.median(np.diff(times_ms)))
    component_gap_ms = max(120.0, 10 * spacing_ms)
    component_1_x = times_ms
    component_2_x = times_ms + (times_ms[-1] - times_ms[0]) + component_gap_ms
    x_by_component = {1: component_1_x, 2: component_2_x}
    ymax = max(25.0, float(np.ceil(frequency["selection_frequency_pct"].max() / 10.0) * 10.0))
    ymax = min(100.0, ymax)
    colors = {"bci": "#DD8452", "control": "#4C72B0"}
    group_titles = {"bci": "BCI training", "control": "Mental rehearsal"}
    decoder_titles = {
        "right": "Right decoder\n(right distractor vs no distractor)",
        "left": "Left decoder\n(left distractor vs no distractor)",
    }
    output_dir = Path(output_dir) if output_dir is not None else REPO_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{filename_stem}.pdf"
    png_path = output_dir / f"{filename_stem}.png"

    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5), sharex=True, sharey=True)
        for row, decoder_side in enumerate(("right", "left")):
            for column, group in enumerate(("bci", "control")):
                ax = axes[row, column]
                panel = frequency.loc[
                    (frequency["decoder_side"] == decoder_side) & (frequency["group"] == group)
                ]
                for component in (1, 2):
                    component_rows = panel.loc[panel["component"] == component].sort_values("time_s")
                    ax.plot(
                        x_by_component[component], component_rows["selection_frequency_pct"],
                        color=colors[group], marker="o", markersize=2.3, linewidth=1.1,
                        markeredgewidth=0, clip_on=True,
                    )
                gap_start = component_1_x[-1] + spacing_ms / 2
                gap_stop = component_2_x[0] - spacing_ms / 2
                ax.axvspan(gap_start, gap_stop, color="0.95", zorder=0)
                ax.axvline(gap_start, color="0.78", linewidth=0.5, zorder=1)
                ax.axvline(gap_stop, color="0.78", linewidth=0.5, zorder=1)
                ax.set_ylim(0, ymax)
                ax.set_xlim(component_1_x[0] - spacing_ms, component_2_x[-1] + spacing_ms)
                ax.set_yticks(np.arange(0, ymax + 0.1, 20 if ymax >= 60 else 10))
                ax.tick_params(direction="out", length=2.5, pad=2)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if row == 0:
                    ax.set_title(group_titles[group], fontsize=8, fontweight="bold", pad=17)
                    ax.text(component_1_x.mean(), ymax * 1.035, "Component 1", ha="center", va="bottom", fontsize=6.5)
                    ax.text(component_2_x.mean(), ymax * 1.035, "Component 2", ha="center", va="bottom", fontsize=6.5)
                if column == 0:
                    ax.set_ylabel("Participants selecting feature (%)")
                    ax.text(
                        -0.37, 0.5, decoder_titles[decoder_side], transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=7.5,
                    )
                if row == 1:
                    tick_labels = np.array([200, 400, 600, 800, 1000])
                    tick_values = np.array([
                        times_ms[np.abs(times_ms - target).argmin()] for target in tick_labels
                    ])
                    ticks = np.concatenate([
                        tick_values,
                        tick_values + (times_ms[-1] - times_ms[0]) + component_gap_ms,
                    ])
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([str(int(value)) for value in np.concatenate([tick_labels, tick_labels])])
                    ax.set_xlabel("Feature time after stimulus onset (ms)")
                ax.text(-0.12, 1.08, chr(ord("a") + row * 2 + column), transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="top")
        fig.subplots_adjust(left=0.20, right=0.99, bottom=0.14, top=0.88, hspace=0.34, wspace=0.20)
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Session-5 top-30 selection-frequency figure passed.")
    print(f"  Shared y-axis limits: 0 to {ymax:.0f}%.")
    print(f"  Saved PDF: {pdf_path}")
    print(f"  Saved PNG: {png_path}")
    return {"pdf_path": pdf_path, "png_path": png_path, "selection_frequency": frequency, "y_limits": (0.0, ymax)}


def summarize_longitudinal_top30_r2_by_session(reference_dir=None, n_features=30):
    """Average frozen-feature r² within subject, then summarize it by assessment.

    This uses only trial-pooled session-level estimates and retains the seven
    assessments: full-dataset Session-1 training (pre-training), decoding on
    intervention days 1--5, and full-dataset Session-5 training (post-training).
    Right and left decoders remain separate because their feature sets and
    target classes differ.
    """
    if not isinstance(n_features, (int, np.integer)) or not 1 <= int(n_features) <= 30:
        raise ValueError("n_features must be an integer from 1 through 30.")
    n_features = int(n_features)
    reference_dir = (
        Path(reference_dir) if reference_dir is not None
        else REPO_ROOT / "analyses" / "session5_feature_references"
    )
    r2_path = reference_dir / "longitudinal_top30_r2.parquet"
    if not r2_path.exists():
        raise FileNotFoundError(f"Longitudinal frozen-feature r² table not found: {r2_path}")
    r2 = pd.read_parquet(r2_path)
    required = {
        "subject_id", "group", "decoder_side", "aggregation_level",
        "evaluation_task", "session_id", "rank", "r2_evaluation",
    }
    missing = required.difference(r2.columns)
    if missing:
        raise ValueError(f"{r2_path.name} is missing required columns {sorted(missing)}.")
    assessment_specs = {
        ("session1_training_pre", 1): (0, "Pre-training"),
        ("decoding", 1): (1, "Intervention day 1"),
        ("decoding", 2): (2, "Intervention day 2"),
        ("decoding", 3): (3, "Intervention day 3"),
        ("decoding", 4): (4, "Intervention day 4"),
        ("decoding", 5): (5, "Intervention day 5"),
        ("session5_training_post", 5): (6, "Post-training"),
    }
    subset = r2.loc[r2["aggregation_level"].eq("session")].copy()
    subset["_assessment_key"] = list(zip(subset["evaluation_task"], subset["session_id"]))
    subset = subset.loc[subset["_assessment_key"].isin(assessment_specs)].copy()
    subset[["assessment_order", "assessment_label"]] = pd.DataFrame(
        subset["_assessment_key"].map(assessment_specs).tolist(), index=subset.index
    )
    subset = subset.drop(columns="_assessment_key")
    subset = subset.loc[subset["rank"].between(1, n_features)].copy()
    if subset.empty:
        raise ValueError("No pooled session-level r² rows were found for the seven assessments.")
    subset["subject_id"] = subset["subject_id"].astype(str).str.lower()
    if not subset["r2_evaluation"].between(0, 1).all():
        raise ValueError("Session-level feature r² values must lie between 0 and 1.")
    if set(subset["group"].unique()) != {"bci", "control"}:
        raise ValueError("Both BCI and control groups must be present in the r² table.")
    if set(subset["decoder_side"].unique()) != {"right", "left"}:
        raise ValueError("Both right and left decoder sides must be present in the r² table.")
    if subset.duplicated(["subject_id", "decoder_side", "assessment_order", "rank"]).any():
        raise ValueError("Duplicate subject/decoder/assessment/rank r² rows were found.")

    feature_counts = subset.groupby(
        ["subject_id", "group", "decoder_side", "assessment_order", "assessment_label"], sort=False
    )["rank"].agg(["size", "nunique"])
    if not feature_counts["size"].eq(n_features).all() or not feature_counts["nunique"].eq(n_features).all():
        invalid = feature_counts.loc[
            ~feature_counts["size"].eq(30) | ~feature_counts["nunique"].eq(30)
        ]
        raise ValueError(
            f"Every subject/decoder/assessment must have exactly {n_features} unique frozen features; "
            f"invalid cells: {invalid.to_dict('index')}"
        )
    expected_subjects = set(EXPECTED_SUBJECTS)
    observed_subjects = set(subset["subject_id"])
    if observed_subjects != expected_subjects:
        raise ValueError(
            "Assessment-level r² does not cover the full expected cohort: "
            f"missing={sorted(expected_subjects - observed_subjects)}, "
            f"unexpected={sorted(observed_subjects - expected_subjects)}."
        )

    subject_session = (
        subset.groupby(
            ["subject_id", "group", "decoder_side", "assessment_order", "assessment_label"],
            as_index=False, sort=True,
        )
        .agg(mean_r2=("r2_evaluation", "mean"), n_features=("r2_evaluation", "size"))
    )
    group_session = (
        subject_session.groupby(
            ["group", "decoder_side", "assessment_order", "assessment_label"],
            as_index=False, sort=True,
        )
        .agg(mean_r2=("mean_r2", "mean"), sd_r2=("mean_r2", "std"),
             n_subjects=("subject_id", "nunique"))
    )
    group_session["sem_r2"] = group_session["sd_r2"] / np.sqrt(group_session["n_subjects"])
    if len(group_session) != 28 or not group_session["n_subjects"].eq(16).all():
        raise ValueError(
            "Expected 16 participants in every group/decoder/assessment cell; got "
            f"{group_session[['group', 'decoder_side', 'assessment_label', 'n_subjects']].to_dict('records')}"
        )
    print(f"Longitudinal assessment-level top-{n_features} r² summary passed.")
    print(f"  Pooled-session r²; {n_features} features per participant/decoder/assessment.")
    print("  Assessments: pre-training, intervention days 1-5, post-training; BCI=16, mental rehearsal=16.")
    return {
        "subject_session": subject_session,
        "group_session": group_session,
        "source_path": r2_path,
        "n_features": n_features,
    }


def plot_longitudinal_top30_r2_by_session(
    r2_summary=None,
    reference_dir=None,
    output_dir=None,
    filename_stem=None,
):
    """Plot group-average frozen top-30 r² trajectories with SEM across seven assessments."""
    import matplotlib.pyplot as plt

    if r2_summary is None:
        r2_summary = summarize_longitudinal_top30_r2_by_session(reference_dir)
    if not {"subject_session", "group_session"}.issubset(r2_summary):
        raise ValueError("r2_summary must contain 'subject_session' and 'group_session' tables.")
    n_features = int(r2_summary.get("n_features", 30))
    if not 1 <= n_features <= 30:
        raise ValueError("r2_summary has an invalid n_features value.")
    subject_session = r2_summary["subject_session"].copy()
    group_session = r2_summary["group_session"].copy()
    required_subject = {"subject_id", "group", "decoder_side", "assessment_order", "mean_r2"}
    required_group = {
        "group", "decoder_side", "assessment_order", "assessment_label",
        "mean_r2", "sem_r2", "n_subjects",
    }
    if missing := required_subject.difference(subject_session.columns):
        raise ValueError(f"subject_session is missing columns {sorted(missing)}.")
    if missing := required_group.difference(group_session.columns):
        raise ValueError(f"group_session is missing columns {sorted(missing)}.")

    output_dir = Path(output_dir) if output_dir is not None else REPO_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename_stem is None:
        filename_stem = f"longitudinal_top{n_features}_feature_r2_by_session"
    colors = {"bci": "#DD8452", "control": "#4C72B0"}
    labels = {"bci": "BCI training", "control": "Mental rehearsal"}
    panel_titles = {
        "right": "Right decoder (right distractor vs no distractor)",
        "left": "Left decoder (left distractor vs no distractor)",
    }
    all_bounds = np.r_[
        (group_session["mean_r2"] - group_session["sem_r2"]).to_numpy(),
        (group_session["mean_r2"] + group_session["sem_r2"]).to_numpy(),
    ]
    y_lower = max(0.0, float(np.floor((all_bounds.min() - 0.001) / 0.01) * 0.01))
    y_upper = float(np.ceil((all_bounds.max() + 0.001) / 0.01) * 0.01)
    if y_upper <= y_lower:
        y_upper = y_lower + 0.01

    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.linewidth": 0.65, "xtick.major.width": 0.65,
        "ytick.major.width": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), sharey=True)
        for panel_index, (ax, side) in enumerate(zip(axes, ("right", "left"))):
            for group in ("bci", "control"):
                means = group_session.loc[
                    (group_session["decoder_side"] == side) & (group_session["group"] == group)
                ].sort_values("assessment_order")
                x = means["assessment_order"].to_numpy(dtype=float)
                mean = means["mean_r2"].to_numpy(dtype=float)
                sem = means["sem_r2"].to_numpy(dtype=float)
                ax.fill_between(x, mean - sem, mean + sem, color=colors[group], alpha=0.16, linewidth=0)
                ax.plot(x, mean, color=colors[group], marker="o", markersize=3.1,
                        linewidth=1.25, label=labels[group], zorder=3)
            ax.set_title(panel_titles[side], pad=6)
            ax.set_xlim(-0.20, 6.20)
            ax.set_ylim(y_lower, y_upper)
            ax.set_xticks(np.arange(0, 7))
            ax.set_xticklabels([
                "Pre-\ntraining", "Day 1", "Day 2", "Day 3", "Day 4", "Day 5",
                "Post-\ntraining",
            ])
            ax.tick_params(direction="out", length=3.0, width=0.65)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel("Training timeline (intervention days 1–5)")
            ax.text(-0.20, 1.02, chr(ord("a") + panel_index), transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="bottom")
        axes[0].set_ylabel(f"Mean r² across frozen top-{n_features} features")
        axes[1].legend(frameon=False, loc="upper left", handlelength=1.7)
        fig.text(0.50, 0.01,
                 "Lines and shading, group mean ± SEM across participants (n = 16/group).",
                 ha="center", va="bottom", fontsize=6.3)
        fig.subplots_adjust(left=0.11, right=0.99, bottom=0.24, top=0.88, wspace=0.18)
        pdf_path = output_dir / f"{filename_stem}.pdf"
        png_path = output_dir / f"{filename_stem}.png"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Longitudinal top-30 r² figure passed.")
    print(f"  Shared y-axis limits: {y_lower:.3f} to {y_upper:.3f} r².")
    print(f"  Saved PDF: {pdf_path}")
    print(f"  Saved PNG: {png_path}")
    return {"pdf_path": pdf_path, "png_path": png_path, "y_limits": (y_lower, y_upper)}


def summarize_longitudinal_top30_r2_combined_decoders(reference_dir=None, n_features=30):
    """Combine right/left decoder r² equally within participant and assessment.

    The aggregation order is fixed: feature r² values are averaged within each
    participant/decoder/assessment, the two decoder means are averaged within
    participant/assessment, then group means and between-participant SEM are
    calculated. This prevents a decoder with more rows from receiving more
    weight.
    """
    base = summarize_longitudinal_top30_r2_by_session(reference_dir, n_features=n_features)
    n_features = int(base["n_features"])
    decoder_subject = base["subject_session"].copy()
    required = {
        "subject_id", "group", "decoder_side", "assessment_order",
        "assessment_label", "mean_r2", "n_features",
    }
    missing = required.difference(decoder_subject.columns)
    if missing:
        raise ValueError(f"Decoder-specific subject table is missing columns {sorted(missing)}.")
    decoder_counts = decoder_subject.groupby(
        ["subject_id", "group", "assessment_order", "assessment_label"], sort=False
    )["decoder_side"].agg(["size", "nunique"])
    if not decoder_counts["size"].eq(2).all() or not decoder_counts["nunique"].eq(2).all():
        raise ValueError("Every participant/assessment must contain exactly one right and one left decoder mean.")
    if not decoder_subject["n_features"].eq(n_features).all():
        raise ValueError(
            f"Each participant/decoder/assessment must represent exactly {n_features} frozen features."
        )
    subject_assessment = (
        decoder_subject.groupby(
            ["subject_id", "group", "assessment_order", "assessment_label"], as_index=False, sort=True
        )
        .agg(mean_r2=("mean_r2", "mean"), n_decoders=("decoder_side", "nunique"))
    )
    group_assessment = (
        subject_assessment.groupby(
            ["group", "assessment_order", "assessment_label"], as_index=False, sort=True
        )
        .agg(mean_r2=("mean_r2", "mean"), sd_r2=("mean_r2", "std"),
             n_subjects=("subject_id", "nunique"))
    )
    group_assessment["sem_r2"] = group_assessment["sd_r2"] / np.sqrt(group_assessment["n_subjects"])
    if len(group_assessment) != 14 or not group_assessment["n_subjects"].eq(16).all():
        raise ValueError("Expected 16 participants in every group/assessment cell after decoder combination.")
    print(f"Combined-decoder longitudinal top-{n_features} r² summary passed.")
    print(f"  Order: mean within {n_features} features -> mean across right/left decoders -> group mean ± SEM.")
    return {
        "subject_assessment": subject_assessment,
        "group_assessment": group_assessment,
        "source_path": base["source_path"],
        "n_features": n_features,
    }


def plot_longitudinal_top30_r2_combined_decoders(
    combined_summary=None,
    reference_dir=None,
    output_dir=None,
    filename_stem=None,
):
    """Plot the equal-weighted combined-decoder frozen-feature r² trajectory."""
    import matplotlib.pyplot as plt

    if combined_summary is None:
        combined_summary = summarize_longitudinal_top30_r2_combined_decoders(reference_dir)
    if "group_assessment" not in combined_summary:
        raise ValueError("combined_summary must contain a 'group_assessment' table.")
    n_features = int(combined_summary.get("n_features", 30))
    if not 1 <= n_features <= 30:
        raise ValueError("combined_summary has an invalid n_features value.")
    summary = combined_summary["group_assessment"].copy()
    required = {"group", "assessment_order", "assessment_label", "mean_r2", "sem_r2", "n_subjects"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"group_assessment is missing columns {sorted(missing)}.")
    if len(summary) != 14 or not summary["n_subjects"].eq(16).all():
        raise ValueError("Combined-decoder plot requires 16 participants in each group/assessment cell.")
    output_dir = Path(output_dir) if output_dir is not None else REPO_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename_stem is None:
        filename_stem = f"longitudinal_top{n_features}_feature_r2_combined_decoders"
    colors = {"bci": "#DD8452", "control": "#4C72B0"}
    labels = {"bci": "BCI training", "control": "Mental rehearsal"}
    bounds = np.r_[
        (summary["mean_r2"] - summary["sem_r2"]).to_numpy(),
        (summary["mean_r2"] + summary["sem_r2"]).to_numpy(),
    ]
    y_lower = max(0.0, float(np.floor((bounds.min() - 0.001) / 0.01) * 0.01))
    y_upper = float(np.ceil((bounds.max() + 0.001) / 0.01) * 0.01)
    if y_upper <= y_lower:
        y_upper = y_lower + 0.01
    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.linewidth": 0.65, "xtick.major.width": 0.65,
        "ytick.major.width": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(4.1, 3.25))
        for group in ("bci", "control"):
            values = summary.loc[summary["group"] == group].sort_values("assessment_order")
            x = values["assessment_order"].to_numpy(dtype=float)
            mean = values["mean_r2"].to_numpy(dtype=float)
            sem = values["sem_r2"].to_numpy(dtype=float)
            ax.fill_between(x, mean - sem, mean + sem, color=colors[group], alpha=0.16, linewidth=0)
            ax.plot(x, mean, color=colors[group], marker="o", markersize=3.1,
                    linewidth=1.25, label=labels[group], zorder=3)
        ax.set_xlim(-0.20, 6.20)
        ax.set_ylim(y_lower, y_upper)
        ax.set_xticks(np.arange(0, 7))
        ax.set_xticklabels(["Pre-\ntraining", "Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Post-\ntraining"])
        ax.set_xlabel("Training timeline (intervention days 1–5)")
        ax.set_ylabel(f"Mean r² across frozen top-{n_features} features and decoders")
        ax.tick_params(direction="out", length=3.0, width=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper left", handlelength=1.7)
        ax.text(-0.13, 1.02, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        fig.text(0.50, 0.01,
                 "Lines and shading, group mean ± SEM across participants (n = 16/group).",
                 ha="center", va="bottom", fontsize=6.3)
        fig.subplots_adjust(left=0.17, right=0.99, bottom=0.25, top=0.95)
        pdf_path = output_dir / f"{filename_stem}.pdf"
        png_path = output_dir / f"{filename_stem}.png"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Combined-decoder longitudinal top-30 r² figure passed.")
    print(f"  Shared uncertainty-aware y-axis limits: {y_lower:.3f} to {y_upper:.3f} r².")
    print(f"  Saved PDF: {pdf_path}")
    print(f"  Saved PNG: {png_path}")
    return {"pdf_path": pdf_path, "png_path": png_path, "y_limits": (y_lower, y_upper)}


def run_top30_combined_decoder_intervention_slope_mixed_model(reference_dir=None):
    """Test whether top-30 combined-decoder r² slopes differ by group on days 1--5.

    The outcome is the equally weighted participant-level mean across the
    right/left decoder means, calculated after averaging each decoder's 30
    frozen features. The model uses only the independent decoding assessments
    on intervention days 1--5; it deliberately excludes pre- and
    post-training task values because the latter is selection-linked.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("statsmodels is required for the mixed-effects slope model.") from exc

    combined = summarize_longitudinal_top30_r2_combined_decoders(
        reference_dir=reference_dir, n_features=30
    )
    data = combined["subject_assessment"].loc[
        combined["subject_assessment"]["assessment_order"].between(1, 5)
    ].copy()
    expected_rows = len(EXPECTED_SUBJECTS) * 5
    if len(data) != expected_rows or data["subject_id"].nunique() != len(EXPECTED_SUBJECTS):
        raise ValueError(
            f"Expected {expected_rows} complete participant-day rows for days 1-5, got {len(data)}."
        )
    day_counts = data.groupby("subject_id")["assessment_order"].nunique()
    if not day_counts.eq(5).all():
        raise ValueError("Every participant must contribute all five intervention-day values.")
    if not (data["mean_r2"] > 0).all() or not np.isfinite(data["mean_r2"]).all():
        raise ValueError("The log-r² mixed model requires finite, strictly positive participant means.")
    expected_group_counts = {"bci": 16, "control": 16}
    observed_group_counts = data.groupby("group")["subject_id"].nunique().to_dict()
    if observed_group_counts != expected_group_counts:
        raise ValueError(f"Unexpected group coverage: {observed_group_counts}.")
    data["intervention_day_centered"] = data["assessment_order"].astype(float) - 3.0
    data["group_bci"] = data["group"].eq("bci").astype(int)
    data["log_mean_r2"] = np.log(data["mean_r2"])

    formula = "log_mean_r2 ~ intervention_day_centered * group_bci"
    fit_attempts = [
        ("random_intercept_and_day_slope", "~intervention_day_centered"),
        ("random_intercept", "1"),
    ]
    fitted = None
    fit_structure = None
    fit_messages = []
    for structure, re_formula in fit_attempts:
        try:
            model = smf.mixedlm(
                formula, data=data, groups=data["subject_id"], re_formula=re_formula
            )
            candidate = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
            random_covariance = np.asarray(candidate.cov_re, dtype=float)
            random_slope_supported = (
                structure == "random_intercept_and_day_slope"
                and candidate.converged
                and random_covariance.shape == (2, 2)
                and np.isfinite(random_covariance).all()
                and random_covariance[1, 1] > 1e-8
            )
            if structure == "random_intercept_and_day_slope" and not random_slope_supported:
                fit_messages.append(
                    "Random day-slope fit was singular, non-converged, or had negligible slope variance; "
                    "used the prespecified random-intercept fallback."
                )
                continue
            if not candidate.converged:
                fit_messages.append(f"{structure} fit did not converge.")
                continue
            fitted = candidate
            fit_structure = structure
            break
        except Exception as exc:
            fit_messages.append(f"{structure} fit failed: {type(exc).__name__}: {exc}")
    if fitted is None:
        raise RuntimeError("Mixed-effects model could not be fit. " + " ".join(fit_messages))

    fixed = fitted.fe_params
    fixed_se = fitted.bse_fe
    fixed_p = fitted.pvalues.loc[fixed.index]
    fixed_cov = fitted.cov_params().loc[fixed.index, fixed.index]
    interaction_term = "intervention_day_centered:group_bci"
    required_terms = {"Intercept", "intervention_day_centered", "group_bci", interaction_term}
    if not required_terms.issubset(fixed.index):
        raise RuntimeError(f"Mixed model lacks required fixed terms: {sorted(required_terms - set(fixed.index))}.")
    z_critical = 1.959963984540054

    def _linear_effect(name, weights):
        weight_vector = pd.Series(0.0, index=fixed.index)
        for term, weight in weights.items():
            weight_vector.loc[term] = weight
        estimate = float(weight_vector @ fixed)
        variance = float(weight_vector @ fixed_cov @ weight_vector)
        if variance < 0:
            raise RuntimeError(f"Negative variance encountered for {name}.")
        se = float(np.sqrt(variance))
        z_value = estimate / se if se > 0 else np.nan
        from scipy.stats import norm
        p_value = float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
        return {
            "effect": name,
            "estimate_log_r2_per_day": estimate,
            "se": se,
            "ci95_low": estimate - z_critical * se,
            "ci95_high": estimate + z_critical * se,
            "z": z_value,
            "p_value": p_value,
            "r2_ratio_per_day": float(np.exp(estimate)),
            "ratio_ci95_low": float(np.exp(estimate - z_critical * se)),
            "ratio_ci95_high": float(np.exp(estimate + z_critical * se)),
        }

    effects = pd.DataFrame([
        _linear_effect("Control daily slope", {"intervention_day_centered": 1.0}),
        _linear_effect("BCI daily slope", {
            "intervention_day_centered": 1.0, interaction_term: 1.0,
        }),
        _linear_effect("BCI minus control daily slope", {interaction_term: 1.0}),
    ])
    fixed_effects = pd.DataFrame({
        "term": fixed.index,
        "estimate": fixed.to_numpy(dtype=float),
        "se": fixed_se.to_numpy(dtype=float),
        "z": (fixed / fixed_se).to_numpy(dtype=float),
        "p_value": fixed_p.to_numpy(dtype=float),
    })
    print("TOP-30 COMBINED-DECODER INTERVENTION-DAY SLOPE MODEL")
    print("  Outcome: log participant mean r²; days 1-5 decoding only.")
    print(f"  Model: {formula} + {fit_structure.replace('_', ' ')}.")
    print("  Direct group trajectory test: BCI minus control daily slope.")
    print(effects.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    if fit_messages:
        print("  Fit notes: " + " ".join(fit_messages))
    return {
        "analysis_data": data.sort_values(["subject_id", "assessment_order"]).reset_index(drop=True),
        "effects": effects,
        "fixed_effects": fixed_effects,
        "model": fitted,
        "model_structure": fit_structure,
        "fit_notes": fit_messages,
        "formula": formula,
    }


def run_top30_decoder_specific_intervention_slope_mixed_model(reference_dir=None):
    """Test decoder-specific BCI/control top-30 r² trajectories on days 1--5.

    This extension keeps right and left decoder means separate and fits a
    day × group × decoder-side model. The three-way interaction tests whether
    the BCI-versus-control slope difference differs by decoder. Planned group
    slope-difference contrasts for the right and left decoder are Holm-adjusted
    as one two-comparison family.
    """
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.multitest import multipletests
        from scipy.stats import norm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("statsmodels and scipy are required for the mixed-effects slope model.") from exc

    base = summarize_longitudinal_top30_r2_by_session(reference_dir, n_features=30)
    data = base["subject_session"].loc[
        base["subject_session"]["assessment_order"].between(1, 5)
    ].copy()
    expected_rows = len(EXPECTED_SUBJECTS) * 2 * 5
    if len(data) != expected_rows or data["subject_id"].nunique() != len(EXPECTED_SUBJECTS):
        raise ValueError(
            f"Expected {expected_rows} complete participant/decoder/day rows, got {len(data)}."
        )
    coverage = data.groupby(["subject_id", "decoder_side"])["assessment_order"].nunique()
    if not coverage.eq(5).all() or set(data["decoder_side"]) != {"right", "left"}:
        raise ValueError("Every participant/decoder must contribute all five intervention-day values.")
    if not data["n_features"].eq(30).all() or not (data["mean_r2"] > 0).all():
        raise ValueError("Decoder-specific model requires 30 strictly positive top-feature means per row.")
    if data.groupby("group")["subject_id"].nunique().to_dict() != {"bci": 16, "control": 16}:
        raise ValueError("Decoder-specific model requires 16 BCI and 16 control participants.")
    data["intervention_day_centered"] = data["assessment_order"].astype(float) - 3.0
    data["group_bci"] = data["group"].eq("bci").astype(int)
    data["decoder_left"] = data["decoder_side"].eq("left").astype(int)
    data["log_mean_r2"] = np.log(data["mean_r2"])

    formula = "log_mean_r2 ~ intervention_day_centered * group_bci * decoder_left"
    fit_attempts = [
        ("random_intercept_and_day_slope", "~intervention_day_centered"),
        ("random_intercept", "1"),
    ]
    fitted, fit_structure, fit_messages = None, None, []
    for structure, re_formula in fit_attempts:
        try:
            model = smf.mixedlm(formula, data=data, groups=data["subject_id"], re_formula=re_formula)
            candidate = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
            covariance = np.asarray(candidate.cov_re, dtype=float)
            slope_supported = (
                structure == "random_intercept_and_day_slope" and candidate.converged
                and covariance.shape == (2, 2) and np.isfinite(covariance).all()
                and covariance[1, 1] > 1e-8
            )
            if structure == "random_intercept_and_day_slope" and not slope_supported:
                fit_messages.append("Random day-slope fit was not supported; used random-intercept fallback.")
                continue
            if not candidate.converged:
                fit_messages.append(f"{structure} fit did not converge.")
                continue
            fitted, fit_structure = candidate, structure
            break
        except Exception as exc:
            fit_messages.append(f"{structure} fit failed: {type(exc).__name__}: {exc}")
    if fitted is None:
        raise RuntimeError("Decoder-specific mixed model could not be fit. " + " ".join(fit_messages))

    fixed = fitted.fe_params
    fixed_cov = fitted.cov_params().loc[fixed.index, fixed.index]
    terms = {
        "day": "intervention_day_centered",
        "day_group": "intervention_day_centered:group_bci",
        "day_group_decoder": "intervention_day_centered:group_bci:decoder_left",
    }
    missing_terms = set(terms.values()).difference(fixed.index)
    if missing_terms:
        raise RuntimeError(f"Mixed model lacks required terms: {sorted(missing_terms)}.")
    z_critical = 1.959963984540054

    def _contrast(name, weights):
        vector = pd.Series(0.0, index=fixed.index)
        for term, weight in weights.items():
            vector.loc[term] = weight
        estimate = float(vector @ fixed)
        variance = float(vector @ fixed_cov @ vector)
        if variance < 0:
            raise RuntimeError(f"Negative contrast variance for {name}.")
        se = float(np.sqrt(variance))
        z_value = estimate / se if se > 0 else np.nan
        p_value = float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
        low, high = estimate - z_critical * se, estimate + z_critical * se
        return {
            "contrast": name,
            "estimate_log_r2_per_day": estimate,
            "se": se,
            "ci95_low": low,
            "ci95_high": high,
            "z": z_value,
            "p_value": p_value,
            "r2_ratio_per_day": float(np.exp(estimate)),
            "ratio_ci95_low": float(np.exp(low)),
            "ratio_ci95_high": float(np.exp(high)),
        }

    contrasts = pd.DataFrame([
        _contrast("Right decoder: BCI minus control daily slope", {terms["day_group"]: 1.0}),
        _contrast("Left decoder: BCI minus control daily slope", {
            terms["day_group"]: 1.0, terms["day_group_decoder"]: 1.0,
        }),
    ])
    contrasts["p_value_holm"] = multipletests(contrasts["p_value"], method="holm")[1]
    three_way = _contrast(
        "Decoder-side difference in BCI minus control daily slope",
        {terms["day_group_decoder"]: 1.0},
    )
    fixed_effects = pd.DataFrame({
        "term": fixed.index,
        "estimate": fixed.to_numpy(dtype=float),
        "se": fitted.bse_fe.to_numpy(dtype=float),
        "z": (fixed / fitted.bse_fe).to_numpy(dtype=float),
        "p_value": fitted.pvalues.loc[fixed.index].to_numpy(dtype=float),
    })
    print("TOP-30 DECODER-SPECIFIC INTERVENTION-DAY SLOPE MODEL")
    print("  Outcome: log participant mean r²; days 1-5 decoding only.")
    print(f"  Model: {formula} + {fit_structure.replace('_', ' ')}.")
    print("  Decoder-specific group-slope contrasts use Holm correction (two contrasts).")
    print(contrasts.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("  Three-way interaction:")
    print(pd.DataFrame([three_way]).to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    if fit_messages:
        print("  Fit notes: " + " ".join(fit_messages))
    return {
        "analysis_data": data.sort_values(["subject_id", "decoder_side", "assessment_order"]).reset_index(drop=True),
        "decoder_group_slope_contrasts": contrasts,
        "three_way_interaction": pd.DataFrame([three_way]),
        "fixed_effects": fixed_effects,
        "model": fitted,
        "model_structure": fit_structure,
        "fit_notes": fit_messages,
        "formula": formula,
    }


def run_top30_intervention_day_mixed_anovas(reference_dir=None):
    """Run separate combined, right, and left 2-group × 5-day mixed ANOVAs.

    Each model uses log participant mean r² from decoding-based intervention
    days 1--5, categorical day, a between-participant group factor, and a
    participant random intercept. Type-III Wald chi-square tests provide the
    ANOVA-style tests of day, group, and their interaction. The combined model
    is the primary analysis; the two decoder-specific group × day interactions
    are Holm-corrected as a two-test follow-up family.
    """
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.multitest import multipletests
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("statsmodels is required for the mixed-ANOVA models.") from exc

    combined = summarize_longitudinal_top30_r2_combined_decoders(
        reference_dir=reference_dir, n_features=30
    )["subject_assessment"].copy()
    decoder = summarize_longitudinal_top30_r2_by_session(
        reference_dir=reference_dir, n_features=30
    )["subject_session"].copy()
    datasets = {"combined_decoders": combined}
    datasets.update({
        side: decoder.loc[decoder["decoder_side"].eq(side)].copy()
        for side in ("right", "left")
    })
    anova_rows, model_rows, fitted_models = [], [], {}
    for outcome_name, values in datasets.items():
        data = values.loc[values["assessment_order"].between(1, 5)].copy()
        expected_rows = len(EXPECTED_SUBJECTS) * 5
        if len(data) != expected_rows or data["subject_id"].nunique() != len(EXPECTED_SUBJECTS):
            raise ValueError(f"{outcome_name}: expected {expected_rows} complete participant-day rows.")
        if not data.groupby("subject_id")["assessment_order"].nunique().eq(5).all():
            raise ValueError(f"{outcome_name}: every participant must contribute all five days.")
        if outcome_name != "combined_decoders" and not data["n_features"].eq(30).all():
            raise ValueError(f"{outcome_name}: requires 30 frozen features per participant/decoder/day.")
        if not (data["mean_r2"] > 0).all():
            raise ValueError(f"{outcome_name}: requires strictly positive top-30 participant means.")
        if data.groupby("group")["subject_id"].nunique().to_dict() != {"bci": 16, "control": 16}:
            raise ValueError(f"{outcome_name}: requires 16 participants per group.")
        data["intervention_day"] = data["assessment_order"].astype(int).astype("category")
        data["group_bci"] = data["group"].eq("bci").astype(int)
        data["log_mean_r2"] = np.log(data["mean_r2"])
        formula = "log_mean_r2 ~ C(intervention_day) * group_bci"
        try:
            fitted = smf.mixedlm(formula, data=data, groups=data["subject_id"]).fit(
                reml=False, method="lbfgs", maxiter=200, disp=False
            )
        except Exception as exc:
            raise RuntimeError(f"{outcome_name}: mixed ANOVA model failed: {exc}") from exc
        if not fitted.converged:
            raise RuntimeError(f"{outcome_name}: mixed ANOVA model did not converge.")
        wald_table = fitted.wald_test_terms(skip_single=False).table.copy()
        term_map = {
            "C(intervention_day)": "Intervention day",
            "group_bci": "Group",
            "C(intervention_day):group_bci": "Group × intervention day",
        }
        missing_terms = set(term_map).difference(wald_table.index)
        if missing_terms:
            raise RuntimeError(f"{outcome_name}: ANOVA table lacks terms {sorted(missing_terms)}.")
        for raw_term, display_term in term_map.items():
            row = wald_table.loc[raw_term]
            statistic = float(np.asarray(row["statistic"]).squeeze())
            anova_rows.append({
                "outcome": outcome_name,
                "effect": display_term,
                "chi2": statistic,
                "df": int(row["df_constraint"]),
                "p_value": float(np.asarray(row["pvalue"]).squeeze()),
            })
        model_rows.append({
            "outcome": outcome_name,
            "formula": formula,
            "n_observations": int(len(data)),
            "n_participants": int(data["subject_id"].nunique()),
            "random_effect": "participant random intercept",
            "converged": bool(fitted.converged),
            "log_likelihood": float(fitted.llf),
        })
        fitted_models[outcome_name] = fitted
    anova_table = pd.DataFrame(anova_rows)
    decoder_interaction = anova_table.loc[
        anova_table["outcome"].isin(["right", "left"])
        & anova_table["effect"].eq("Group × intervention day")
    ].copy()
    corrected = multipletests(decoder_interaction["p_value"], method="holm")[1]
    anova_table["p_value_holm_decoder_specific"] = np.nan
    anova_table.loc[decoder_interaction.index, "p_value_holm_decoder_specific"] = corrected
    print("TOP-30 INTERVENTION-DAY MIXED ANOVAS")
    print("  Outcome: log participant mean r²; Days 1-5 decoding only.")
    print("  Type-III Wald chi-square tests from participant-random-intercept mixed models.")
    print("  Right/left Group × intervention day p-values are Holm-corrected together.")
    print(anova_table.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    return {
        "anova_table": anova_table,
        "model_info": pd.DataFrame(model_rows),
        "models": fitted_models,
    }


def run_checkpointed_session5_reference_build(
    subject_ids=None,
    project_root=PROJECT_ROOT,
    output_dir=None,
    n_pruning_iterations=20,
    random_seed=20260812,
    resume=True,
):
    """Build and checkpoint frozen Session-5 references across a cohort.

    The validated Session-5 manifest is built once. Each participant is then
    fitted and persisted independently, so a later model failure cannot erase
    completed references. ``resume=True`` validates and skips complete saved
    references; incomplete or inconsistent artifacts are rebuilt. The retained
    build log is a per-decoder audit of selection status and pruning outcomes.
    """
    subjects = [str(subject).lower().strip() for subject in (
        EXPECTED_SUBJECTS if subject_ids is None else subject_ids
    )]
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("subject_ids must contain one or more unique participant IDs.")
    unknown = sorted(set(subjects).difference(EXPECTED_SUBJECTS))
    if unknown:
        raise ValueError(f"Unknown participant ID(s): {unknown}.")
    output_dir = Path(output_dir) if output_dir is not None else REPO_ROOT / "analyses" / "session5_feature_references"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "session5_reference_build_log.parquet"
    session5_model_inputs = build_session5_training_model_manifest(
        subject_ids=subjects,
        project_root=project_root,
    )
    manifest_subjects = set(session5_model_inputs["manifest"]["subject_id"].astype(str).str.lower())
    if manifest_subjects != set(subjects):
        raise RuntimeError("Validated Session-5 manifest does not match the requested cohort.")

    def _read_complete_subject_summary(subject_id):
        summary_path = output_dir / "session5_reference_summary.parquet"
        clean_path = output_dir / "session5_reference_clean_trials.parquet"
        top30_path = output_dir / "session5_reference_top30.parquet"
        settings_path = output_dir / "session5_reference_settings.json"
        transform_dir = output_dir / "transforms"
        required_paths = [summary_path, clean_path, top30_path, settings_path]
        if not all(path.exists() for path in required_paths):
            return None
        try:
            summary = pd.read_parquet(summary_path)
            clean_trials = pd.read_parquet(clean_path)
            top30 = pd.read_parquet(top30_path)
            subject_summary = summary.loc[
                summary["subject_id"].astype(str).str.lower().eq(subject_id)
            ].copy()
            if subject_summary["decoder_side"].value_counts().to_dict() != {"right": 1, "left": 1}:
                return None
            for decoder_side in ("right", "left"):
                side_top30 = top30.loc[
                    (top30["subject_id"].astype(str).str.lower() == subject_id)
                    & (top30["decoder_side"] == decoder_side)
                ]
                side_clean = clean_trials.loc[
                    (clean_trials["subject_id"].astype(str).str.lower() == subject_id)
                    & (clean_trials["decoder_side"] == decoder_side)
                ]
                if len(side_top30) != 30 or side_clean.empty:
                    return None
                transform_path = transform_dir / f"{subject_id}_{decoder_side}_frozen_reference.npz"
                if not transform_path.exists():
                    return None
                load_session5_frozen_feature_reference(subject_id, decoder_side, output_dir=output_dir)
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            return None
        return subject_summary

    def _checkpoint_log(new_rows):
        new_log = pd.DataFrame(new_rows)
        if log_path.exists():
            existing_log = pd.read_parquet(log_path)
            existing_log = existing_log.loc[
                ~existing_log["subject_id"].astype(str).str.lower().isin(
                    new_log["subject_id"].astype(str).str.lower()
                )
            ]
            combined_log = pd.concat([existing_log, new_log], ignore_index=True)
        else:
            combined_log = new_log
        if combined_log.duplicated(["subject_id", "decoder_side"]).any():
            raise RuntimeError("Session-5 reference build log has duplicate subject/decoder rows.")
        temporary_path = log_path.with_name(f".{log_path.stem}.tmp.parquet")
        combined_log.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, log_path)
        return combined_log

    def _rows_from_summary(summary, status, error_message=None):
        rows = summary.copy()
        rows["build_status"] = status
        rows["error_type"] = None
        rows["error_message"] = error_message
        rows["attempted_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        return rows.to_dict("records")

    print(f"\nSESSION-5 FROZEN REFERENCE COHORT BUILD: {len(subjects)} participant(s)")
    all_log = None
    for subject_index, subject_id in enumerate(subjects, start=1):
        print(f"\n[{subject_index}/{len(subjects)}] {subject_id}")
        existing_summary = _read_complete_subject_summary(subject_id) if resume else None
        if existing_summary is not None:
            all_log = _checkpoint_log(_rows_from_summary(existing_summary, "skipped_existing"))
            print(f"{subject_id}: existing complete reference verified; skipped.")
            continue
        try:
            reference_result = build_session5_top30_feature_references_for_subject(
                subject_id,
                session5_model_inputs,
                n_pruning_iterations=n_pruning_iterations,
                random_seed=random_seed,
            )
            save_session5_top30_feature_references(reference_result, output_dir=output_dir)
            all_log = _checkpoint_log(_rows_from_summary(reference_result["summary"], "completed"))
        except Exception as exc:
            failure_row = {
                "subject_id": subject_id,
                "group": get_subject_group(subject_id),
                "decoder_side": "all",
                "build_status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "attempted_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            all_log = _checkpoint_log([failure_row])
            print(f"WARNING: {subject_id} reference build failed; checkpointed and continuing. {exc}")
    if all_log is None:
        raise RuntimeError("No Session-5 reference build records were generated.")
    counts = all_log["build_status"].value_counts().to_dict()
    print(f"\nSession-5 reference cohort build finished. Status rows: {counts}")
    print(f"Build log: {log_path}")
    return {
        "model_inputs": session5_model_inputs,
        "build_log": all_log.sort_values(["subject_id", "decoder_side"]).reset_index(drop=True),
        "build_log_path": log_path,
        "output_dir": output_dir,
    }
