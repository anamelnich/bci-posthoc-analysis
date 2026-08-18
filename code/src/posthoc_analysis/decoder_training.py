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
        if not np.array_equal(training_classes, np.array([0, 1])):
            raise ValueError(
                f"Fold {fold_number} (held-out run {heldout_run_id}) is untrainable "
                f"after balancing: n_training={int(training_mask.sum())}, "
                f"classes={training_classes.tolist()}."
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
    manifest_rows, trial_tables, issues = [], [], []
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
    task_rank = {"session1_training_pre": 0, "decoding": 1}
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
    print(
        "Order: Session 1 training (pre) -> Session 1--5 decoding; "
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
    session5_model_inputs,
    n_pruning_iterations=20,
    random_seed=20260812,
):
    """Build frozen right/no and left/no Session-5 top-30 references.

    This reproducible orchestration function uses only the supplied validated
    four-run Session-5 training input.  For each decoder side it performs
    conventional-pair construction, iterative pruning with leave-one-run-out
    CV, then a final clean-trial xDAWN/z-score/r2 refit.  No decoding-session
    data are used in selection.
    """
    subject_id = str(subject_id).lower().strip()
    preprocessed = preprocess_session5_training_subject(subject_id, session5_model_inputs)
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
