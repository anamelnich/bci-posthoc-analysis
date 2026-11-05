"""Historical unfiltered Pd sensitivity analyses; current filtered Pd remains primary."""

from .eeg import (
    _cohens_d_independent,
    _cohens_dz,
    _epoch_channel_time_trials,
    _extract_training_stimulus_events_from_status,
    _get_training_run_gdf_files_for_session,
    _infer_display_scale,
    _mean_ci,
    _plot_channel_average_epoch,
    _plot_channel_average_epoch_comparison,
    _plot_eeg_segment,
    _plot_raw_filtered_psd,
    _prepare_plot_environment,
    _resolve_gdf_path,
    _validate_paired_training_files,
    _welch_df,
    _welch_mean_difference_ci,
    baseline_correct_epochs,
    compute_pd_positive_auc,
    find_run_gdf_file,
    load_and_plot_eeg_run_segment,
    load_filter_and_plot_eeg_run_segment,
    load_filter_epoch_and_plot_training_stimulus_epochs,
    load_filter_epoch_baseline_correct_and_plot_training,
    load_filter_epoch_baseline_correct_training_run,
    plot_group_pd_pre_post,
    run_pd_positive_auc_mixed_anova,
    run_pd_positive_auc_planned_contrasts,
    select_analysis_eeg_channels,
    summarize_group_pd_pre_post,
    summarize_pd_positive_auc,
 )


from pathlib import Path


import os


import numpy as np


from .analysis import load_training_analysis_file


from .config import PROJECT_ROOT


from .config import EXPECTED_SUBJECTS, get_subject_group


from .triggers import (
    FIXATION_CODE,
    FS,
    RESPONSE_CODE,
    STIMULUS_CODES,
    TRAINING_TRIALS,
    load_training_trigger_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


FIGURES_DIR = REPO_ROOT / "figures" / "historical_pd"


EXPECTED_TOTAL_CHANNELS = 67


EXPECTED_EEG_CHANNELS = 64


EXPECTED_EOG_CHANNELS = 2


DEFAULT_EXCLUDED_NON_EEG_CHANNELS = ("M1", "M2", "EOG", "sens7", "sens8")


DEFAULT_BANDPASS_L_FREQ = 0.1


DEFAULT_BANDPASS_H_FREQ = 20.0


DEFAULT_EPOCH_TMIN = -0.5


DEFAULT_EPOCH_TMAX = 1.0


DEFAULT_BASELINE_TMIN = -0.2


DEFAULT_BASELINE_TMAX = 0.0


DEFAULT_PD_LEFT_CHANNEL = "PO7"


DEFAULT_PD_RIGHT_CHANNEL = "PO8"


EXPECTED_TRAINING_RUNS_BY_SESSION = {1: 8, 5: 4}


TRAINING_CONDITION_ORDER = (
    "distractor_left",
    "distractor_right",
    "no_distractor",
)


TRAINING_CONDITION_EEG_AVERAGES_FILENAME = "training_session_condition_eeg_averages.csv"


DEFAULT_PD_PLOT_TIME_WINDOW = (-0.2, 0.7)


DEFAULT_PD_DECODER_TIME_WINDOW = (0.2, 0.5)


POSTER_PD_FIGURE_SIZE_CM = (12.46, 13.01)


POSTER_PD_SESSION_COLORS = {
    "bci": {1: "#F4A261", 5: "#C65D28"},
    "control": {1: "#3182BD", 5: "#08306B"},
}


DEFAULT_PD_AUC_OUTPUT_FILENAME = "training_pd_positive_auc_0p2_0p5.csv"


DEFAULT_PD_NEGATIVE_AUC_OUTPUT_FILENAME = "training_pd_negative_auc_0p5_0p9.csv"


DEFAULT_PD_MEAN_AMPLITUDE_OUTPUT_FILENAME = "training_pd_mean_amplitude_0p5_0p9.csv"


DEFAULT_PD_MEAN_ABSOLUTE_AMPLITUDE_OUTPUT_FILENAME = (
    "training_pd_mean_absolute_amplitude_0p2_0p5.csv"
)


def resolve_training_condition_eeg_averages_csv(
    csv_path=None,
    project_root=PROJECT_ROOT,
    repo_root=REPO_ROOT,
):
    """Resolve the condition-average EEG CSV without recomputing raw data.

    An explicit ``csv_path`` takes precedence. Otherwise, precomputed study
    outputs in ``PROJECT_ROOT/analyses`` are preferred, with the repository's
    ``analyses`` directory retained as a portable fallback.
    """
    if csv_path is not None:
        resolved_path = Path(csv_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(
                "Explicit training condition-average EEG CSV was not found: "
                f"{resolved_path}"
            )
        print(f"Using explicit training condition-average EEG CSV: {resolved_path}")
        return resolved_path

    candidates = (
        Path(project_root) / "analyses" / TRAINING_CONDITION_EEG_AVERAGES_FILENAME,
        Path(repo_root) / "analyses" / TRAINING_CONDITION_EEG_AVERAGES_FILENAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            print(f"Using precomputed training condition-average EEG CSV: {candidate}")
            return candidate

    checked_paths = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Training condition-average EEG CSV was not found. Checked:\n  - "
        f"{checked_paths}\n"
        "Generate it from raw training GDF files only after confirming that no "
        "precomputed copy is available."
    )


def _plot_pd_grand_average(time, pd_grand_average, n_trials, figsize=(7, 4)):
    """Plot grand-average Pd across distractor trials."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, pd_grand_average, color="black", linewidth=1.7)
    ax.axvline(0, color="0.25", linestyle="--", linewidth=0.9)
    ax.axhline(0, color="0.8", linewidth=0.7)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Pd amplitude (native GDF units)")
    ax.set_title(f"Grand-average Pd across {n_trials} distractor trials")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def compute_pd_difference_wave(
    baseline_corrected_epochs,
    eeg_labels,
    stimulus_events,
    time=None,
    left_channel=DEFAULT_PD_LEFT_CHANNEL,
    right_channel=DEFAULT_PD_RIGHT_CHANNEL,
):
    """Compute single-trial and grand-average Pd from PO7/PO8 difference waves.

    For right-distractor trials (trigger 32), Pd is ``PO7 - PO8``. For
    left-distractor trials (trigger 44), Pd is ``PO8 - PO7``. No-distractor
    trials (trigger 8) are ignored.
    """
    epochs = np.asarray(baseline_corrected_epochs)
    if epochs.ndim != 3:
        raise ValueError(
            "baseline_corrected_epochs must be channels x samples x trials, "
            f"got shape {epochs.shape}."
        )
    if not np.isfinite(epochs).all():
        bad_count = int(np.size(epochs) - np.isfinite(epochs).sum())
        raise ValueError(f"Baseline-corrected epochs contain {bad_count} non-finite values.")
    if left_channel not in eeg_labels:
        raise ValueError(f"Required Pd channel {left_channel!r} not found in EEG labels.")
    if right_channel not in eeg_labels:
        raise ValueError(f"Required Pd channel {right_channel!r} not found in EEG labels.")

    stimulus_events = np.asarray(stimulus_events)
    if stimulus_events.ndim != 2 or stimulus_events.shape[1] < 3:
        raise ValueError(
            "stimulus_events must be an events array with at least 3 columns "
            f"(sample, previous, code), got shape {stimulus_events.shape}."
        )
    if stimulus_events.shape[0] != epochs.shape[2]:
        raise ValueError(
            "Number of stimulus events must match epoch trial dimension; got "
            f"{stimulus_events.shape[0]} events and {epochs.shape[2]} trials."
        )
    if time is not None and len(time) != epochs.shape[1]:
        raise ValueError(
            f"len(time) must match epoch sample count; got {len(time)} and "
            f"{epochs.shape[1]}."
        )

    left_idx = eeg_labels.index(left_channel)
    right_idx = eeg_labels.index(right_channel)
    codes = stimulus_events[:, 2].astype(int)
    right_mask = codes == 32
    left_mask = codes == 44
    no_distractor_mask = codes == 8
    unexpected_codes = sorted(set(codes) - {8, 32, 44})
    if unexpected_codes:
        raise ValueError(f"Unexpected stimulus trigger codes for Pd: {unexpected_codes}.")
    if not np.any(right_mask):
        raise ValueError("No right-distractor trigger 32 trials found for Pd.")
    if not np.any(left_mask):
        raise ValueError("No left-distractor trigger 44 trials found for Pd.")

    pd_trials = []
    pd_trial_codes = []
    pd_trial_indices = []
    for trial_idx, code in enumerate(codes):
        if code == 32:
            pd_waveform = epochs[left_idx, :, trial_idx] - epochs[right_idx, :, trial_idx]
        elif code == 44:
            pd_waveform = epochs[right_idx, :, trial_idx] - epochs[left_idx, :, trial_idx]
        else:
            continue
        pd_trials.append(pd_waveform)
        pd_trial_codes.append(code)
        pd_trial_indices.append(trial_idx)

    pd_trials = np.stack(pd_trials, axis=1)
    pd_grand_average = np.nanmean(pd_trials, axis=1)
    code_counts = {
        "no_distractor_ignored": int(np.sum(no_distractor_mask)),
        "distractor_right_32": int(np.sum(right_mask)),
        "distractor_left_44": int(np.sum(left_mask)),
        "distractor_trials_used": int(pd_trials.shape[1]),
    }
    print("Pd difference-wave summary:")
    print(f"  Channels: {left_channel}, {right_channel}")
    print("  Right distractor (32): PO7 - PO8")
    print("  Left distractor (44): PO8 - PO7")
    print(f"  Trial counts: {code_counts}")
    print(f"  Pd trial matrix shape (samples x distractor trials): {pd_trials.shape}")
    print(f"  Pd grand-average shape (samples): {pd_grand_average.shape}")

    pd_fig = None
    pd_ax = None
    if time is not None:
        pd_fig, pd_ax = _plot_pd_grand_average(
            time=time,
            pd_grand_average=pd_grand_average,
            n_trials=pd_trials.shape[1],
        )

    return {
        "pd_trials_samples_by_trials": pd_trials,
        "pd_grand_average": pd_grand_average,
        "pd_trial_codes": np.asarray(pd_trial_codes),
        "pd_trial_indices": np.asarray(pd_trial_indices),
        "pd_code_counts": code_counts,
        "pd_left_channel": left_channel,
        "pd_right_channel": right_channel,
        "pd_fig": pd_fig,
        "pd_ax": pd_ax,
    }


def compute_run_condition_eeg_averages(
    baseline_corrected_epochs,
    stimulus_events,
    eeg_labels,
):
    """Average one run's baseline-corrected epochs by stimulus condition."""
    epochs = np.asarray(baseline_corrected_epochs)
    stimulus_events = np.asarray(stimulus_events)
    if epochs.ndim != 3:
        raise ValueError(f"Expected epochs as channels x samples x trials, got {epochs.shape}.")
    if stimulus_events.shape[0] != epochs.shape[2]:
        raise ValueError(
            f"Stimulus event count {stimulus_events.shape[0]} does not match "
            f"trial count {epochs.shape[2]}."
        )
    if len(eeg_labels) != epochs.shape[0]:
        raise ValueError(
            f"EEG label count {len(eeg_labels)} does not match channel count "
            f"{epochs.shape[0]}."
        )

    condition_averages = {}
    condition_counts = {}
    codes = stimulus_events[:, 2].astype(int)
    for code, condition in STIMULUS_CODES.items():
        mask = codes == code
        if not np.any(mask):
            raise ValueError(f"No trials found for condition {condition} / trigger {code}.")
        # Store samples x channels to match the requested 768 x 61 matrix.
        condition_averages[condition] = np.nanmean(epochs[:, :, mask], axis=2).T
        condition_counts[condition] = int(np.sum(mask))

    expected_counts = {
        "no_distractor": 30,
        "distractor_right": 15,
        "distractor_left": 15,
    }
    if condition_counts != expected_counts:
        raise ValueError(
            f"Unexpected condition trial counts. Expected {expected_counts}, "
            f"got {condition_counts}."
        )

    return condition_averages, condition_counts


def generate_session_condition_eeg_averages_csv(
    project_root=PROJECT_ROOT,
    output_path=None,
    subjects=None,
    sessions=(1, 5),
    l_freq=DEFAULT_BANDPASS_L_FREQ,
    h_freq=DEFAULT_BANDPASS_H_FREQ,
    tmin=DEFAULT_EPOCH_TMIN,
    tmax=DEFAULT_EPOCH_TMAX,
    baseline_tmin=DEFAULT_BASELINE_TMIN,
    baseline_tmax=DEFAULT_BASELINE_TMAX,
    verbose_run_details=False,
    allow_incomplete_sessions=True,
):
    """Save subject/session/condition EEG average matrices for training data.

    Each saved row is one time sample for one subject/session/condition. Channel
    columns store the session-level average matrix with shape 768 x 61.
    Run-level condition averages are computed first, then averaged across runs
    within session.
    """
    import contextlib
    import io
    import pandas as pd

    project_root = Path(project_root)
    if output_path is None:
        output_path = REPO_ROOT / "exploratory_results" / "historical_pd" / "training_session_condition_eeg_averages.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subjects = sorted(EXPECTED_SUBJECTS if subjects is None else subjects)
    all_rows = []
    manifest_rows = []
    reference_labels = None
    reference_time = None

    print(
        "Generating session-level condition EEG averages using canonical Pd "
        "preprocessing: 0.1-20 Hz zero-phase filter, stimulus epochs, "
        "-0.2 to 0 s baseline correction."
    )
    print(f"Subjects: {len(subjects)}; sessions: {sessions}")

    for subject_id in subjects:
        group = get_subject_group(subject_id)
        for session in sessions:
            gdf_files = _get_training_run_gdf_files_for_session(
                subject_id,
                session,
                project_root=project_root,
                allow_incomplete=allow_incomplete_sessions,
            )
            expected_run_count = EXPECTED_TRAINING_RUNS_BY_SESSION[session]
            print(
                f"\nProcessing {subject_id} session {session}: "
                f"{len(gdf_files)} training runs."
            )

            run_condition_averages = {condition: [] for condition in STIMULUS_CODES.values()}
            run_condition_counts = []
            run_status_counts = []
            for run_number, gdf_path in enumerate(gdf_files, start=1):
                output_context = (
                    contextlib.nullcontext()
                    if verbose_run_details
                    else contextlib.redirect_stdout(io.StringIO())
                )
                with output_context:
                    run_results = load_filter_epoch_baseline_correct_training_run(
                        gdf_path=gdf_path,
                        l_freq=l_freq,
                        h_freq=h_freq,
                        tmin=tmin,
                        tmax=tmax,
                        baseline_tmin=baseline_tmin,
                        baseline_tmax=baseline_tmax,
                    )
                if reference_labels is None:
                    reference_labels = list(run_results["eeg_labels"])
                    reference_time = np.asarray(run_results["time"])
                else:
                    if list(run_results["eeg_labels"]) != reference_labels:
                        raise ValueError(
                            f"Channel labels differ for {gdf_path}."
                        )
                    if not np.allclose(run_results["time"], reference_time):
                        raise ValueError(f"Epoch time vector differs for {gdf_path}.")

                condition_averages, condition_counts = compute_run_condition_eeg_averages(
                    baseline_corrected_epochs=run_results["baseline_corrected_epochs"],
                    stimulus_events=run_results["stimulus_events"],
                    eeg_labels=run_results["eeg_labels"],
                )
                status_codes, status_counts = np.unique(
                    run_results["task_events"][:, 2],
                    return_counts=True,
                )
                status_count_summary = dict(zip(status_codes.tolist(), status_counts.tolist()))
                run_status_counts.append(status_count_summary)
                response_count = status_count_summary.get(RESPONSE_CODE, 0)
                if response_count != TRAINING_TRIALS:
                    print(
                        "  WARNING: "
                        f"{gdf_path.name} has {response_count}/{TRAINING_TRIALS} "
                        "response Status events; stimulus anchors are complete."
                    )
                for condition, average_matrix in condition_averages.items():
                    run_condition_averages[condition].append(average_matrix)
                run_condition_counts.append(condition_counts)
                print(
                    f"  Run {run_number}/{expected_run_count}: "
                    f"{gdf_path.name}, counts={condition_counts}"
                )

            for condition in TRAINING_CONDITION_ORDER:
                run_stack = np.stack(run_condition_averages[condition], axis=0)
                session_matrix = np.nanmean(run_stack, axis=0)
                if session_matrix.shape != (len(reference_time), len(reference_labels)):
                    raise ValueError(
                        f"Unexpected session matrix shape for {subject_id} "
                        f"session {session} {condition}: {session_matrix.shape}."
                    )

                for sample_index, time_sec in enumerate(reference_time):
                    row = {
                        "subject_id": subject_id,
                        "group": group,
                        "session_id": session,
                        "condition": condition,
                        "sample_index": sample_index,
                        "time_sec": float(time_sec),
                        "n_runs": len(gdf_files),
                        "expected_n_runs": expected_run_count,
                    }
                    row.update({
                        channel: float(session_matrix[sample_index, channel_index])
                        for channel_index, channel in enumerate(reference_labels)
                    })
                    all_rows.append(row)

                manifest_rows.append({
                    "subject_id": subject_id,
                    "group": group,
                    "session_id": session,
                    "condition": condition,
                    "n_runs": len(gdf_files),
                    "expected_n_runs": expected_run_count,
                    "matrix_shape": f"{session_matrix.shape[0]}x{session_matrix.shape[1]}",
                    "run_condition_counts": str(run_condition_counts),
                    "run_status_counts": str(run_status_counts),
                })

    df = pd.DataFrame(all_rows)
    manifest = pd.DataFrame(manifest_rows)
    df.to_csv(output_path, index=False)
    manifest_path = output_path.with_name(output_path.stem + "_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    expected_rows = len(subjects) * len(sessions) * len(TRAINING_CONDITION_ORDER) * len(reference_time)
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, wrote {len(df)} rows.")
    print("\nSession condition EEG average CSV saved.")
    print(f"  Output: {output_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"  DataFrame shape: {df.shape}")
    print(
        f"  Each subject/session/condition matrix: "
        f"{len(reference_time)} samples x {len(reference_labels)} channels."
    )

    return {
        "output_path": output_path,
        "manifest_path": manifest_path,
        "dataframe": df,
        "manifest": manifest,
        "eeg_labels": reference_labels,
        "time": reference_time,
    }


def compute_subject_session_pd_from_condition_averages(
    csv_path=None,
    left_channel=DEFAULT_PD_LEFT_CHANNEL,
    right_channel=DEFAULT_PD_RIGHT_CHANNEL,
    expected_samples=768,
    distractor_condition=None,
):
    """Compute subject/session Pd waveforms from condition-average EEG CSV.

    The input CSV contains session-level condition averages rather than
    single-trial data. Because training has balanced left/right distractor
    counts within each run, the default subject/session Pd is the equal average
    of right-distractor ``PO7 - PO8`` and left-distractor ``PO8 - PO7`` waves.
    Set ``distractor_condition`` to ``"distractor_right"`` or
    ``"distractor_left"`` to retain that side's condition-specific waveform.
    No-distractor rows are always validated but not used for Pd.
    """
    import pandas as pd

    csv_path = resolve_training_condition_eeg_averages_csv(csv_path=csv_path)

    df = pd.read_csv(csv_path)
    required_columns = {
        "subject_id",
        "group",
        "session_id",
        "condition",
        "sample_index",
        "time_sec",
        "n_runs",
        "expected_n_runs",
        left_channel,
        right_channel,
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Training condition EEG average CSV is missing required columns: "
            f"{missing_columns}"
        )
    if df.empty:
        raise ValueError(f"Training condition EEG average CSV is empty: {csv_path}")
    if not np.isfinite(df[[left_channel, right_channel, "time_sec"]].to_numpy()).all():
        raise ValueError(
            f"Columns {left_channel}, {right_channel}, and time_sec must be finite."
        )

    observed_conditions = set(df["condition"].unique())
    expected_conditions = set(TRAINING_CONDITION_ORDER)
    if observed_conditions != expected_conditions:
        raise ValueError(
            f"Expected conditions {sorted(expected_conditions)}, "
            f"found {sorted(observed_conditions)}."
        )
    observed_sessions = sorted(df["session_id"].unique().tolist())
    if observed_sessions != [1, 5]:
        raise ValueError(f"Expected sessions [1, 5], found {observed_sessions}.")
    valid_distractor_conditions = {"distractor_right", "distractor_left"}
    if distractor_condition is not None and distractor_condition not in valid_distractor_conditions:
        raise ValueError(
            "distractor_condition must be None, 'distractor_right', or "
            f"'distractor_left'; got {distractor_condition!r}."
        )

    index_columns = ["subject_id", "group", "session_id", "condition"]
    sample_counts = df.groupby(index_columns, observed=False)["sample_index"].nunique()
    bad_sample_counts = sample_counts[sample_counts != expected_samples]
    if not bad_sample_counts.empty:
        raise ValueError(
            f"Expected {expected_samples} samples per subject/session/condition. "
            f"Bad cells: {bad_sample_counts.to_dict()}"
        )

    duplicated = df.duplicated(
        ["subject_id", "group", "session_id", "condition", "sample_index"]
    )
    if duplicated.any():
        raise ValueError(
            "Duplicate rows found for subject/group/session/condition/sample_index; "
            f"first duplicates at rows {df.index[duplicated].tolist()[:10]}."
        )

    expected_sample_index = np.arange(expected_samples)
    for cell, cell_df in df.groupby(index_columns, observed=False):
        sample_index = cell_df.sort_values("sample_index")["sample_index"].to_numpy()
        if not np.array_equal(sample_index, expected_sample_index):
            raise ValueError(
                "Sample indices must be consecutive 0.."
                f"{expected_samples - 1}; bad cell {cell}."
            )

    right = df[df["condition"] == "distractor_right"].copy()
    left = df[df["condition"] == "distractor_left"].copy()
    right["pd_amplitude_uv"] = right[left_channel] - right[right_channel]
    left["pd_amplitude_uv"] = left[right_channel] - left[left_channel]
    if distractor_condition == "distractor_right":
        distractor_pd = right
        expected_condition_count = 1
    elif distractor_condition == "distractor_left":
        distractor_pd = left
        expected_condition_count = 1
    else:
        distractor_pd = pd.concat([right, left], ignore_index=True)
        expected_condition_count = 2

    subject_session_pd = (
        distractor_pd
        .groupby(["subject_id", "group", "session_id", "sample_index"], observed=False)
        .agg(
            time_sec=("time_sec", "mean"),
            pd_amplitude_uv=("pd_amplitude_uv", "mean"),
            n_distractor_conditions=("condition", "nunique"),
            n_runs=("n_runs", "first"),
            expected_n_runs=("expected_n_runs", "first"),
        )
        .reset_index()
    )
    bad_condition_counts = subject_session_pd[
        subject_session_pd["n_distractor_conditions"] != expected_condition_count
    ]
    if not bad_condition_counts.empty:
        raise ValueError(
            "Unexpected number of distractor conditions in a subject/session/sample. "
            f"Expected {expected_condition_count}; bad rows: "
            f"{bad_condition_counts.head().to_dict('records')}"
        )

    subject_session_counts = (
        subject_session_pd
        .groupby(["subject_id", "group", "session_id"], observed=False)
        ["sample_index"]
        .nunique()
    )
    bad_pd_counts = subject_session_counts[subject_session_counts != expected_samples]
    if not bad_pd_counts.empty:
        raise ValueError(
            f"Expected {expected_samples} Pd samples per subject/session. "
            f"Bad cells: {bad_pd_counts.to_dict()}"
        )

    summary = (
        subject_session_pd
        .groupby(["group", "session_id"], observed=False)["subject_id"]
        .nunique()
        .reset_index(name="n_subjects")
        .sort_values(["group", "session_id"])
    )
    print("Subject/session Pd waveform validation:")
    print(f"  Source CSV: {csv_path}")
    print(f"  Input rows x columns: {df.shape}")
    print(
        "  Pd convention: right distractor = "
        f"{left_channel}-{right_channel}; left distractor = "
        f"{right_channel}-{left_channel}; no-distractor ignored."
    )
    print(
        "  Included distractor condition: "
        f"{distractor_condition if distractor_condition is not None else 'balanced left/right average'}"
    )
    print(f"  Subject-session Pd rows: {subject_session_pd.shape[0]}")
    print("  Subjects contributing by group/session:")
    print(summary.to_string(index=False))

    return subject_session_pd


def plot_combined_group_pd_pre_post(
    group_pd_summary,
    time_window=DEFAULT_PD_PLOT_TIME_WINDOW,
    decoder_window=DEFAULT_PD_DECODER_TIME_WINDOW,
    y_limits=(-1.5, 1.5),
    colors=None,
    group_session_rt_sec=None,
    condition_label=None,
    figsize=(5.5, 3.2),
):
    """Plot BCI and control pre/post Pd waveforms in a publication-ready layout.

    The two panels share axes to make group differences directly interpretable.
    Lines show group means, translucent bands show SEM, and the shaded region
    marks the prespecified 0.2--0.5 s positive-Pd AUC window. When supplied,
    group/session mean reaction times are drawn as color-matched dotted lines.
    """
    _prepare_plot_environment()
    import matplotlib.pyplot as plt

    required_columns = {
        "group",
        "session_id",
        "sample_index",
        "time_sec",
        "mean_pd_uv",
        "sem_pd_uv",
        "n_subjects",
    }
    missing_columns = sorted(required_columns - set(group_pd_summary.columns))
    if missing_columns:
        raise ValueError(f"group_pd_summary is missing columns: {missing_columns}")
    if time_window[0] >= time_window[1]:
        raise ValueError(f"time_window must be increasing, got {time_window}.")
    if decoder_window[0] >= decoder_window[1]:
        raise ValueError(f"decoder_window must be increasing, got {decoder_window}.")
    if y_limits[0] >= y_limits[1]:
        raise ValueError(f"y_limits must be increasing, got {y_limits}.")

    if colors is None:
        colors = {1: "#3B6FB6", 5: "#D97941"}
    missing_colors = sorted({1, 5} - set(colors))
    if missing_colors:
        raise ValueError(f"colors is missing session IDs: {missing_colors}")

    group_specs = (
        ("bci", "BCI", "a"),
        ("control", "Mental rehearsal", "b"),
    )
    session_labels = {1: "Pre-training", 5: "Post-training"}
    group_session_rt_sec = {} if group_session_rt_sec is None else group_session_rt_sec
    rt_values = []
    for group, _, _ in group_specs:
        session_rt = group_session_rt_sec.get(group, {})
        for session_id in (1, 5):
            rt_sec = session_rt.get(session_id)
            if rt_sec is None:
                continue
            if not np.isfinite(rt_sec) or rt_sec <= 0:
                raise ValueError(
                    f"Reaction time for {group}, session {session_id} must be "
                    f"a positive finite value; got {rt_sec!r}."
                )
            rt_values.append(float(rt_sec))
    plot_time_window = (
        time_window[0],
        max(time_window[1], max(rt_values) + 0.04) if rt_values else time_window[1],
    )
    plot_data = group_pd_summary[
        (group_pd_summary["time_sec"] >= plot_time_window[0])
        & (group_pd_summary["time_sec"] <= plot_time_window[1])
    ].copy()
    if plot_data.empty:
        raise ValueError(f"No Pd samples found in time window {plot_time_window}.")

    rc_params = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.25,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        handles = []
        subject_counts = {}

        for ax, (group, group_label, panel_label) in zip(axes, group_specs):
            group_data = plot_data[plot_data["group"].str.lower() == group]
            if group_data.empty:
                raise ValueError(f"No Pd summary rows found for group {group!r}.")

            n_by_session = (
                group_data.groupby("session_id", observed=False)["n_subjects"]
                .max()
                .to_dict()
            )
            if set(n_by_session) != {1, 5} or n_by_session[1] != n_by_session[5]:
                raise ValueError(
                    f"Expected matched Session 1/5 sample sizes for {group}; "
                    f"found {n_by_session}."
                )
            subject_counts[group] = int(n_by_session[1])

            ax.axvspan(
                decoder_window[0],
                decoder_window[1],
                color="#D9D9D9",
                alpha=0.7,
                linewidth=0,
                zorder=0,
            )
            ax.axvline(0, color="#333333", linestyle=(0, (3, 2)), linewidth=0.8)
            ax.axhline(0, color="#9A9A9A", linewidth=0.6)

            for session_id in (1, 5):
                session_data = group_data[
                    group_data["session_id"] == session_id
                ].sort_values("time_sec")
                if session_data.empty:
                    raise ValueError(
                        f"No Pd rows for group {group}, session {session_id}."
                    )
                time = session_data["time_sec"].to_numpy()
                mean = session_data["mean_pd_uv"].to_numpy()
                sem = session_data["sem_pd_uv"].to_numpy()
                line = ax.plot(
                    time,
                    mean,
                    color=colors[session_id],
                    linewidth=1.5,
                    label=session_labels[session_id],
                    zorder=3,
                )[0]
                ax.fill_between(
                    time,
                    mean - sem,
                    mean + sem,
                    color=colors[session_id],
                    alpha=0.18,
                    linewidth=0,
                    zorder=2,
                )
                if not handles:
                    handles.append(line)
                elif session_id == 5 and len(handles) == 1:
                    handles.append(line)

                rt_sec = group_session_rt_sec.get(group, {}).get(session_id)
                if rt_sec is not None:
                    ax.axvline(
                        rt_sec,
                        color=colors[session_id],
                        linestyle=(0, (1.2, 1.8)),
                        linewidth=1.0,
                        alpha=0.95,
                        zorder=4,
                    )

            ax.set_xlim(plot_time_window)
            ax.set_ylim(y_limits)
            ax.set_xticks(np.arange(time_window[0], plot_time_window[1] + 1e-9, 0.2))
            ax.set_title(f"{group_label} (n = {subject_counts[group]})", pad=5)
            ax.text(
                -0.16,
                1.03,
                panel_label,
                transform=ax.transAxes,
                fontsize=9,
                fontweight="bold",
                va="bottom",
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_ylabel("Pd amplitude (µV)")
        from matplotlib.lines import Line2D
        legend_handles = handles.copy()
        legend_labels = [session_labels[1], session_labels[5]]
        if rt_values:
            legend_handles.append(
                Line2D([0], [0], color="#555555", linestyle=(0, (1.2, 1.8)), linewidth=1.0)
            )
            legend_labels.append("Mean RT")
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(legend_handles),
            handlelength=2.2,
            columnspacing=1.5,
        )
        if condition_label is not None:
            fig.text(0.55, 0.85, str(condition_label), ha="center", va="center", fontsize=7)
        fig.text(0.55, 0.05, "Time from stimulus onset (s)", ha="center", va="center")
        panel_top = 0.75 if condition_label is not None else 0.81
        fig.subplots_adjust(left=0.1, right=0.995, bottom=0.19, top=panel_top, wspace=0.14)

    print(
        "Plotted combined Pd pre/post figure: "
        f"time={plot_time_window[0]:g} to {plot_time_window[1]:g} s; "
        f"AUC window={decoder_window[0]:g} to {decoder_window[1]:g} s; "
        f"subjects/group={subject_counts}; "
        f"RT lines={'included' if rt_values else 'not included'}."
    )
    return fig, axes


def load_and_plot_combined_group_pd_pre_post_from_condition_csv(
    csv_path=None,
    output_dir=None,
    save_figure=True,
    figure_formats=("pdf", "png"),
    include_distractor_rt=True,
    rt_csv_path=None,
    distractor_condition=None,
    figure_stem="training_pd_pre_post_combined",
    y_limits=(-1.5, 1.5),
    condition_label=None,
):
    """Load Pd data and save a two-panel BCI/control pre/post waveform figure.

    By default, this adds group/session mean RT lines calculated from cleaned
    correct distractor trials in the consolidated training CSV.
    """
    csv_path = resolve_training_condition_eeg_averages_csv(csv_path=csv_path)
    subject_session_pd = compute_subject_session_pd_from_condition_averages(
        csv_path=csv_path,
        distractor_condition=distractor_condition,
    )
    group_pd_summary = summarize_group_pd_pre_post(subject_session_pd)

    rt_summary = None
    group_session_rt_sec = None
    if include_distractor_rt:
        from .behavioral import summarize_distractor_trial_reaction_time_from_training_csv

        rt_results = summarize_distractor_trial_reaction_time_from_training_csv(
            csv_path=rt_csv_path,
            save_figure=False,
        )
        rt_summary = rt_results["cell_summary"].copy()
        required_rt_columns = {"group", "session_id", "mean_rt_ms"}
        missing_rt_columns = sorted(required_rt_columns - set(rt_summary.columns))
        if missing_rt_columns:
            raise ValueError(
                "Distractor RT summary is missing required columns: "
                f"{missing_rt_columns}"
            )
        rt_group_map = {"experimental": "bci", "control": "control"}
        group_session_rt_sec = {"bci": {}, "control": {}}
        for row in rt_summary.itertuples(index=False):
            mapped_group = rt_group_map.get(row.group)
            if mapped_group is None:
                raise ValueError(f"Unexpected RT group label: {row.group!r}")
            session_id = int(row.session_id)
            if session_id not in {1, 5}:
                raise ValueError(f"Unexpected RT session ID: {session_id}")
            if session_id in group_session_rt_sec[mapped_group]:
                raise ValueError(
                    f"Duplicate RT value for {mapped_group}, session {session_id}."
                )
            group_session_rt_sec[mapped_group][session_id] = float(row.mean_rt_ms) / 1000.0
        missing_rt_cells = [
            (group, session_id)
            for group in ("bci", "control")
            for session_id in (1, 5)
            if session_id not in group_session_rt_sec[group]
        ]
        if missing_rt_cells:
            raise ValueError(f"Missing distractor RT group/session cells: {missing_rt_cells}")
        print(f"Using distractor-trial mean RTs (s): {group_session_rt_sec}")

    fig, axes = plot_combined_group_pd_pre_post(
        group_pd_summary,
        group_session_rt_sec=group_session_rt_sec,
        y_limits=y_limits,
        condition_label=condition_label,
    )

    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    saved_paths = []
    if save_figure:
        if not figure_stem or not str(figure_stem).strip():
            raise ValueError("figure_stem must be a non-empty filename stem.")
        output_dir.mkdir(parents=True, exist_ok=True)
        for figure_format in figure_formats:
            path = output_dir / f"{figure_stem}.{figure_format}"
            fig.savefig(path, dpi=600, bbox_inches="tight")
            saved_paths.append(path)
            print(f"Saved publication Pd figure: {path}")

    return {
        "subject_session_pd": subject_session_pd,
        "group_pd_summary": group_pd_summary,
        "distractor_rt_summary": rt_summary,
        "group_session_rt_sec": group_session_rt_sec,
        "distractor_condition": distractor_condition,
        "figure": fig,
        "axes": axes,
        "saved_paths": saved_paths,
    }


def _determine_shared_pd_y_limits(
    group_pd_summaries,
    time_window=(-0.2, 1.0),
    padding_fraction=0.05,
    rounding_increment=0.5,
):
    """Determine symmetric y-limits shared across comparable Pd summaries."""
    if not group_pd_summaries:
        raise ValueError("group_pd_summaries must contain at least one summary table.")
    if time_window[0] >= time_window[1]:
        raise ValueError(f"time_window must be increasing, got {time_window}.")
    if padding_fraction < 0:
        raise ValueError(f"padding_fraction must be non-negative, got {padding_fraction}.")
    if rounding_increment <= 0:
        raise ValueError(
            f"rounding_increment must be positive, got {rounding_increment}."
        )

    values = []
    for summary in group_pd_summaries:
        required_columns = {"time_sec", "mean_pd_uv", "sem_pd_uv"}
        missing_columns = sorted(required_columns - set(summary.columns))
        if missing_columns:
            raise ValueError(
                "Pd summary is missing columns required for shared y-limits: "
                f"{missing_columns}"
            )
        subset = summary[
            (summary["time_sec"] >= time_window[0])
            & (summary["time_sec"] <= time_window[1])
        ]
        values.extend((subset["mean_pd_uv"] - subset["sem_pd_uv"]).tolist())
        values.extend((subset["mean_pd_uv"] + subset["sem_pd_uv"]).tolist())

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite Pd mean ± SEM values available for shared y-limits.")
    maximum_absolute_value = float(np.max(np.abs(values)))
    limit = np.ceil(maximum_absolute_value * (1 + padding_fraction) / rounding_increment)
    limit *= rounding_increment
    if limit <= 0:
        limit = rounding_increment
    y_limits = (-float(limit), float(limit))
    print(
        "Shared Pd y-limits across comparison: "
        f"{y_limits[0]:g} to {y_limits[1]:g} µV "
        f"(time={time_window[0]:g} to {time_window[1]:g} s)."
    )
    return y_limits


def load_and_plot_lateralized_pd_pre_post_from_condition_csv(
    csv_path=None,
    output_dir=None,
    save_figures=True,
    figure_formats=("pdf", "png"),
    include_distractor_rt=True,
    rt_csv_path=None,
):
    """Save matched right- and left-distractor Pd pre/post waveform figures.

    The right-distractor figure uses ``PO7 - PO8`` and the left-distractor
    figure uses ``PO8 - PO7``. Both retain the combined figure's shared BCI/
    control layout, fixed y-axis limits, and optional distractor-trial RT lines.
    """
    csv_path = resolve_training_condition_eeg_averages_csv(csv_path=csv_path)
    condition_specs = (
        ("distractor_right", "right", "PO7 − PO8", "training_pd_right_distractor_pre_post"),
        ("distractor_left", "left", "PO8 − PO7", "training_pd_left_distractor_pre_post"),
    )
    comparison_summaries = []
    for condition, _, _, _ in condition_specs:
        side_pd = compute_subject_session_pd_from_condition_averages(
            csv_path=csv_path,
            distractor_condition=condition,
        )
        comparison_summaries.append(summarize_group_pd_pre_post(side_pd))
    shared_y_limits = _determine_shared_pd_y_limits(comparison_summaries)

    results = {}
    for condition, side_label, convention, figure_stem in condition_specs:
        print(f"\nGenerating {side_label}-distractor Pd figure ({convention}).")
        result = load_and_plot_combined_group_pd_pre_post_from_condition_csv(
            csv_path=csv_path,
            output_dir=output_dir,
            save_figure=save_figures,
            figure_formats=figure_formats,
            include_distractor_rt=include_distractor_rt,
            rt_csv_path=rt_csv_path,
            distractor_condition=condition,
            figure_stem=figure_stem,
            y_limits=shared_y_limits,
            condition_label=f"{side_label.capitalize()} distractor: {convention}",
        )
        result["side_label"] = side_label
        result["convention"] = convention
        result["shared_y_limits"] = shared_y_limits
        results[side_label] = result
    return results


def load_and_plot_group_pd_pre_post_from_condition_csv(
    csv_path=None,
    output_dir=None,
    save_figures=True,
    figure_formats=("pdf", "png"),
    figure_size_cm=POSTER_PD_FIGURE_SIZE_CM,
):
    """Load condition-average EEG CSV and make poster-format Pd figures.

    ``figure_size_cm`` is (width, height) in centimetres and is applied to
    both group figures so their poster placement is directly comparable.
    """
    csv_path = resolve_training_condition_eeg_averages_csv(csv_path=csv_path)
    subject_session_pd = compute_subject_session_pd_from_condition_averages(csv_path=csv_path)
    group_pd_summary = summarize_group_pd_pre_post(subject_session_pd)
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    if save_figures:
        output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}
    axes = {}
    saved_paths = []
    for group, label, stem in (
        ("bci", "BCI", "training_pd_bci_pre_post"),
        ("control", "Mental rehearsal", "training_pd_control_pre_post"),
    ):
        fig, ax = plot_group_pd_pre_post(
            group_pd_summary=group_pd_summary,
            group=group,
            group_label=label,
            figure_size_cm=figure_size_cm,
        )
        figures[group] = fig
        axes[group] = ax
        if save_figures:
            for figure_format in figure_formats:
                path = output_dir / f"{stem}.{figure_format}"
                # Preserve the requested poster canvas dimensions in the final
                # PDF/PNG rather than trimming its physical page bounds.
                fig.savefig(path, dpi=300)
                saved_paths.append(path)
                print(f"Saved figure: {path}")

    return {
        "subject_session_pd": subject_session_pd,
        "group_pd_summary": group_pd_summary,
        "figures": figures,
        "axes": axes,
        "saved_paths": saved_paths,
    }


def load_compute_and_run_pd_positive_auc_anova(
    csv_path=None,
    output_path=None,
    time_window=DEFAULT_PD_DECODER_TIME_WINDOW,
    save_auc=True,
    distractor_condition=None,
):
    """Load condition-average CSV, compute positive Pd AUC, and run ANOVA.

    Set ``distractor_condition`` to ``"distractor_right"`` or
    ``"distractor_left"`` for a side-specific positive-Pd AUC analysis.
    """
    subject_session_pd = compute_subject_session_pd_from_condition_averages(
        csv_path=csv_path,
        distractor_condition=distractor_condition,
    )
    auc_df = compute_pd_positive_auc(subject_session_pd, time_window=time_window)
    cell_summary = summarize_pd_positive_auc(auc_df)
    anova_results = run_pd_positive_auc_mixed_anova(auc_df)
    planned_contrasts = run_pd_positive_auc_planned_contrasts(auc_df)

    if save_auc:
        if output_path is None:
            condition_suffixes = {
                None: "",
                "distractor_right": "_right_distractor",
                "distractor_left": "_left_distractor",
            }
            if distractor_condition not in condition_suffixes:
                raise ValueError(
                    "distractor_condition must be None, 'distractor_right', or "
                    f"'distractor_left'; got {distractor_condition!r}."
                )
            default_stem = Path(DEFAULT_PD_AUC_OUTPUT_FILENAME).stem
            output_filename = (
                f"{default_stem}{condition_suffixes[distractor_condition]}.csv"
            )
            output_path = REPO_ROOT / "exploratory_results" / "historical_pd" / output_filename
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        auc_df.to_csv(output_path, index=False)
        print(f"\nSaved subject/session Pd positive AUC table: {output_path}")
    else:
        output_path = None

    return {
        "subject_session_pd": subject_session_pd,
        "pd_positive_auc": auc_df,
        "cell_summary": cell_summary,
        "anova_table": anova_results["anova_table"],
        "planned_contrasts": planned_contrasts,
        "anova_diagnostics": anova_results["diagnostics"],
        "anova_input_data": anova_results["input_data"],
        "distractor_condition": distractor_condition,
        "output_path": output_path,
    }


def compute_pd_negative_auc(
    subject_session_pd,
    time_window=(0.5, 0.9),
):
    """Compute signed negative Pd area for each subject/session.

    Negative AUC is the trapezoidal integral of ``min(Pd, 0)`` over time, so
    values are zero or negative and units are microvolts x seconds.
    """
    import pandas as pd

    required_columns = {
        "subject_id",
        "group",
        "session_id",
        "sample_index",
        "time_sec",
        "pd_amplitude_uv",
    }
    missing_columns = sorted(required_columns - set(subject_session_pd.columns))
    if missing_columns:
        raise ValueError(f"subject_session_pd is missing columns: {missing_columns}")
    if time_window[0] >= time_window[1]:
        raise ValueError(f"time_window must be increasing, got {time_window}.")

    data = subject_session_pd.copy()
    data = data[
        (data["time_sec"] >= time_window[0])
        & (data["time_sec"] <= time_window[1])
    ].copy()
    if data.empty:
        raise ValueError(f"No Pd samples found in time window {time_window}.")
    if not np.isfinite(data[["time_sec", "pd_amplitude_uv"]].to_numpy()).all():
        raise ValueError("Pd negative AUC input contains non-finite time or amplitude values.")

    rows = []
    for (subject_id, group, session_id), cell in data.groupby(
        ["subject_id", "group", "session_id"],
        observed=False,
    ):
        cell = cell.sort_values("time_sec")
        time = cell["time_sec"].to_numpy()
        pd_wave = cell["pd_amplitude_uv"].to_numpy()
        if len(time) < 2:
            raise ValueError(
                f"Need at least two samples for AUC; got {len(time)} for "
                f"{subject_id} session {session_id}."
            )
        if np.any(np.diff(time) <= 0):
            raise ValueError(
                f"Time samples must be strictly increasing for "
                f"{subject_id} session {session_id}."
            )
        negative_wave = np.minimum(pd_wave, 0.0)
        negative_auc = float(np.trapezoid(negative_wave, time))
        rows.append({
            "subject_id": subject_id,
            "group": group,
            "session_id": int(session_id),
            "time_window_start_sec": float(time_window[0]),
            "time_window_end_sec": float(time_window[1]),
            "n_samples": int(len(time)),
            "negative_auc_uv_sec": negative_auc,
            "mean_pd_uv_in_window": float(np.mean(pd_wave)),
            "mean_negative_pd_uv_in_window": float(np.mean(negative_wave)),
        })

    auc_df = pd.DataFrame(rows).sort_values(["group", "subject_id", "session_id"])
    expected_sessions = [1, 5]
    observed_sessions = sorted(auc_df["session_id"].unique().tolist())
    if observed_sessions != expected_sessions:
        raise ValueError(
            f"Expected Pd negative AUC sessions {expected_sessions}, found {observed_sessions}."
        )
    duplicate_cells = (
        auc_df.groupby(["subject_id", "session_id"], observed=False)
        .size()
        .reset_index(name="n_rows")
    )
    duplicate_cells = duplicate_cells[duplicate_cells["n_rows"] != 1]
    if not duplicate_cells.empty:
        raise ValueError(
            "Expected exactly one Pd negative AUC row per subject/session. Problem cells:\n"
            f"{duplicate_cells.to_string(index=False)}"
        )

    counts = (
        auc_df.groupby(["group", "session_id"], observed=False)["subject_id"]
        .nunique()
        .reset_index(name="n_subjects")
        .sort_values(["group", "session_id"])
    )
    sample_counts = (
        auc_df.groupby(["group", "session_id"], observed=False)["n_samples"]
        .unique()
        .reset_index(name="n_samples")
        .sort_values(["group", "session_id"])
    )
    print("Pd negative AUC summary:")
    print(
        f"  Window: {time_window[0]:g} to {time_window[1]:g} s; "
        "signed integral of min(Pd, 0)."
    )
    print("  Subjects by group/session:")
    print(counts.to_string(index=False))
    print("  Samples per group/session:")
    print(sample_counts.to_string(index=False))
    return auc_df


def _rename_negative_auc_for_positive_auc_helpers(auc_df):
    renamed = auc_df.rename(
        columns={"negative_auc_uv_sec": "positive_auc_uv_sec"}
    ).copy()
    return renamed


def summarize_pd_negative_auc(auc_df):
    """Summarize signed negative Pd AUC by group/session."""
    required_columns = {"subject_id", "group", "session_id", "negative_auc_uv_sec"}
    missing_columns = sorted(required_columns - set(auc_df.columns))
    if missing_columns:
        raise ValueError(f"auc_df is missing columns: {missing_columns}")

    summary = (
        auc_df.groupby(["group", "session_id"], observed=False)["negative_auc_uv_sec"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "mean_negative_auc_uv_sec",
            "std": "sd_negative_auc_uv_sec",
            "count": "n_subjects",
        })
    )
    summary["sem_negative_auc_uv_sec"] = (
        summary["sd_negative_auc_uv_sec"] / np.sqrt(summary["n_subjects"])
    )
    print("\nPd negative AUC cell summary:")
    print(summary.to_string(index=False))
    return summary


def run_pd_negative_auc_mixed_anova(auc_df):
    """Run Group x Session mixed ANOVA on signed negative Pd AUC."""
    import io
    from contextlib import redirect_stdout

    helper_input = _rename_negative_auc_for_positive_auc_helpers(auc_df)
    with redirect_stdout(io.StringIO()):
        results = run_pd_positive_auc_mixed_anova(helper_input)
    table = results["anova_table"].copy()
    diagnostics = results["diagnostics"].copy()
    input_data = results["input_data"].rename(
        columns={"positive_auc_uv_sec": "negative_auc_uv_sec"}
    )
    diagnostics["grand_mean_negative_auc_uv_sec"] = diagnostics.pop(
        "grand_mean_positive_auc_uv_sec"
    )
    print("=" * 80)
    print("PD NEGATIVE AUC MIXED-DESIGN ANOVA")
    print("=" * 80)
    print("Design: Group (between: BCI vs mental rehearsal) x Session (within: 1 vs 5)")
    print(f"Subjects by group: {diagnostics['subjects_by_group']}")
    print("\nMixed-design ANOVA table:")
    print(table.to_string(index=False))
    return {
        "anova_table": table,
        "diagnostics": diagnostics,
        "input_data": input_data,
    }


def run_pd_negative_auc_planned_contrasts(auc_df):
    """Run planned contrasts for signed negative Pd AUC."""
    import io
    from contextlib import redirect_stdout

    helper_input = _rename_negative_auc_for_positive_auc_helpers(auc_df)
    with redirect_stdout(io.StringIO()):
        contrasts = run_pd_positive_auc_planned_contrasts(helper_input).copy()
    contrasts = contrasts.rename(
        columns={
            "estimate_uv_sec": "estimate_negative_auc_uv_sec",
            "ci95_low_uv_sec": "ci95_low_negative_auc_uv_sec",
            "ci95_high_uv_sec": "ci95_high_negative_auc_uv_sec",
        }
    )
    print("\nPlanned Pd negative AUC contrasts:")
    print(contrasts.to_string(index=False))
    return contrasts


def load_compute_and_run_pd_negative_auc_anova(
    csv_path=None,
    output_path=None,
    time_window=(0.5, 0.9),
    save_auc=True,
):
    """Load condition-average CSV, compute signed negative Pd AUC, and run ANOVA."""
    subject_session_pd = compute_subject_session_pd_from_condition_averages(csv_path=csv_path)
    auc_df = compute_pd_negative_auc(subject_session_pd, time_window=time_window)
    cell_summary = summarize_pd_negative_auc(auc_df)
    anova_results = run_pd_negative_auc_mixed_anova(auc_df)
    planned_contrasts = run_pd_negative_auc_planned_contrasts(auc_df)

    if save_auc:
        if output_path is None:
            output_path = REPO_ROOT / "exploratory_results" / "historical_pd" / DEFAULT_PD_NEGATIVE_AUC_OUTPUT_FILENAME
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        auc_df.to_csv(output_path, index=False)
        print(f"\nSaved subject/session Pd negative AUC table: {output_path}")
    else:
        output_path = None

    return {
        "subject_session_pd": subject_session_pd,
        "pd_negative_auc": auc_df,
        "cell_summary": cell_summary,
        "anova_table": anova_results["anova_table"],
        "planned_contrasts": planned_contrasts,
        "anova_diagnostics": anova_results["diagnostics"],
        "anova_input_data": anova_results["input_data"],
        "output_path": output_path,
    }


def compute_pd_mean_amplitude(
    subject_session_pd,
    time_window=(0.5, 0.9),
):
    """Compute mean Pd amplitude for each subject/session in a time window."""
    import pandas as pd

    required_columns = {
        "subject_id",
        "group",
        "session_id",
        "sample_index",
        "time_sec",
        "pd_amplitude_uv",
    }
    missing_columns = sorted(required_columns - set(subject_session_pd.columns))
    if missing_columns:
        raise ValueError(f"subject_session_pd is missing columns: {missing_columns}")
    if time_window[0] >= time_window[1]:
        raise ValueError(f"time_window must be increasing, got {time_window}.")

    data = subject_session_pd.copy()
    data = data[
        (data["time_sec"] >= time_window[0])
        & (data["time_sec"] <= time_window[1])
    ].copy()
    if data.empty:
        raise ValueError(f"No Pd samples found in time window {time_window}.")
    if not np.isfinite(data[["time_sec", "pd_amplitude_uv"]].to_numpy()).all():
        raise ValueError("Pd mean-amplitude input contains non-finite time or amplitude values.")

    rows = []
    for (subject_id, group, session_id), cell in data.groupby(
        ["subject_id", "group", "session_id"],
        observed=False,
    ):
        cell = cell.sort_values("time_sec")
        time = cell["time_sec"].to_numpy()
        pd_wave = cell["pd_amplitude_uv"].to_numpy()
        if len(time) < 2:
            raise ValueError(
                f"Need at least two samples for mean amplitude; got {len(time)} for "
                f"{subject_id} session {session_id}."
            )
        if np.any(np.diff(time) <= 0):
            raise ValueError(
                f"Time samples must be strictly increasing for "
                f"{subject_id} session {session_id}."
            )
        rows.append({
            "subject_id": subject_id,
            "group": group,
            "session_id": int(session_id),
            "time_window_start_sec": float(time_window[0]),
            "time_window_end_sec": float(time_window[1]),
            "n_samples": int(len(time)),
            "mean_pd_amplitude_uv": float(np.mean(pd_wave)),
            "sd_pd_amplitude_uv_within_window": float(np.std(pd_wave, ddof=1)),
        })

    mean_df = pd.DataFrame(rows).sort_values(["group", "subject_id", "session_id"])
    expected_sessions = [1, 5]
    observed_sessions = sorted(mean_df["session_id"].unique().tolist())
    if observed_sessions != expected_sessions:
        raise ValueError(
            f"Expected Pd mean-amplitude sessions {expected_sessions}, found {observed_sessions}."
        )
    duplicate_cells = (
        mean_df.groupby(["subject_id", "session_id"], observed=False)
        .size()
        .reset_index(name="n_rows")
    )
    duplicate_cells = duplicate_cells[duplicate_cells["n_rows"] != 1]
    if not duplicate_cells.empty:
        raise ValueError(
            "Expected exactly one Pd mean-amplitude row per subject/session. Problem cells:\n"
            f"{duplicate_cells.to_string(index=False)}"
        )

    counts = (
        mean_df.groupby(["group", "session_id"], observed=False)["subject_id"]
        .nunique()
        .reset_index(name="n_subjects")
        .sort_values(["group", "session_id"])
    )
    sample_counts = (
        mean_df.groupby(["group", "session_id"], observed=False)["n_samples"]
        .unique()
        .reset_index(name="n_samples")
        .sort_values(["group", "session_id"])
    )
    print("Pd mean-amplitude summary:")
    print(f"  Window: {time_window[0]:g} to {time_window[1]:g} s.")
    print("  Subjects by group/session:")
    print(counts.to_string(index=False))
    print("  Samples per group/session:")
    print(sample_counts.to_string(index=False))
    return mean_df


def summarize_pd_mean_amplitude(mean_df):
    """Summarize mean Pd amplitude by group/session."""
    required_columns = {"subject_id", "group", "session_id", "mean_pd_amplitude_uv"}
    missing_columns = sorted(required_columns - set(mean_df.columns))
    if missing_columns:
        raise ValueError(f"mean_df is missing columns: {missing_columns}")

    summary = (
        mean_df.groupby(["group", "session_id"], observed=False)["mean_pd_amplitude_uv"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "mean_pd_amplitude_uv",
            "std": "sd_pd_amplitude_uv",
            "count": "n_subjects",
        })
    )
    summary["sem_pd_amplitude_uv"] = (
        summary["sd_pd_amplitude_uv"] / np.sqrt(summary["n_subjects"])
    )
    print("\nPd mean-amplitude cell summary:")
    print(summary.to_string(index=False))
    return summary


def _rename_mean_amplitude_for_positive_auc_helpers(mean_df):
    return mean_df.rename(
        columns={"mean_pd_amplitude_uv": "positive_auc_uv_sec"}
    ).copy()


def run_pd_mean_amplitude_mixed_anova(mean_df):
    """Run Group x Session mixed ANOVA on mean Pd amplitude."""
    import io
    from contextlib import redirect_stdout

    helper_input = _rename_mean_amplitude_for_positive_auc_helpers(mean_df)
    with redirect_stdout(io.StringIO()):
        results = run_pd_positive_auc_mixed_anova(helper_input)
    table = results["anova_table"].copy()
    diagnostics = results["diagnostics"].copy()
    input_data = results["input_data"].rename(
        columns={"positive_auc_uv_sec": "mean_pd_amplitude_uv"}
    )
    diagnostics["grand_mean_pd_amplitude_uv"] = diagnostics.pop(
        "grand_mean_positive_auc_uv_sec"
    )
    print("=" * 80)
    print("PD MEAN-AMPLITUDE MIXED-DESIGN ANOVA")
    print("=" * 80)
    print("Design: Group (between: BCI vs mental rehearsal) x Session (within: 1 vs 5)")
    print(f"Subjects by group: {diagnostics['subjects_by_group']}")
    print("\nMixed-design ANOVA table:")
    print(table.to_string(index=False))
    return {
        "anova_table": table,
        "diagnostics": diagnostics,
        "input_data": input_data,
    }


def run_pd_mean_amplitude_planned_contrasts(mean_df):
    """Run planned contrasts for mean Pd amplitude."""
    import io
    from contextlib import redirect_stdout

    helper_input = _rename_mean_amplitude_for_positive_auc_helpers(mean_df)
    with redirect_stdout(io.StringIO()):
        contrasts = run_pd_positive_auc_planned_contrasts(helper_input).copy()
    contrasts = contrasts.rename(
        columns={
            "estimate_uv_sec": "estimate_uv",
            "ci95_low_uv_sec": "ci95_low_uv",
            "ci95_high_uv_sec": "ci95_high_uv",
        }
    )
    print("\nPlanned Pd mean-amplitude contrasts:")
    print(contrasts.to_string(index=False))
    return contrasts


def load_compute_and_run_pd_mean_amplitude_anova(
    csv_path=None,
    output_path=None,
    time_window=(0.5, 0.9),
    save_table=True,
):
    """Load condition-average CSV, compute mean Pd amplitude, and run ANOVA."""
    subject_session_pd = compute_subject_session_pd_from_condition_averages(csv_path=csv_path)
    mean_df = compute_pd_mean_amplitude(subject_session_pd, time_window=time_window)
    cell_summary = summarize_pd_mean_amplitude(mean_df)
    anova_results = run_pd_mean_amplitude_mixed_anova(mean_df)
    planned_contrasts = run_pd_mean_amplitude_planned_contrasts(mean_df)

    if save_table:
        if output_path is None:
            output_path = REPO_ROOT / "exploratory_results" / "historical_pd" / DEFAULT_PD_MEAN_AMPLITUDE_OUTPUT_FILENAME
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mean_df.to_csv(output_path, index=False)
        print(f"\nSaved subject/session Pd mean-amplitude table: {output_path}")
    else:
        output_path = None

    return {
        "subject_session_pd": subject_session_pd,
        "pd_mean_amplitude": mean_df,
        "cell_summary": cell_summary,
        "anova_table": anova_results["anova_table"],
        "planned_contrasts": planned_contrasts,
        "anova_diagnostics": anova_results["diagnostics"],
        "anova_input_data": anova_results["input_data"],
        "output_path": output_path,
    }


def compute_pd_mean_absolute_amplitude(
    subject_session_pd,
    time_window=DEFAULT_PD_DECODER_TIME_WINDOW,
):
    """Compute the mean absolute combined Pd amplitude per subject/session.

    This measure is the arithmetic mean of ``abs(Pd(t))`` across samples in
    the requested window. It captures waveform magnitude irrespective of
    polarity and is therefore distinct from signed mean amplitude or AUC.
    """
    import pandas as pd

    required_columns = {
        "subject_id", "group", "session_id", "sample_index", "time_sec",
        "pd_amplitude_uv",
    }
    missing_columns = sorted(required_columns - set(subject_session_pd.columns))
    if missing_columns:
        raise ValueError(f"subject_session_pd is missing columns: {missing_columns}")
    if time_window[0] >= time_window[1]:
        raise ValueError(f"time_window must be increasing, got {time_window}.")

    data = subject_session_pd.loc[
        (subject_session_pd["time_sec"] >= time_window[0])
        & (subject_session_pd["time_sec"] <= time_window[1])
    ].copy()
    if data.empty:
        raise ValueError(f"No Pd samples found in time window {time_window}.")
    if not np.isfinite(data[["time_sec", "pd_amplitude_uv"]].to_numpy()).all():
        raise ValueError("Pd mean-absolute-amplitude input contains non-finite values.")

    rows = []
    for (subject_id, group, session_id), cell in data.groupby(
        ["subject_id", "group", "session_id"], observed=False
    ):
        cell = cell.sort_values("time_sec")
        time = cell["time_sec"].to_numpy()
        waveform = cell["pd_amplitude_uv"].to_numpy()
        if len(time) < 2 or np.any(np.diff(time) <= 0):
            raise ValueError(
                "Pd samples must contain at least two strictly increasing time points "
                f"for {subject_id} session {session_id}."
            )
        rows.append({
            "subject_id": subject_id,
            "group": group,
            "session_id": int(session_id),
            "time_window_start_sec": float(time_window[0]),
            "time_window_end_sec": float(time_window[1]),
            "n_samples": int(len(time)),
            "mean_absolute_pd_amplitude_uv": float(np.mean(np.abs(waveform))),
        })

    mean_abs_df = pd.DataFrame(rows).sort_values(["group", "subject_id", "session_id"])
    observed_sessions = sorted(mean_abs_df["session_id"].unique().tolist())
    if observed_sessions != [1, 5]:
        raise ValueError(f"Expected sessions [1, 5], found {observed_sessions}.")
    cells = mean_abs_df.groupby(["subject_id", "session_id"], observed=False).size()
    if not (cells == 1).all():
        raise ValueError("Expected exactly one mean-absolute Pd row per subject/session.")
    counts = (mean_abs_df.groupby(["group", "session_id"], observed=False)["subject_id"]
              .nunique().reset_index(name="n_subjects"))
    sample_counts = (mean_abs_df.groupby(["group", "session_id"], observed=False)["n_samples"]
                     .unique().reset_index(name="n_samples"))
    print("Pd mean-absolute-amplitude summary:")
    print(f"  Window: {time_window[0]:g} to {time_window[1]:g} s; measure: mean(abs(Pd)).")
    print("  Subjects by group/session:")
    print(counts.to_string(index=False))
    print("  Samples per group/session:")
    print(sample_counts.to_string(index=False))
    return mean_abs_df


def summarize_pd_mean_absolute_amplitude(mean_abs_df):
    """Summarize mean absolute Pd amplitude by group and session."""
    value_col = "mean_absolute_pd_amplitude_uv"
    required_columns = {"subject_id", "group", "session_id", value_col}
    missing_columns = sorted(required_columns - set(mean_abs_df.columns))
    if missing_columns:
        raise ValueError(f"mean_abs_df is missing columns: {missing_columns}")
    summary = (mean_abs_df.groupby(["group", "session_id"], observed=False)[value_col]
               .agg(["mean", "std", "count"]).reset_index()
               .rename(columns={"mean": value_col, "std": "sd_mean_absolute_pd_amplitude_uv",
                                "count": "n_subjects"}))
    summary["sem_mean_absolute_pd_amplitude_uv"] = (
        summary["sd_mean_absolute_pd_amplitude_uv"] / np.sqrt(summary["n_subjects"])
    )
    print("\nPd mean-absolute-amplitude cell summary:")
    print(summary.to_string(index=False))
    return summary


def _rename_mean_absolute_amplitude_for_positive_auc_helpers(mean_abs_df):
    return mean_abs_df.rename(
        columns={"mean_absolute_pd_amplitude_uv": "positive_auc_uv_sec"}
    ).copy()


def run_pd_mean_absolute_amplitude_mixed_anova(mean_abs_df):
    """Run Group x Session mixed ANOVA on mean absolute Pd amplitude."""
    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        results = run_pd_positive_auc_mixed_anova(
            _rename_mean_absolute_amplitude_for_positive_auc_helpers(mean_abs_df)
        )
    diagnostics = results["diagnostics"].copy()
    diagnostics["grand_mean_absolute_pd_amplitude_uv"] = diagnostics.pop(
        "grand_mean_positive_auc_uv_sec"
    )
    input_data = results["input_data"].rename(
        columns={"positive_auc_uv_sec": "mean_absolute_pd_amplitude_uv"}
    )
    print("=" * 80)
    print("PD MEAN-ABSOLUTE-AMPLITUDE MIXED-DESIGN ANOVA")
    print("=" * 80)
    print("Design: Group (between: BCI vs mental rehearsal) x Session (within: 1 vs 5)")
    print(f"Subjects by group: {diagnostics['subjects_by_group']}")
    print("\nMixed-design ANOVA table:")
    print(results["anova_table"].to_string(index=False))
    return {"anova_table": results["anova_table"].copy(), "diagnostics": diagnostics,
            "input_data": input_data}


def run_pd_mean_absolute_amplitude_planned_contrasts(mean_abs_df):
    """Run planned contrasts for mean absolute Pd amplitude."""
    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        contrasts = run_pd_positive_auc_planned_contrasts(
            _rename_mean_absolute_amplitude_for_positive_auc_helpers(mean_abs_df)
        ).copy()
    contrasts = contrasts.rename(columns={
        "estimate_uv_sec": "estimate_uv", "ci95_low_uv_sec": "ci95_low_uv",
        "ci95_high_uv_sec": "ci95_high_uv",
    })
    print("\nPlanned Pd mean-absolute-amplitude contrasts:")
    print(contrasts.to_string(index=False))
    return contrasts


def load_compute_and_run_pd_mean_absolute_amplitude_anova(
    csv_path=None,
    output_path=None,
    time_window=DEFAULT_PD_DECODER_TIME_WINDOW,
    save_table=True,
):
    """Compute combined Pd mean absolute amplitude and run Group x Session tests."""
    subject_session_pd = compute_subject_session_pd_from_condition_averages(csv_path=csv_path)
    mean_abs_df = compute_pd_mean_absolute_amplitude(subject_session_pd, time_window=time_window)
    cell_summary = summarize_pd_mean_absolute_amplitude(mean_abs_df)
    anova_results = run_pd_mean_absolute_amplitude_mixed_anova(mean_abs_df)
    planned_contrasts = run_pd_mean_absolute_amplitude_planned_contrasts(mean_abs_df)
    if save_table:
        if output_path is None:
            output_path = REPO_ROOT / "exploratory_results" / "historical_pd" / DEFAULT_PD_MEAN_ABSOLUTE_AMPLITUDE_OUTPUT_FILENAME
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mean_abs_df.to_csv(output_path, index=False)
        print(f"\nSaved subject/session Pd mean-absolute-amplitude table: {output_path}")
    else:
        output_path = None
    return {
        "subject_session_pd": subject_session_pd,
        "pd_mean_absolute_amplitude": mean_abs_df,
        "cell_summary": cell_summary,
        "anova_table": anova_results["anova_table"],
        "planned_contrasts": planned_contrasts,
        "anova_diagnostics": anova_results["diagnostics"],
        "anova_input_data": anova_results["input_data"],
        "output_path": output_path,
    }


def _cluster_mass_statistics(t_values, threshold):
    """Return contiguous supra-threshold cluster indices and absolute t masses."""
    supra_threshold = np.abs(t_values) >= threshold
    padded = np.pad(supra_threshold.astype(int), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [
        (int(start), int(stop), float(np.abs(t_values[start:stop]).sum()))
        for start, stop in zip(starts, stops)
    ]


def run_pd_pre_post_cluster_permutation(
    subject_session_pd,
    time_window=(0.0, 1.2),
    n_permutations=10000,
    cluster_forming_alpha=0.05,
    cluster_alpha=0.05,
    random_seed=20260811,
):
    """Run paired, two-sided cluster permutation tests on pre/post Pd waves.

    The test is performed independently within the BCI and mental-rehearsal
    groups using sign-flips of each participant's post-minus-pre waveform.
    Cluster mass is the sum of absolute one-sample t statistics. The returned
    cluster p-values control family-wise error over the tested time samples.
    """
    import pandas as pd
    from scipy import stats

    required_columns = {"subject_id", "group", "session_id", "sample_index", "time_sec", "pd_amplitude_uv"}
    missing_columns = sorted(required_columns - set(subject_session_pd.columns))
    if missing_columns:
        raise ValueError(f"subject_session_pd is missing columns: {missing_columns}")
    if time_window[0] >= time_window[1]:
        raise ValueError(f"time_window must be increasing, got {time_window}.")
    if not isinstance(n_permutations, int) or n_permutations < 1000:
        raise ValueError("n_permutations must be an integer of at least 1000.")
    if not 0 < cluster_forming_alpha < 1 or not 0 < cluster_alpha < 1:
        raise ValueError("cluster alpha values must be strictly between zero and one.")

    data = subject_session_pd.loc[
        (subject_session_pd["time_sec"] >= time_window[0])
        & (subject_session_pd["time_sec"] <= time_window[1])
    ].copy()
    if data.empty:
        raise ValueError(f"No Pd data found in requested time window {time_window}.")
    available_max = float(subject_session_pd["time_sec"].max())
    effective_window = (float(data["time_sec"].min()), float(data["time_sec"].max()))
    if time_window[1] > available_max:
        print(
            f"Requested end time {time_window[1]:g} s exceeds available Pd data "
            f"({available_max:g} s); testing through {effective_window[1]:g} s."
        )
    sample_time = data[["sample_index", "time_sec"]].drop_duplicates().sort_values("sample_index")
    if sample_time["sample_index"].duplicated().any() or not np.all(np.diff(sample_time["time_sec"]) > 0):
        raise ValueError("Pd sample indices must map one-to-one to strictly increasing times.")

    rng = np.random.default_rng(random_seed)
    cluster_rows, point_rows, group_rows = [], [], []
    for group in ("bci", "control"):
        group_data = data[data["group"].str.lower() == group].copy()
        if group_data.empty:
            raise ValueError(f"No Pd data found for group {group!r}.")
        pivot = group_data.pivot(index="subject_id", columns=["session_id", "sample_index"], values="pd_amplitude_uv")
        required_columns = [(session, sample) for session in (1, 5) for sample in sample_time["sample_index"]]
        missing = [column for column in required_columns if column not in pivot.columns]
        if missing or pivot[required_columns].isna().any().any():
            raise ValueError(f"Incomplete paired Pd data for {group}; missing session/sample values.")
        pre = pivot.loc[:, [(1, sample) for sample in sample_time["sample_index"]]].to_numpy()
        post = pivot.loc[:, [(5, sample) for sample in sample_time["sample_index"]]].to_numpy()
        if pre.shape != post.shape or pre.shape[0] < 2:
            raise ValueError(f"Expected matched paired waveforms for {group}; got {pre.shape}, {post.shape}.")
        differences = post - pre
        n_subjects, n_times = differences.shape
        t_values, uncorrected_p = stats.ttest_1samp(differences, 0.0, axis=0)
        if not np.isfinite(t_values).all():
            raise ValueError(f"Non-finite t statistics in {group} cluster test.")
        threshold = float(stats.t.ppf(1 - cluster_forming_alpha / 2, n_subjects - 1))
        observed_clusters = _cluster_mass_statistics(t_values, threshold)

        # Under sign-flipping, the per-timepoint sum of squares is invariant;
        # this permits a reproducible vectorized paired-permutation calculation.
        sum_squares = np.square(differences).sum(axis=0)
        null_max_masses = np.zeros(n_permutations, dtype=float)
        for permutation_idx in range(n_permutations):
            signs = rng.choice(np.array([-1.0, 1.0]), size=n_subjects)
            permuted_mean = signs @ differences / n_subjects
            variance = (sum_squares - n_subjects * np.square(permuted_mean)) / (n_subjects - 1)
            permuted_t = permuted_mean / np.sqrt(variance / n_subjects)
            permuted_clusters = _cluster_mass_statistics(permuted_t, threshold)
            null_max_masses[permutation_idx] = max((mass for _, _, mass in permuted_clusters), default=0.0)

        significant_by_sample = np.zeros(n_times, dtype=bool)
        cluster_id_by_sample = np.full(n_times, -1, dtype=int)
        for cluster_id, (start, stop, mass) in enumerate(observed_clusters, start=1):
            p_value = float((1 + np.count_nonzero(null_max_masses >= mass)) / (n_permutations + 1))
            is_significant = p_value < cluster_alpha
            if is_significant:
                significant_by_sample[start:stop] = True
            cluster_id_by_sample[start:stop] = cluster_id
            cluster_rows.append({
                "group": group, "cluster_id": cluster_id, "start_time_sec": float(sample_time.iloc[start]["time_sec"]),
                "end_time_sec": float(sample_time.iloc[stop - 1]["time_sec"]), "n_samples": stop - start,
                "cluster_mass": mass, "cluster_p_value": p_value,
                "is_significant_fwer": is_significant, "direction": "post > pre" if t_values[start:stop].mean() > 0 else "post < pre",
                "n_subjects": n_subjects, "cluster_forming_t_threshold": threshold,
            })
        for time_idx, (_, sample) in enumerate(sample_time.iterrows()):
            point_rows.append({
                "group": group, "sample_index": int(sample["sample_index"]), "time_sec": float(sample["time_sec"]),
                "t_value": float(t_values[time_idx]), "uncorrected_p_value": float(uncorrected_p[time_idx]),
                "cluster_id": None if cluster_id_by_sample[time_idx] < 0 else int(cluster_id_by_sample[time_idx]),
                "in_significant_cluster_fwer": bool(significant_by_sample[time_idx]),
            })
        group_rows.append({"group": group, "n_subjects": n_subjects, "n_times": n_times,
                           "effective_start_sec": effective_window[0], "effective_end_sec": effective_window[1],
                           "n_permutations": n_permutations, "cluster_forming_alpha": cluster_forming_alpha,
                           "cluster_alpha": cluster_alpha, "n_significant_clusters": int(sum(
                               row["group"] == group and row["is_significant_fwer"] for row in cluster_rows
                           ))})

    clusters = pd.DataFrame(cluster_rows)
    if clusters.empty:
        clusters = pd.DataFrame(columns=["group", "cluster_id", "start_time_sec", "end_time_sec", "n_samples", "cluster_mass", "cluster_p_value", "is_significant_fwer", "direction", "n_subjects", "cluster_forming_t_threshold"])
    points = pd.DataFrame(point_rows)
    group_summary = pd.DataFrame(group_rows)
    print("\nPaired pre/post Pd cluster-permutation test:")
    print(f"  Requested window: {time_window[0]:g} to {time_window[1]:g} s; effective window: {effective_window[0]:g} to {effective_window[1]:g} s.")
    print(f"  Two-sided cluster-forming alpha={cluster_forming_alpha:g}; FWER cluster alpha={cluster_alpha:g}; permutations={n_permutations}; seed={random_seed}.")
    if clusters.empty:
        print("  No supra-threshold clusters were observed.")
    else:
        print(clusters.to_string(index=False))
    return {"clusters": clusters, "pointwise_statistics": points, "group_summary": group_summary,
            "effective_time_window": effective_window, "n_permutations": n_permutations,
            "cluster_forming_alpha": cluster_forming_alpha, "cluster_alpha": cluster_alpha,
            "random_seed": random_seed}


def plot_pd_pre_post_cluster_permutation(
    group_pd_summary,
    cluster_results,
    decoder_window=DEFAULT_PD_DECODER_TIME_WINDOW,
    figsize=(5.5, 3.25),
):
    """Plot combined Pd pre/post waves and FWER-significant cluster time spans."""
    _prepare_plot_environment()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    effective_window = cluster_results["effective_time_window"]
    plot_data = group_pd_summary.loc[
        (group_pd_summary["time_sec"] >= effective_window[0]) & (group_pd_summary["time_sec"] <= effective_window[1])
    ].copy()
    if plot_data.empty:
        raise ValueError("No waveform summary data in cluster-test time window.")
    clusters = cluster_results["clusters"]
    if not {"group", "is_significant_fwer", "start_time_sec", "end_time_sec"}.issubset(clusters.columns):
        raise ValueError("cluster_results clusters table has missing required columns.")
    extent = np.abs(np.r_[
        (plot_data["mean_pd_uv"] - plot_data["sem_pd_uv"]).to_numpy(),
        (plot_data["mean_pd_uv"] + plot_data["sem_pd_uv"]).to_numpy(),
    ]).max()
    y_limit = max(1.5, float(np.ceil((extent + 0.25) * 2) / 2))
    colors = {1: "#3B6FB6", 5: "#D97941"}
    labels = {1: "Pre-training", 5: "Post-training"}
    group_specs = (("bci", "BCI", "a"), ("control", "Mental rehearsal", "b"))
    rc_params = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                 "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
                 "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                 "axes.linewidth": .5, "xtick.major.width": .5, "ytick.major.width": .5,
                 "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        for ax, (group, group_label, panel_label) in zip(axes, group_specs):
            group_data = plot_data[plot_data["group"].str.lower() == group]
            if group_data.empty:
                raise ValueError(f"No waveform summary rows for {group}.")
            n_subjects = int(group_data["n_subjects"].max())
            ax.axvspan(*decoder_window, color="#D9D9D9", alpha=.7, linewidth=0, zorder=0)
            ax.axvline(0, color="#333333", linestyle=(0, (3, 2)), linewidth=.8, zorder=1)
            ax.axhline(0, color="#9A9A9A", linewidth=.6, zorder=1)
            for session_id in (1, 5):
                wave = group_data[group_data["session_id"] == session_id].sort_values("time_sec")
                ax.plot(wave["time_sec"], wave["mean_pd_uv"], color=colors[session_id], linewidth=1.5, zorder=3)
                ax.fill_between(wave["time_sec"], wave["mean_pd_uv"] - wave["sem_pd_uv"], wave["mean_pd_uv"] + wave["sem_pd_uv"], color=colors[session_id], alpha=.18, linewidth=0, zorder=2)
            significant = clusters[(clusters["group"] == group) & clusters["is_significant_fwer"]]
            for _, cluster in significant.iterrows():
                ax.plot([cluster["start_time_sec"], cluster["end_time_sec"]], [y_limit * .92, y_limit * .92], color="#202020", linewidth=2.0, solid_capstyle="butt", clip_on=False, zorder=4)
            ax.set_xlim(effective_window)
            ax.set_ylim(-y_limit, y_limit)
            ax.set_xticks(np.arange(0, effective_window[1] + .001, .2))
            ax.set_title(f"{group_label} (n = {n_subjects})", pad=5)
            ax.text(-.16, 1.03, panel_label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel("Pd amplitude (µV)")
        fig.legend([Line2D([0], [0], color=colors[1], lw=1.5), Line2D([0], [0], color=colors[5], lw=1.5), Line2D([0], [0], color="#202020", lw=2)], [labels[1], labels[5], "Significant cluster (FWER p < .05)"], loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=3, frameon=False, handlelength=2.1, columnspacing=1.2)
        fig.text(.55, .05, "Time from stimulus onset (s)", ha="center", va="center")
        fig.subplots_adjust(left=.1, right=.995, bottom=.19, top=.81, wspace=.14)
    print(f"Plotted cluster-permutation Pd figure: shared y limits ±{y_limit:g} µV; effective time range {effective_window[0]:g} to {effective_window[1]:g} s.")
    return fig, axes


def load_run_and_plot_pd_pre_post_cluster_permutation(
    csv_path=None, output_dir=None, figure_stem="training_pd_pre_post_cluster_permutation",
    n_permutations=10000, random_seed=20260811, save_outputs=True,
):
    """Run combined-Pd paired cluster tests and save a publication-ready figure."""
    subject_session_pd = compute_subject_session_pd_from_condition_averages(csv_path=csv_path)
    group_summary = summarize_group_pd_pre_post(subject_session_pd)
    cluster_results = run_pd_pre_post_cluster_permutation(
        subject_session_pd, n_permutations=n_permutations, random_seed=random_seed
    )
    fig, axes = plot_pd_pre_post_cluster_permutation(group_summary, cluster_results)
    output_dir = FIGURES_DIR if output_dir is None else Path(output_dir)
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_paths = {}
        for extension in ("pdf", "png"):
            path = output_dir / f"{figure_stem}.{extension}"
            fig.savefig(path, dpi=400 if extension == "png" else None, bbox_inches="tight")
            figure_paths[extension] = path
        analyses_dir = REPO_ROOT / "analyses" / "eeg_pd" / "cluster_permutation"
        analyses_dir.mkdir(parents=True, exist_ok=True)
        cluster_path = analyses_dir / f"{figure_stem}_clusters.csv"
        point_path = analyses_dir / f"{figure_stem}_pointwise_statistics.csv"
        cluster_results["clusters"].to_csv(cluster_path, index=False)
        cluster_results["pointwise_statistics"].to_csv(point_path, index=False)
        print(f"Saved cluster tables: {cluster_path}; {point_path}")
    else:
        figure_paths = {}
        cluster_path = point_path = None
    return {"subject_session_pd": subject_session_pd, "group_pd_summary": group_summary,
            "cluster_results": cluster_results, "figure": fig, "axes": axes,
            "figure_paths": figure_paths, "cluster_table_path": cluster_path,
            "pointwise_table_path": point_path}


def load_compute_and_plot_training_pd(
    subject_id=None,
    session=1,
    run=1,
    gdf_path=None,
    project_root=PROJECT_ROOT,
    l_freq=DEFAULT_BANDPASS_L_FREQ,
    h_freq=DEFAULT_BANDPASS_H_FREQ,
    tmin=DEFAULT_EPOCH_TMIN,
    tmax=DEFAULT_EPOCH_TMAX,
    baseline_tmin=DEFAULT_BASELINE_TMIN,
    baseline_tmax=DEFAULT_BASELINE_TMAX,
    excluded_channels=DEFAULT_EXCLUDED_NON_EEG_CHANNELS,
    expected_fs=FS,
):
    """Run the single-run preprocessing chain and compute grand-average Pd."""
    baseline_results = load_filter_epoch_baseline_correct_and_plot_training(
        subject_id=subject_id,
        session=session,
        run=run,
        gdf_path=gdf_path,
        project_root=project_root,
        l_freq=l_freq,
        h_freq=h_freq,
        tmin=tmin,
        tmax=tmax,
        baseline_tmin=baseline_tmin,
        baseline_tmax=baseline_tmax,
        plot_channel=DEFAULT_PD_LEFT_CHANNEL,
        excluded_channels=excluded_channels,
        expected_fs=expected_fs,
    )
    pd_results = compute_pd_difference_wave(
        baseline_corrected_epochs=baseline_results["baseline_corrected_epochs"],
        eeg_labels=baseline_results["eeg_labels"],
        stimulus_events=baseline_results["stimulus_events"],
        time=baseline_results["time"],
    )
    combined = dict(baseline_results)
    combined.update(pd_results)
    return combined

