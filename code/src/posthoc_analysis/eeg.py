"""EEG loading and quick-look plotting utilities."""

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
FIGURES_DIR = REPO_ROOT / "figures"
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


def find_run_gdf_file(
    subject_id,
    session=1,
    run=1,
    task="training",
    project_root=PROJECT_ROOT,
):
    """Find a single GDF file for a subject/session/task/run.

    Practice runs are excluded by exact task-name matching, so ``task="training"``
    resolves only folders ending in ``_training``.
    """
    project_root = Path(project_root)
    subject_id = str(subject_id).lower().strip()
    task = str(task).strip()

    if session < 1:
        raise ValueError(f"session must be 1-based, got {session}.")
    if run < 1:
        raise ValueError(f"run must be 1-based, got {run}.")

    subject_dir = project_root / subject_id
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    session_dirs = sorted(
        path
        for path in subject_dir.iterdir()
        if path.is_dir() and path.name.startswith(f"{subject_id}_")
    )
    if len(session_dirs) < session:
        raise FileNotFoundError(
            f"{subject_id} has {len(session_dirs)} session folders under {subject_dir}; "
            f"cannot select session {session}."
        )

    session_dir = session_dirs[session - 1]
    run_dirs = sorted(
        path
        for path in session_dir.iterdir()
        if path.is_dir() and path.name.endswith(f"_{task}")
    )
    if len(run_dirs) < run:
        raise FileNotFoundError(
            f"{subject_id} session {session} has {len(run_dirs)} '{task}' run folders; "
            f"cannot select run {run}."
        )

    run_dir = run_dirs[run - 1]
    gdf_files = sorted(run_dir.glob("*.gdf"))
    if len(gdf_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .gdf file in {run_dir}, found {len(gdf_files)}."
        )

    print(
        f"Resolved {subject_id} session {session} {task} run {run}: "
        f"{gdf_files[0].name}"
    )
    return gdf_files[0]


def select_analysis_eeg_channels(
    channel_labels,
    excluded_channels=DEFAULT_EXCLUDED_NON_EEG_CHANNELS,
    status_labels=("Status", "trigger"),
):
    """Return scalp EEG labels after removing mastoid, EOG, and status channels."""
    if not channel_labels:
        raise ValueError("channel_labels is empty; cannot select EEG channels.")

    excluded_lookup = {label.lower() for label in excluded_channels}
    status_lookup = {label.lower() for label in status_labels}
    selected = []
    excluded_found = []
    status_found = []

    for label in channel_labels:
        label_lower = label.lower()
        if label_lower in status_lookup:
            status_found.append(label)
        elif label_lower in excluded_lookup:
            excluded_found.append(label)
        else:
            selected.append(label)

    missing_exclusions = sorted(
        set(excluded_channels) - {label for label in excluded_found},
        key=str.lower,
    )
    if missing_exclusions:
        print(
            "WARNING: expected non-EEG channels not found in file labels: "
            f"{missing_exclusions}"
        )
    if not status_found:
        print("WARNING: no Status/trigger channel label detected.")
    if not selected:
        raise ValueError("No analysis EEG channels remain after exclusions.")

    return selected, excluded_found, status_found


def _prepare_plot_environment():
    """Configure writable cache locations before importing plotting/MNE tools."""
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path("/private/tmp") / "matplotlib-codex"),
    )
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _validate_paired_training_files(gdf_path, task):
    """Validate paired training trigger and analysis files when available."""
    paired_trigger_path = gdf_path.with_suffix(".triggers.txt")
    paired_analysis_path = gdf_path.with_suffix(".analysis.txt")
    if task != "training":
        return

    if paired_trigger_path.exists():
        trigger_df = load_training_trigger_file(paired_trigger_path)
        print(f"Validated paired trigger file: {trigger_df.shape[0]} rows.")
    else:
        print(f"WARNING: paired trigger file not found: {paired_trigger_path}")

    if paired_analysis_path.exists():
        analysis_df = load_training_analysis_file(paired_analysis_path)
        print(f"Validated paired analysis file: {analysis_df.shape[0]} rows.")
    else:
        print(f"WARNING: paired analysis file not found: {paired_analysis_path}")


def _resolve_gdf_path(subject_id, session, run, task, gdf_path, project_root):
    """Resolve a direct or subject/session/run GDF path."""
    if gdf_path is None:
        if subject_id is None:
            raise ValueError("subject_id is required when gdf_path is not supplied.")
        gdf_path = find_run_gdf_file(
            subject_id=subject_id,
            session=session,
            run=run,
            task=task,
            project_root=project_root,
        )
    else:
        gdf_path = Path(gdf_path)

    if not gdf_path.exists():
        raise FileNotFoundError(f"GDF file not found: {gdf_path}")
    return gdf_path


def _get_training_run_gdf_files_for_session(
    subject_id,
    session,
    project_root=PROJECT_ROOT,
    allow_incomplete=False,
):
    """Return non-practice training GDF files for one subject/session."""
    if session not in EXPECTED_TRAINING_RUNS_BY_SESSION:
        raise ValueError(
            f"Training EEG averaging is only expected for sessions "
            f"{sorted(EXPECTED_TRAINING_RUNS_BY_SESSION)}, got {session}."
        )
    subject_id = str(subject_id).lower().strip()
    subject_dir = Path(project_root) / subject_id
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    session_dirs = sorted(
        path
        for path in subject_dir.iterdir()
        if path.is_dir() and path.name.startswith(f"{subject_id}_")
    )
    if len(session_dirs) < session:
        raise FileNotFoundError(
            f"{subject_id} has {len(session_dirs)} session folders; "
            f"cannot select session {session}."
        )

    session_dir = session_dirs[session - 1]
    run_dirs = sorted(
        path
        for path in session_dir.iterdir()
        if path.is_dir() and path.name.endswith("_training")
    )
    expected_count = EXPECTED_TRAINING_RUNS_BY_SESSION[session]
    if len(run_dirs) != expected_count:
        message = (
            f"{subject_id} session {session} expected {expected_count} "
            f"non-practice training runs, found {len(run_dirs)} in {session_dir}."
        )
        if not allow_incomplete or len(run_dirs) == 0:
            raise ValueError(message)
        print(f"WARNING: {message} Proceeding with available runs.")

    gdf_files = []
    for run_index, run_dir in enumerate(run_dirs, start=1):
        gdf_matches = sorted(run_dir.glob("*.gdf"))
        trigger_matches = sorted(run_dir.glob("*.triggers.txt"))
        analysis_matches = sorted(run_dir.glob("*.analysis.txt"))
        if len(gdf_matches) != 1:
            raise FileNotFoundError(
                f"{subject_id} session {session} run {run_index} expected one GDF "
                f"file in {run_dir}, found {len(gdf_matches)}."
            )
        if len(trigger_matches) != 1:
            raise FileNotFoundError(
                f"{subject_id} session {session} run {run_index} expected one "
                f"trigger file in {run_dir}, found {len(trigger_matches)}."
            )
        if len(analysis_matches) != 1:
            raise FileNotFoundError(
                f"{subject_id} session {session} run {run_index} expected one "
                f"analysis file in {run_dir}, found {len(analysis_matches)}."
            )
        gdf_files.append(gdf_matches[0])

    return gdf_files


def _infer_display_scale(eeg_segment):
    """Infer whether the loaded GDF values need microvolt conversion for plots."""
    native_min = float(np.nanmin(eeg_segment))
    native_median = float(np.nanmedian(eeg_segment))
    native_max = float(np.nanmax(eeg_segment))
    native_robust_range = float(
        np.nanpercentile(eeg_segment, 95) - np.nanpercentile(eeg_segment, 5)
    )
    if native_robust_range < 1e-3:
        plot_segment = eeg_segment * 1e6
        display_unit = "microvolts"
    else:
        plot_segment = eeg_segment
        display_unit = "native GDF units"
        print(
            "NOTE: EEG values are not being multiplied by 1e6 because the "
            f"native robust range is {native_robust_range:.3g}; this file appears "
            "to already be stored in large display-scale units."
        )

    print(
        "Segment amplitude check (native units): "
        f"min={native_min:.2f}, median={native_median:.2f}, max={native_max:.2f}"
    )
    print(f"Plotting amplitude in {display_unit}.")
    return plot_segment, display_unit


def _plot_eeg_segment(
    segment,
    times,
    eeg_labels,
    title,
    display_unit,
    figsize=(14, 18),
):
    """Plot an offset EEG segment for visual inspection."""
    import matplotlib.pyplot as plt

    segment_centered = segment - np.nanmedian(segment, axis=1, keepdims=True)
    channel_ranges = (
        np.nanpercentile(segment_centered, 95, axis=1)
        - np.nanpercentile(segment_centered, 5, axis=1)
    )
    spacing = max(float(np.nanmedian(channel_ranges) * 1.5), 20.0)
    offsets = np.arange(len(eeg_labels))[::-1] * spacing

    fig, ax = plt.subplots(figsize=figsize)
    for idx, label in enumerate(eeg_labels):
        ax.plot(
            times,
            segment_centered[idx] + offsets[idx],
            color="black",
            linewidth=0.45,
        )

    ax.set_yticks(offsets)
    ax.set_yticklabels(eeg_labels, fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"EEG electrodes (offset; {display_unit})")
    ax.set_title(title)
    ax.set_xlim(times[0], times[-1])
    ax.grid(axis="x", color="0.85", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def _plot_raw_filtered_psd(
    raw_data,
    filtered_data,
    fs,
    l_freq,
    h_freq,
    figsize=(9, 5),
):
    """Plot channel-averaged Welch PSD before and after filtering."""
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    if raw_data.shape != filtered_data.shape:
        raise ValueError(
            "raw_data and filtered_data must have the same shape for PSD "
            f"comparison; got {raw_data.shape} and {filtered_data.shape}."
        )

    nperseg = min(raw_data.shape[1], int(round(fs * 20)))
    if nperseg < fs:
        raise ValueError(
            f"Recording is too short for PSD validation: nperseg={nperseg}, fs={fs}."
        )
    freqs, raw_psd = welch(raw_data, fs=fs, axis=1, nperseg=nperseg)
    _, filtered_psd = welch(filtered_data, fs=fs, axis=1, nperseg=nperseg)
    raw_mean_psd = np.nanmean(raw_psd, axis=0)
    filtered_mean_psd = np.nanmean(filtered_psd, axis=0)

    positive = freqs > 0
    psd_floor = np.finfo(float).tiny
    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(
        freqs[positive],
        np.maximum(raw_mean_psd[positive], psd_floor),
        color="0.6",
        linewidth=1.4,
        label="Raw",
    )
    ax.semilogy(
        freqs[positive],
        np.maximum(filtered_mean_psd[positive], psd_floor),
        color="black",
        linewidth=1.6,
        label=f"Zero-phase {l_freq:g}-{h_freq:g} Hz",
    )
    ax.axvspan(l_freq, h_freq, color="0.9", zorder=0)
    ax.axvline(l_freq, color="0.2", linestyle="--", linewidth=0.8)
    ax.axvline(h_freq, color="0.2", linestyle="--", linewidth=0.8)
    ax.set_xlim(0.03, 60)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean PSD across analysis EEG channels")
    ax.set_title("Raw vs filtered EEG PSD")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    low_band = (freqs > 0) & (freqs < l_freq)
    high_band = (freqs > h_freq) & (freqs <= 60)
    pass_band = (freqs >= max(l_freq, 0.5)) & (freqs <= h_freq)
    print("PSD validation summary:")
    if np.any(low_band):
        low_ratio = (
            np.nanmean(filtered_mean_psd[low_band])
            / np.nanmean(raw_mean_psd[low_band])
        )
        print(f"  Mean PSD ratio below {l_freq:g} Hz, filtered/raw: {low_ratio:.4g}")
    if np.any(pass_band):
        pass_ratio = (
            np.nanmean(filtered_mean_psd[pass_band])
            / np.nanmean(raw_mean_psd[pass_band])
        )
        print(
            f"  Mean PSD ratio inside {max(l_freq, 0.5):g}-{h_freq:g} Hz, "
            f"filtered/raw: {pass_ratio:.4g}"
        )
    if np.any(high_band):
        high_ratio = (
            np.nanmean(filtered_mean_psd[high_band])
            / np.nanmean(raw_mean_psd[high_band])
        )
        print(f"  Mean PSD ratio above {h_freq:g} Hz, filtered/raw: {high_ratio:.4g}")

    return {
        "fig": fig,
        "ax": ax,
        "freqs": freqs,
        "raw_mean_psd": raw_mean_psd,
        "filtered_mean_psd": filtered_mean_psd,
    }


def _extract_training_stimulus_events_from_status(raw, stim_channel="Status"):
    """Extract and validate training stimulus events from the GDF Status channel."""
    import mne

    all_events = mne.find_events(
        raw,
        stim_channel=stim_channel,
        shortest_event=1,
        verbose="ERROR",
    )
    if all_events.size == 0:
        raise ValueError(f"No events found in Status channel '{stim_channel}'.")

    task_codes = {FIXATION_CODE, *STIMULUS_CODES.keys(), RESPONSE_CODE}
    task_events = all_events[np.isin(all_events[:, 2], sorted(task_codes))]
    ignored_events = all_events[~np.isin(all_events[:, 2], sorted(task_codes))]
    if len(ignored_events):
        ignored_codes, ignored_counts = np.unique(
            ignored_events[:, 2],
            return_counts=True,
        )
        print(
            "Ignored non-training Status events: "
            f"{dict(zip(ignored_codes.tolist(), ignored_counts.tolist()))}"
        )

    task_codes_found, task_counts = np.unique(task_events[:, 2], return_counts=True)
    task_count_summary = dict(zip(task_codes_found.tolist(), task_counts.tolist()))
    print(f"Training Status event counts: {task_count_summary}")
    fixation_count = task_count_summary.get(FIXATION_CODE, 0)
    response_count = task_count_summary.get(RESPONSE_CODE, 0)
    stimulus_mask = np.isin(task_events[:, 2], list(STIMULUS_CODES))
    stimulus_events = task_events[stimulus_mask]
    if fixation_count != TRAINING_TRIALS:
        raise ValueError(
            f"Expected {TRAINING_TRIALS} fixation Status events, got "
            f"{fixation_count}."
        )
    if len(stimulus_events) != TRAINING_TRIALS:
        raise ValueError(
            f"Expected {TRAINING_TRIALS} stimulus Status events for epoching, "
            f"got {len(stimulus_events)}."
        )
    if response_count != TRAINING_TRIALS:
        print(
            "WARNING: response Status event count differs from expected "
            f"{TRAINING_TRIALS}: got {response_count}. Stimulus-locked epoching "
            "will continue because all stimulus events are present."
        )

    task_event_indices = np.where(stimulus_mask)[0]
    for trial_idx, task_event_index in enumerate(task_event_indices, start=1):
        if task_event_index == 0:
            raise ValueError(
                f"Stimulus event {trial_idx} has no preceding training Status event."
            )
        previous_code = int(task_events[task_event_index - 1, 2])
        stimulus_code = int(task_events[task_event_index, 2])
        if previous_code != FIXATION_CODE:
            raise ValueError(
                f"Stimulus event {trial_idx} expected preceding fixation code "
                f"{FIXATION_CODE}, got {previous_code} before stimulus "
                f"{stimulus_code}."
            )
        if task_event_index + 1 < len(task_events):
            next_code = int(task_events[task_event_index + 1, 2])
            if next_code not in {RESPONSE_CODE, FIXATION_CODE}:
                raise ValueError(
                    f"Stimulus event {trial_idx} expected next Status event to be "
                    f"response {RESPONSE_CODE} or next fixation {FIXATION_CODE}, "
                    f"got {next_code}."
                )
    stimulus_codes, stimulus_counts = np.unique(
        stimulus_events[:, 2],
        return_counts=True,
    )
    print(
        "Stimulus events selected for epoching from Status channel: "
        f"{dict(zip(stimulus_codes.tolist(), stimulus_counts.tolist()))}"
    )
    return stimulus_events, task_events, all_events


def _epoch_channel_time_trials(data, event_samples, fs, tmin, tmax):
    """Slice continuous data into channel x sample x trial epochs."""
    if data.ndim != 2:
        raise ValueError(f"data must be channels x samples, got shape {data.shape}.")
    if tmin >= 0:
        raise ValueError(f"tmin should be negative for pre-stimulus epochs, got {tmin}.")
    if tmax <= 0:
        raise ValueError(f"tmax should be positive for post-stimulus epochs, got {tmax}.")
    if tmax <= tmin:
        raise ValueError(f"tmax must be greater than tmin, got {tmax} <= {tmin}.")

    start_offset = int(round(tmin * fs))
    stop_offset = int(round(tmax * fs))
    expected_samples = stop_offset - start_offset
    if expected_samples <= 0:
        raise ValueError(
            f"Epoch sample count must be positive, got {expected_samples}."
        )

    epochs = []
    kept_event_samples = []
    dropped = []
    for trial_idx, event_sample in enumerate(event_samples, start=1):
        start = int(event_sample) + start_offset
        stop = int(event_sample) + stop_offset
        if start < 0 or stop > data.shape[1]:
            dropped.append((trial_idx, int(event_sample), start, stop))
            continue
        epoch = data[:, start:stop]
        if epoch.shape[1] != expected_samples:
            raise ValueError(
                f"Trial {trial_idx} epoch has {epoch.shape[1]} samples; "
                f"expected {expected_samples}."
            )
        epochs.append(epoch)
        kept_event_samples.append(int(event_sample))

    if dropped:
        raise ValueError(
            "Some epochs would exceed recording bounds: "
            f"{dropped[:5]}{'...' if len(dropped) > 5 else ''}"
        )
    if not epochs:
        raise ValueError("No epochs were created.")

    epoch_data = np.stack(epochs, axis=2)
    time = np.arange(start_offset, stop_offset) / fs
    zero_index = int(np.where(np.isclose(time, 0.0))[0][0])
    return epoch_data, time, kept_event_samples, zero_index


def _plot_channel_average_epoch(epoch_data, time, eeg_labels, channel, figsize=(7, 4)):
    """Plot one channel averaged across all epochs."""
    import matplotlib.pyplot as plt

    if channel not in eeg_labels:
        raise ValueError(
            f"Channel {channel!r} not found in analysis EEG labels. "
            f"Available labels include: {eeg_labels[:10]}..."
        )
    channel_idx = eeg_labels.index(channel)
    mean_waveform = np.nanmean(epoch_data[channel_idx, :, :], axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, mean_waveform, color="black", linewidth=1.5)
    ax.axvline(0, color="0.25", linestyle="--", linewidth=0.9)
    ax.axhline(0, color="0.8", linewidth=0.7)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel(f"{channel} amplitude (native GDF units)")
    ax.set_title(f"{channel} average across {epoch_data.shape[2]} stimulus-locked trials")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax, mean_waveform


def _plot_channel_average_epoch_comparison(
    before_epoch_data,
    after_epoch_data,
    time,
    eeg_labels,
    channel,
    before_label="Filtered",
    after_label="Baseline corrected",
    baseline_tmin=DEFAULT_BASELINE_TMIN,
    baseline_tmax=DEFAULT_BASELINE_TMAX,
    figsize=(7, 4),
):
    """Plot one channel average before and after a processing step."""
    import matplotlib.pyplot as plt

    if before_epoch_data.shape != after_epoch_data.shape:
        raise ValueError(
            "before_epoch_data and after_epoch_data must have matching shapes; "
            f"got {before_epoch_data.shape} and {after_epoch_data.shape}."
        )
    if channel not in eeg_labels:
        raise ValueError(
            f"Channel {channel!r} not found in analysis EEG labels. "
            f"Available labels include: {eeg_labels[:10]}..."
        )

    channel_idx = eeg_labels.index(channel)
    before_mean = np.nanmean(before_epoch_data[channel_idx, :, :], axis=1)
    after_mean = np.nanmean(after_epoch_data[channel_idx, :, :], axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, before_mean, color="0.65", linewidth=1.2, label=before_label)
    ax.plot(time, after_mean, color="black", linewidth=1.6, label=after_label)
    ax.axvspan(baseline_tmin, baseline_tmax, color="0.92", zorder=0)
    ax.axvline(0, color="0.25", linestyle="--", linewidth=0.9)
    ax.axhline(0, color="0.8", linewidth=0.7)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel(f"{channel} amplitude (native GDF units)")
    ax.set_title(
        f"{channel} grand average across {before_epoch_data.shape[2]} trials"
    )
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax, before_mean, after_mean


def baseline_correct_epochs(
    epoch_data,
    time,
    baseline_tmin=DEFAULT_BASELINE_TMIN,
    baseline_tmax=DEFAULT_BASELINE_TMAX,
):
    """Baseline-correct channel x sample x trial epochs.

    The baseline mean is computed separately for each channel and trial over the
    requested time window, then subtracted from every timepoint in that
    corresponding channel/trial epoch.
    """
    epoch_data = np.asarray(epoch_data)
    time = np.asarray(time)
    if epoch_data.ndim != 3:
        raise ValueError(
            "epoch_data must be channels x samples x trials, "
            f"got shape {epoch_data.shape}."
        )
    if time.ndim != 1:
        raise ValueError(f"time must be a 1D vector, got shape {time.shape}.")
    if epoch_data.shape[1] != len(time):
        raise ValueError(
            "The epoch sample dimension must match len(time); got "
            f"{epoch_data.shape[1]} samples and {len(time)} timepoints."
        )
    if baseline_tmin >= baseline_tmax:
        raise ValueError(
            f"baseline_tmin must be less than baseline_tmax, got "
            f"{baseline_tmin} >= {baseline_tmax}."
        )

    baseline_mask = (time >= baseline_tmin) & (time < baseline_tmax)
    baseline_indices = np.where(baseline_mask)[0]
    if baseline_indices.size == 0:
        raise ValueError(
            f"No baseline samples found for {baseline_tmin:g} to "
            f"{baseline_tmax:g} s."
        )
    baseline_start = int(baseline_indices[0])
    baseline_stop = int(baseline_indices[-1] + 1)

    if not np.isfinite(epoch_data).all():
        bad_count = int(np.size(epoch_data) - np.isfinite(epoch_data).sum())
        raise ValueError(f"Epoch data contains {bad_count} non-finite values.")

    baseline_mean = np.nanmean(
        epoch_data[:, baseline_start:baseline_stop, :],
        axis=1,
        keepdims=True,
    )
    corrected = epoch_data - baseline_mean
    corrected_baseline_mean = np.nanmean(
        corrected[:, baseline_start:baseline_stop, :],
        axis=1,
    )
    max_abs_residual = float(np.nanmax(np.abs(corrected_baseline_mean)))
    if max_abs_residual > 1e-8:
        print(
            "WARNING: baseline residual after correction is larger than expected: "
            f"max abs residual={max_abs_residual:.4g}."
        )

    print("Baseline correction summary:")
    print(
        f"  Baseline window: {baseline_tmin:g} to {baseline_tmax:g} s "
        "(stop sample exclusive)"
    )
    print(
        f"  Baseline sample indices within epoch: {baseline_start}:{baseline_stop}"
    )
    print(f"  Baseline sample count: {baseline_stop - baseline_start}")
    print(
        "  Input/output shape (channels x samples x trials): "
        f"{epoch_data.shape}"
    )
    print(
        "  Per-channel/per-trial baseline means subtracted; "
        f"max abs residual baseline mean={max_abs_residual:.4g}."
    )

    return {
        "baseline_corrected_epochs": corrected,
        "baseline_mean_channels_one_trial": baseline_mean,
        "baseline_indices": (baseline_start, baseline_stop),
        "baseline_time": time[baseline_start:baseline_stop],
        "max_abs_residual_baseline_mean": max_abs_residual,
    }


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


def load_and_plot_eeg_run_segment(
    subject_id=None,
    session=1,
    run=1,
    task="training",
    gdf_path=None,
    project_root=PROJECT_ROOT,
    start_sec=30.0,
    duration_sec=10.0,
    excluded_channels=DEFAULT_EXCLUDED_NON_EEG_CHANNELS,
    expected_fs=FS,
    figsize=(14, 18),
):
    """Load one EEG run and plot an offset segment of analysis EEG electrodes.

    Parameters
    ----------
    subject_id : str, optional
        Subject identifier such as ``"e21"``. Required when ``gdf_path`` is not
        supplied.
    session : int, default=1
        1-based session index within the subject folder.
    run : int, default=1
        1-based run index within the selected task, excluding practice by exact
        task matching.
    task : str, default="training"
        Exact task folder suffix to use, for example ``"training"``.
    gdf_path : str or Path, optional
        Direct path to a GDF file. If supplied, subject/session/run resolution is
        skipped.
    project_root : str or Path, default=PROJECT_ROOT
        Base project data directory.
    start_sec : float, default=30.0
        Start time in seconds for the plotted segment.
    duration_sec : float, default=10.0
        Duration in seconds for the plotted segment.
    excluded_channels : tuple, default=("M1", "M2", "EOG", "sens7", "sens8")
        Non-scalp-EEG channels to remove before plotting or returning analysis
        arrays. The default excludes mastoids and EOG channels.
    expected_fs : int, default=512
        Expected sampling rate from the preprocessing documentation.
    figsize : tuple, default=(14, 18)
        Matplotlib figure size.

    Returns
    -------
    dict
        Contains the MNE Raw object, EEG segment array, figure, axis, and
        metadata useful for follow-up inspection.
    """
    gdf_path = _resolve_gdf_path(
        subject_id=subject_id,
        session=session,
        run=run,
        task=task,
        gdf_path=gdf_path,
        project_root=project_root,
    )
    _validate_paired_training_files(gdf_path, task)
    _prepare_plot_environment()
    import mne

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    channel_labels = list(raw.ch_names)
    n_channels = len(channel_labels)
    n_samples = raw.n_times
    duration_total_sec = n_samples / fs

    if not np.isclose(fs, expected_fs):
        print(
            f"WARNING: expected sampling rate {expected_fs} Hz from docs, "
            f"but file reports {fs:g} Hz."
        )
    if n_channels != EXPECTED_TOTAL_CHANNELS:
        print(
            f"WARNING: docs describe {EXPECTED_TOTAL_CHANNELS} total channels "
            f"(64 EEG + 2 EOG + Status), file has {n_channels}."
        )

    start_sample = int(round(start_sec * fs))
    stop_sample = int(round((start_sec + duration_sec) * fs))
    if start_sample < 0:
        raise ValueError(f"start_sec must be non-negative, got {start_sec}.")
    if stop_sample > n_samples:
        raise ValueError(
            f"Requested {start_sec:g}-{start_sec + duration_sec:g} s, but recording "
            f"is only {duration_total_sec:.2f} s long."
        )

    eeg_labels, excluded_found, status_labels = select_analysis_eeg_channels(
        channel_labels,
        excluded_channels=excluded_channels,
    )
    if len(eeg_labels) > EXPECTED_EEG_CHANNELS:
        print(
            f"WARNING: selected {len(eeg_labels)} analysis EEG channels, "
            f"which exceeds the documented first-{EXPECTED_EEG_CHANNELS} EEG block."
        )
    eeg_data = raw.get_data(picks=eeg_labels)
    eeg_shape_samples_channels = (eeg_data.shape[1], eeg_data.shape[0])
    eeg_segment = eeg_data[:, start_sample:stop_sample]
    if not np.isfinite(eeg_segment).all():
        bad_count = int(np.size(eeg_segment) - np.isfinite(eeg_segment).sum())
        raise ValueError(f"EEG segment contains {bad_count} non-finite values.")

    print(
        "Analysis EEG data shape after excluding non-EEG channels "
        f"(samples x electrodes): {eeg_shape_samples_channels}"
    )
    print(f"Full recording shape (samples x channels): {(n_samples, n_channels)}")
    print(f"Sampling rate: {fs:g} Hz")
    print(f"Recording duration: {duration_total_sec:.2f} s")
    print(
        f"Channel count: {n_channels}; plotting {len(eeg_labels)} analysis EEG "
        "electrodes after exclusions."
    )
    print(f"Excluded non-analysis channels: {excluded_found}")
    print(f"First 8 EEG labels: {eeg_labels[:8]}")
    print(f"Status/trigger channel labels detected: {status_labels or 'none'}")
    print(
        f"Plot segment: {start_sec:g}-{start_sec + duration_sec:g} s "
        f"({start_sample}:{stop_sample} samples), shape {eeg_segment.T.shape} "
        "(samples x EEG electrodes)."
    )
    times = np.arange(start_sample, stop_sample) / fs
    plot_segment, display_unit = _infer_display_scale(eeg_segment)
    fig, ax = _plot_eeg_segment(
        segment=plot_segment,
        times=times,
        eeg_labels=eeg_labels,
        title=(
            f"{gdf_path.stem}: {len(eeg_labels)} analysis EEG electrodes, "
            f"{duration_sec:g} s from {start_sec:g} s"
        ),
        display_unit=display_unit,
        figsize=figsize,
    )

    return {
        "raw": raw,
        "gdf_path": gdf_path,
        "eeg_labels": eeg_labels,
        "excluded_non_eeg_channels": excluded_found,
        "eeg_shape_samples_channels": eeg_shape_samples_channels,
        "segment_samples_by_channels": eeg_segment.T,
        "display_unit": display_unit,
        "sample_rate": fs,
        "start_sample": start_sample,
        "stop_sample": stop_sample,
        "fig": fig,
        "ax": ax,
    }


def load_filter_epoch_baseline_correct_and_plot_training(
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
    plot_channel="PO7",
    excluded_channels=DEFAULT_EXCLUDED_NON_EEG_CHANNELS,
    expected_fs=FS,
):
    """Filter, stimulus-epoch, baseline-correct, and plot one training run."""
    epoch_results = load_filter_epoch_and_plot_training_stimulus_epochs(
        subject_id=subject_id,
        session=session,
        run=run,
        gdf_path=gdf_path,
        project_root=project_root,
        l_freq=l_freq,
        h_freq=h_freq,
        tmin=tmin,
        tmax=tmax,
        plot_channel=plot_channel,
        excluded_channels=excluded_channels,
        expected_fs=expected_fs,
    )
    baseline_results = baseline_correct_epochs(
        epoch_results["epoch_data_channels_samples_trials"],
        epoch_results["time"],
        baseline_tmin=baseline_tmin,
        baseline_tmax=baseline_tmax,
    )
    baseline_fig, baseline_ax, filtered_mean, corrected_mean = (
        _plot_channel_average_epoch_comparison(
            before_epoch_data=epoch_results["epoch_data_channels_samples_trials"],
            after_epoch_data=baseline_results["baseline_corrected_epochs"],
            time=epoch_results["time"],
            eeg_labels=epoch_results["eeg_labels"],
            channel=plot_channel,
            baseline_tmin=baseline_tmin,
            baseline_tmax=baseline_tmax,
        )
    )

    combined = dict(epoch_results)
    combined.update(baseline_results)
    combined.update({
        "baseline_tmin": baseline_tmin,
        "baseline_tmax": baseline_tmax,
        "baseline_fig": baseline_fig,
        "baseline_ax": baseline_ax,
        "plot_channel_filtered_mean": filtered_mean,
        "plot_channel_baseline_corrected_mean": corrected_mean,
    })
    return combined


def load_filter_epoch_baseline_correct_training_run(
    gdf_path,
    l_freq=DEFAULT_BANDPASS_L_FREQ,
    h_freq=DEFAULT_BANDPASS_H_FREQ,
    tmin=DEFAULT_EPOCH_TMIN,
    tmax=DEFAULT_EPOCH_TMAX,
    baseline_tmin=DEFAULT_BASELINE_TMIN,
    baseline_tmax=DEFAULT_BASELINE_TMAX,
    excluded_channels=DEFAULT_EXCLUDED_NON_EEG_CHANNELS,
    expected_fs=FS,
    validate_paired_files=True,
):
    """Preprocess one training run without plotting for batch analyses."""
    gdf_path = _resolve_gdf_path(
        subject_id=None,
        session=None,
        run=None,
        task="training",
        gdf_path=gdf_path,
        project_root=PROJECT_ROOT,
    )
    if validate_paired_files:
        _validate_paired_training_files(gdf_path, task="training")
    _prepare_plot_environment()
    import mne

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    if not np.isclose(fs, expected_fs):
        raise ValueError(
            f"Expected sampling rate {expected_fs} Hz, file reports {fs:g} Hz "
            f"for {gdf_path}."
        )

    stimulus_events, task_events, all_events = _extract_training_stimulus_events_from_status(
        raw,
        stim_channel="Status",
    )
    eeg_labels, excluded_found, status_labels = select_analysis_eeg_channels(
        list(raw.ch_names),
        excluded_channels=excluded_channels,
    )
    raw_analysis_eeg = raw.copy().pick(eeg_labels)
    filtered_analysis_eeg = raw_analysis_eeg.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks="data",
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )
    filtered_data = filtered_analysis_eeg.get_data()
    epoch_data, time, kept_event_samples, zero_index = _epoch_channel_time_trials(
        data=filtered_data,
        event_samples=stimulus_events[:, 0],
        fs=fs,
        tmin=tmin,
        tmax=tmax,
    )
    baseline_results = baseline_correct_epochs(
        epoch_data,
        time,
        baseline_tmin=baseline_tmin,
        baseline_tmax=baseline_tmax,
    )

    expected_shape = (len(eeg_labels), int(round((tmax - tmin) * fs)), TRAINING_TRIALS)
    if baseline_results["baseline_corrected_epochs"].shape != expected_shape:
        raise ValueError(
            f"Unexpected baseline-corrected epoch shape for {gdf_path}. "
            f"Expected {expected_shape}, got "
            f"{baseline_results['baseline_corrected_epochs'].shape}."
        )

    return {
        "gdf_path": gdf_path,
        "eeg_labels": eeg_labels,
        "excluded_non_eeg_channels": excluded_found,
        "status_labels": status_labels,
        "stimulus_events": stimulus_events,
        "task_events": task_events,
        "all_status_events": all_events,
        "epoch_data_channels_samples_trials": epoch_data,
        "baseline_corrected_epochs": baseline_results["baseline_corrected_epochs"],
        "baseline_indices": baseline_results["baseline_indices"],
        "max_abs_residual_baseline_mean": baseline_results[
            "max_abs_residual_baseline_mean"
        ],
        "time": time,
        "zero_index": zero_index,
        "sample_rate": fs,
        "kept_event_samples": kept_event_samples,
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
        output_path = project_root / "analyses" / "training_session_condition_eeg_averages.csv"
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


def summarize_group_pd_pre_post(
    subject_session_pd,
    groups=("bci", "control"),
    sessions=(1, 5),
):
    """Average subject-level Pd waveforms by group/session and compute SEM."""
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

    data = subject_session_pd.copy()
    summary = (
        data.groupby(["group", "session_id", "sample_index"], observed=False)
        .agg(
            time_sec=("time_sec", "mean"),
            mean_pd_uv=("pd_amplitude_uv", "mean"),
            sd_pd_uv=("pd_amplitude_uv", "std"),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )
    summary["sem_pd_uv"] = summary["sd_pd_uv"] / np.sqrt(summary["n_subjects"])
    summary["sem_pd_uv"] = summary["sem_pd_uv"].fillna(0.0)

    expected_cells = pd.MultiIndex.from_product(
        [groups, sessions],
        names=["group", "session_id"],
    )
    observed_cells = (
        summary[["group", "session_id"]]
        .drop_duplicates()
        .set_index(["group", "session_id"])
        .index
    )
    missing_cells = expected_cells.difference(observed_cells)
    if len(missing_cells):
        raise ValueError(
            "Missing group/session cells in Pd summary: "
            f"{list(missing_cells)}"
        )

    print("Group Pd pre/post summary:")
    print(
        summary.groupby(["group", "session_id"], observed=False)["n_subjects"]
        .max()
        .reset_index()
        .sort_values(["group", "session_id"])
        .to_string(index=False)
    )
    return summary


def plot_group_pd_pre_post(
    group_pd_summary,
    group,
    group_label=None,
    time_window=DEFAULT_PD_PLOT_TIME_WINDOW,
    decoder_window=DEFAULT_PD_DECODER_TIME_WINDOW,
    session_labels=None,
    colors=None,
    figsize=(4.2, 4.2),
    y_limits=(-1.5, 1.5),
    session_rt_sec=None,
    ax=None,
):
    """Plot Session 1 vs Session 5 group-average Pd with subject SEM shading."""
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

    group = str(group).lower().strip()
    if group_label is None:
        group_label = {"bci": "BCI", "control": "Mental rehearsal"}.get(group, group)
    if session_labels is None:
        session_labels = {1: "Session 1", 5: "Session 5"}
    if colors is None:
        colors = {1: "#4C72B0", 5: "#DD8452"}

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    subset = group_pd_summary[group_pd_summary["group"].str.lower() == group].copy()
    if subset.empty:
        raise ValueError(f"No Pd summary rows found for group {group!r}.")
    subset = subset[
        (subset["time_sec"] >= time_window[0])
        & (subset["time_sec"] <= time_window[1])
    ].sort_values(["session_id", "sample_index"])
    if subset.empty:
        raise ValueError(f"No Pd samples found in time window {time_window}.")

    for session_id in (1, 5):
        session_df = subset[subset["session_id"] == session_id].sort_values("time_sec")
        if session_df.empty:
            raise ValueError(f"No Pd rows for group {group}, session {session_id}.")
        time = session_df["time_sec"].to_numpy()
        mean = session_df["mean_pd_uv"].to_numpy()
        sem = session_df["sem_pd_uv"].to_numpy()
        color = colors[session_id]
        label = session_labels.get(session_id, f"Session {session_id}")
        ax.plot(time, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(time, mean - sem, mean + sem, color=color, alpha=0.22, linewidth=0)

    if session_rt_sec is not None:
        for session_id, rt_sec in session_rt_sec.items():
            if rt_sec is None or not np.isfinite(rt_sec):
                continue
            if time_window[0] <= rt_sec <= time_window[1]:
                ax.axvline(
                    rt_sec,
                    color=colors[int(session_id)],
                    linestyle=":",
                    linewidth=1.1,
                    alpha=0.95,
                )

    ax.axvspan(
        decoder_window[0],
        decoder_window[1],
        color="0.85",
        alpha=0.55,
        linewidth=0,
        zorder=0,
    )
    ax.axvline(0, color="0.2", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.set_xlim(time_window)
    if y_limits is not None:
        if y_limits[0] >= y_limits[1]:
            raise ValueError(f"y_limits must be increasing, got {y_limits}.")
        ax.set_ylim(y_limits)
    xticks = np.arange(time_window[0], time_window[1] + 1e-9, 0.2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{tick:.1f}" for tick in xticks])
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Pd amplitude (microvolts)")
    ax.set_title(f"{group_label}: Pd pre vs post")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    plotted_subjects = (
        group_pd_summary[group_pd_summary["group"].str.lower() == group]
        .groupby("session_id", observed=False)["n_subjects"]
        .max()
        .to_dict()
    )
    print(
        f"Plotted {group_label} Pd pre/post over {time_window[0]:g} to "
        f"{time_window[1]:g} s; decoder window shaded "
        f"{decoder_window[0]:g} to {decoder_window[1]:g} s. "
        f"Subjects/session: {plotted_subjects}"
    )
    return fig, ax


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
):
    """Load condition-average EEG CSV and plot Pd pre/post for each group."""
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
        )
        figures[group] = fig
        axes[group] = ax
        if save_figures:
            for figure_format in figure_formats:
                path = output_dir / f"{stem}.{figure_format}"
                fig.savefig(path, dpi=300, bbox_inches="tight")
                saved_paths.append(path)
                print(f"Saved figure: {path}")

    return {
        "subject_session_pd": subject_session_pd,
        "group_pd_summary": group_pd_summary,
        "figures": figures,
        "axes": axes,
        "saved_paths": saved_paths,
    }


def compute_pd_positive_auc(
    subject_session_pd,
    time_window=DEFAULT_PD_DECODER_TIME_WINDOW,
):
    """Compute subject/session positive Pd area in the requested time window.

    Positive AUC is the trapezoidal integral of ``max(Pd, 0)`` over time, so
    units are microvolts x seconds.
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
        raise ValueError("Pd AUC input contains non-finite time or amplitude values.")

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
        positive_wave = np.maximum(pd_wave, 0.0)
        positive_auc = float(np.trapezoid(positive_wave, time))
        rows.append({
            "subject_id": subject_id,
            "group": group,
            "session_id": int(session_id),
            "time_window_start_sec": float(time_window[0]),
            "time_window_end_sec": float(time_window[1]),
            "n_samples": int(len(time)),
            "positive_auc_uv_sec": positive_auc,
            "mean_pd_uv_in_window": float(np.mean(pd_wave)),
            "mean_positive_pd_uv_in_window": float(np.mean(positive_wave)),
        })

    auc_df = pd.DataFrame(rows).sort_values(["group", "subject_id", "session_id"])
    expected_sessions = [1, 5]
    observed_sessions = sorted(auc_df["session_id"].unique().tolist())
    if observed_sessions != expected_sessions:
        raise ValueError(
            f"Expected Pd AUC sessions {expected_sessions}, found {observed_sessions}."
        )
    duplicate_cells = (
        auc_df.groupby(["subject_id", "session_id"], observed=False)
        .size()
        .reset_index(name="n_rows")
    )
    duplicate_cells = duplicate_cells[duplicate_cells["n_rows"] != 1]
    if not duplicate_cells.empty:
        raise ValueError(
            "Expected exactly one Pd AUC row per subject/session. Problem cells:\n"
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
    print("Pd positive AUC summary:")
    print(
        f"  Window: {time_window[0]:g} to {time_window[1]:g} s; "
        "integral of max(Pd, 0)."
    )
    print("  Subjects by group/session:")
    print(counts.to_string(index=False))
    print("  Samples per group/session:")
    print(sample_counts.to_string(index=False))
    return auc_df


def summarize_pd_positive_auc(auc_df):
    """Summarize positive Pd AUC by group/session."""
    import pandas as pd

    required_columns = {"subject_id", "group", "session_id", "positive_auc_uv_sec"}
    missing_columns = sorted(required_columns - set(auc_df.columns))
    if missing_columns:
        raise ValueError(f"auc_df is missing columns: {missing_columns}")

    summary = (
        auc_df.groupby(["group", "session_id"], observed=False)["positive_auc_uv_sec"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "mean_positive_auc_uv_sec",
            "std": "sd_positive_auc_uv_sec",
            "count": "n_subjects",
        })
    )
    summary["sem_positive_auc_uv_sec"] = (
        summary["sd_positive_auc_uv_sec"] / np.sqrt(summary["n_subjects"])
    )
    print("\nPd positive AUC cell summary:")
    print(summary.to_string(index=False))
    return summary


def run_pd_positive_auc_mixed_anova(auc_df):
    """Run Group x Session mixed ANOVA on subject-level positive Pd AUC."""
    from scipy import stats
    import pandas as pd

    required_columns = {"subject_id", "group", "session_id", "positive_auc_uv_sec"}
    missing_columns = sorted(required_columns - set(auc_df.columns))
    if missing_columns:
        raise ValueError(f"Pd positive AUC ANOVA requires columns: {missing_columns}")

    data = auc_df[list(required_columns)].copy()
    data["session_id"] = data["session_id"].astype(int)
    data = data.dropna(subset=["subject_id", "group", "session_id", "positive_auc_uv_sec"])

    print("=" * 80)
    print("PD POSITIVE AUC MIXED-DESIGN ANOVA")
    print("=" * 80)
    print("Design: Group (between: BCI vs mental rehearsal) x Session (within: 1 vs 5)")

    groups = sorted(data["group"].unique().tolist())
    sessions = sorted(data["session_id"].unique().tolist())
    if groups != ["bci", "control"]:
        raise ValueError(f"Expected groups ['bci', 'control'], found {groups}.")
    if sessions != [1, 5]:
        raise ValueError(f"Expected sessions [1, 5], found {sessions}.")

    counts = (
        data.groupby(["subject_id", "group"], observed=False)["session_id"]
        .nunique()
        .reset_index(name="n_sessions")
    )
    incomplete = counts[counts["n_sessions"] != len(sessions)]
    if not incomplete.empty:
        raise ValueError(
            "Mixed ANOVA requires complete Session 1 and Session 5 Pd AUC for "
            f"every subject. Incomplete subjects:\n{incomplete.to_string(index=False)}"
        )

    duplicate_cells = (
        data.groupby(["subject_id", "session_id"], observed=False)
        .size()
        .reset_index(name="n_rows")
    )
    duplicate_cells = duplicate_cells[duplicate_cells["n_rows"] != 1]
    if not duplicate_cells.empty:
        raise ValueError(
            "Expected exactly one Pd AUC row per subject/session. Problem cells:\n"
            f"{duplicate_cells.to_string(index=False)}"
        )

    subjects_by_group = (
        data[["subject_id", "group"]]
        .drop_duplicates()
        .groupby("group", observed=False)
        .size()
        .to_dict()
    )
    if len(set(subjects_by_group.values())) != 1:
        raise ValueError(
            "This ANOVA helper expects a balanced group design. "
            f"Subject counts by group: {subjects_by_group}"
        )
    print(f"Subjects by group: {subjects_by_group}")

    value_col = "positive_auc_uv_sec"
    grand_mean = data[value_col].mean()
    group_means = data.groupby("group", observed=False)[value_col].mean()
    session_means = data.groupby("session_id", observed=False)[value_col].mean()
    group_session_means = data.groupby(["group", "session_id"], observed=False)[value_col].mean()
    subject_means = data.groupby(["group", "subject_id"], observed=False)[value_col].mean()

    n_sessions = len(sessions)
    n_total_subjects = data["subject_id"].nunique()
    n_by_group = data[["subject_id", "group"]].drop_duplicates().groupby("group").size()

    ss_group = n_sessions * sum(
        n_by_group[group] * (group_means[group] - grand_mean) ** 2
        for group in groups
    )
    ss_subject_group = n_sessions * sum(
        (subject_means[(group, subject_id)] - group_means[group]) ** 2
        for group in groups
        for subject_id in data.loc[data["group"] == group, "subject_id"].unique()
    )
    ss_session = n_total_subjects * sum(
        (session_means[session_id] - grand_mean) ** 2
        for session_id in sessions
    )
    ss_group_session = sum(
        n_by_group[group] * (
            group_session_means[(group, session_id)]
            - group_means[group]
            - session_means[session_id]
            + grand_mean
        ) ** 2
        for group in groups
        for session_id in sessions
    )
    ss_error = 0.0
    for row in data.itertuples(index=False):
        subject_mean = subject_means[(row.group, row.subject_id)]
        group_session_mean = group_session_means[(row.group, row.session_id)]
        group_mean = group_means[row.group]
        value = getattr(row, value_col)
        ss_error += (value - subject_mean - group_session_mean + group_mean) ** 2

    df_group = len(groups) - 1
    df_subject_group = n_total_subjects - len(groups)
    df_session = len(sessions) - 1
    df_group_session = df_group * df_session
    df_error = df_subject_group * df_session

    ms_group = ss_group / df_group
    ms_subject_group = ss_subject_group / df_subject_group
    ms_session = ss_session / df_session
    ms_group_session = ss_group_session / df_group_session
    ms_error = ss_error / df_error

    rows = [
        {
            "effect": "Group",
            "ss": ss_group,
            "df": df_group,
            "ms": ms_group,
            "error_term": "Subject(Group)",
            "error_df": df_subject_group,
            "F": ms_group / ms_subject_group,
            "p_value": stats.f.sf(ms_group / ms_subject_group, df_group, df_subject_group),
            "partial_eta_sq": ss_group / (ss_group + ss_subject_group),
        },
        {
            "effect": "Session",
            "ss": ss_session,
            "df": df_session,
            "ms": ms_session,
            "error_term": "Session x Subject(Group)",
            "error_df": df_error,
            "F": ms_session / ms_error,
            "p_value": stats.f.sf(ms_session / ms_error, df_session, df_error),
            "partial_eta_sq": ss_session / (ss_session + ss_error),
        },
        {
            "effect": "Group x Session",
            "ss": ss_group_session,
            "df": df_group_session,
            "ms": ms_group_session,
            "error_term": "Session x Subject(Group)",
            "error_df": df_error,
            "F": ms_group_session / ms_error,
            "p_value": stats.f.sf(ms_group_session / ms_error, df_group_session, df_error),
            "partial_eta_sq": ss_group_session / (ss_group_session + ss_error),
        },
    ]
    anova_table = pd.DataFrame(rows)
    print("\nMixed-design ANOVA table:")
    print(anova_table.to_string(index=False))

    diagnostics = {
        "grand_mean_positive_auc_uv_sec": float(grand_mean),
        "subjects_by_group": subjects_by_group,
        "n_subjects": int(n_total_subjects),
        "sessions": sessions,
        "ss_subject_group": float(ss_subject_group),
        "df_subject_group": int(df_subject_group),
        "ss_error": float(ss_error),
        "df_error": int(df_error),
    }
    return {
        "anova_table": anova_table,
        "diagnostics": diagnostics,
        "input_data": data,
    }


def _mean_ci(values, confidence=0.95):
    """Return a t-based confidence interval around a one-sample mean."""
    from scipy import stats

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    sem = stats.sem(values)
    if not np.isfinite(sem):
        return np.nan, np.nan
    interval = stats.t.interval(confidence, len(values) - 1, loc=np.mean(values), scale=sem)
    return float(interval[0]), float(interval[1])


def _welch_df(sample_a, sample_b):
    """Welch-Satterthwaite degrees of freedom for independent samples."""
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    var_a = np.var(sample_a, ddof=1)
    var_b = np.var(sample_b, ddof=1)
    n_a = len(sample_a)
    n_b = len(sample_b)
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = ((var_a / n_a) ** 2 / (n_a - 1)) + ((var_b / n_b) ** 2 / (n_b - 1))
    return float(numerator / denominator) if denominator > 0 else np.nan


def _welch_mean_difference_ci(sample_a, sample_b, confidence=0.95):
    """Return CI for mean(sample_a) - mean(sample_b) using Welch SE."""
    from scipy import stats

    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    if len(sample_a) < 2 or len(sample_b) < 2:
        return np.nan, np.nan
    diff = np.mean(sample_a) - np.mean(sample_b)
    se = np.sqrt(np.var(sample_a, ddof=1) / len(sample_a) + np.var(sample_b, ddof=1) / len(sample_b))
    df = _welch_df(sample_a, sample_b)
    if not np.isfinite(se) or not np.isfinite(df):
        return np.nan, np.nan
    critical = stats.t.ppf((1 + confidence) / 2, df)
    return float(diff - critical * se), float(diff + critical * se)


def _cohens_dz(diff_values):
    """Cohen's dz for paired differences."""
    diff_values = np.asarray(diff_values, dtype=float)
    sd = np.std(diff_values, ddof=1)
    return float(np.mean(diff_values) / sd) if len(diff_values) > 1 and sd > 0 else np.nan


def _cohens_d_independent(sample_a, sample_b):
    """Pooled Cohen's d for independent samples."""
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    if len(sample_a) < 2 or len(sample_b) < 2:
        return np.nan
    pooled_var = (
        ((len(sample_a) - 1) * np.var(sample_a, ddof=1))
        + ((len(sample_b) - 1) * np.var(sample_b, ddof=1))
    ) / (len(sample_a) + len(sample_b) - 2)
    pooled_sd = np.sqrt(pooled_var)
    return float((np.mean(sample_a) - np.mean(sample_b)) / pooled_sd) if pooled_sd > 0 else np.nan


def run_pd_positive_auc_planned_contrasts(auc_df):
    """Run planned contrasts for subject-level positive Pd AUC.

    Planned contrasts:
    - BCI Session 5 minus Session 1
    - Control Session 5 minus Session 1
    - BCI minus Control at Session 1
    - BCI minus Control at Session 5
    """
    from scipy import stats
    import pandas as pd

    required_columns = {"subject_id", "group", "session_id", "positive_auc_uv_sec"}
    missing_columns = sorted(required_columns - set(auc_df.columns))
    if missing_columns:
        raise ValueError(f"Pd positive AUC planned contrasts require columns: {missing_columns}")

    data = auc_df[list(required_columns)].copy()
    data["session_id"] = data["session_id"].astype(int)
    data = data.dropna(subset=["subject_id", "group", "session_id", "positive_auc_uv_sec"])
    groups = sorted(data["group"].unique().tolist())
    sessions = sorted(data["session_id"].unique().tolist())
    if groups != ["bci", "control"]:
        raise ValueError(f"Expected groups ['bci', 'control'], found {groups}.")
    if sessions != [1, 5]:
        raise ValueError(f"Expected sessions [1, 5], found {sessions}.")

    rows = []
    value_col = "positive_auc_uv_sec"

    for group in ["bci", "control"]:
        group_df = data[data["group"] == group]
        wide = group_df.pivot_table(index="subject_id", columns="session_id", values=value_col)
        wide = wide.dropna(subset=[1, 5])
        if wide.empty:
            raise ValueError(f"No complete Session 1/5 rows for group {group}.")
        diff = wide[5] - wide[1]
        t_stat, p_value = stats.ttest_rel(wide[5], wide[1])
        ci_low, ci_high = _mean_ci(diff)
        rows.append({
            "contrast": f"{group}: Session 5 - Session 1",
            "contrast_type": "within_group_change",
            "group": group,
            "session_id": "5-1",
            "n_subjects": int(len(wide)),
            "mean_session_1": float(wide[1].mean()),
            "mean_session_5": float(wide[5].mean()),
            "estimate_uv_sec": float(diff.mean()),
            "statistic": float(t_stat),
            "df": float(len(wide) - 1),
            "p_value": float(p_value),
            "effect_size": _cohens_dz(diff),
            "effect_size_label": "cohens_dz",
            "ci95_low_uv_sec": ci_low,
            "ci95_high_uv_sec": ci_high,
        })

    for session_id in [1, 5]:
        session_df = data[data["session_id"] == session_id]
        bci_values = session_df.loc[session_df["group"] == "bci", value_col].to_numpy()
        control_values = session_df.loc[session_df["group"] == "control", value_col].to_numpy()
        if len(bci_values) == 0 or len(control_values) == 0:
            raise ValueError(f"Missing BCI or control values for session {session_id}.")
        t_stat, p_value = stats.ttest_ind(bci_values, control_values, equal_var=False)
        ci_low, ci_high = _welch_mean_difference_ci(bci_values, control_values)
        rows.append({
            "contrast": f"Session {session_id}: BCI - Control",
            "contrast_type": "between_group_difference",
            "group": "bci-control",
            "session_id": int(session_id),
            "n_subjects": int(len(bci_values) + len(control_values)),
            "mean_bci": float(np.mean(bci_values)),
            "mean_control": float(np.mean(control_values)),
            "estimate_uv_sec": float(np.mean(bci_values) - np.mean(control_values)),
            "statistic": float(t_stat),
            "df": _welch_df(bci_values, control_values),
            "p_value": float(p_value),
            "effect_size": _cohens_d_independent(bci_values, control_values),
            "effect_size_label": "cohens_d",
            "ci95_low_uv_sec": ci_low,
            "ci95_high_uv_sec": ci_high,
        })

    contrast_df = pd.DataFrame(rows)
    print("\nPlanned Pd positive AUC contrasts:")
    print(contrast_df.to_string(index=False))
    return contrast_df


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
            output_path = PROJECT_ROOT / "analyses" / output_filename
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
            output_path = PROJECT_ROOT / "analyses" / DEFAULT_PD_NEGATIVE_AUC_OUTPUT_FILENAME
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
            output_path = PROJECT_ROOT / "analyses" / DEFAULT_PD_MEAN_AMPLITUDE_OUTPUT_FILENAME
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
            output_path = PROJECT_ROOT / "analyses" / DEFAULT_PD_MEAN_ABSOLUTE_AMPLITUDE_OUTPUT_FILENAME
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


def load_filter_epoch_and_plot_training_stimulus_epochs(
    subject_id=None,
    session=1,
    run=1,
    gdf_path=None,
    project_root=PROJECT_ROOT,
    l_freq=DEFAULT_BANDPASS_L_FREQ,
    h_freq=DEFAULT_BANDPASS_H_FREQ,
    tmin=DEFAULT_EPOCH_TMIN,
    tmax=DEFAULT_EPOCH_TMAX,
    plot_channel="PO7",
    excluded_channels=DEFAULT_EXCLUDED_NON_EEG_CHANNELS,
    expected_fs=FS,
):
    """Filter one training run and epoch to stimulus triggers from the Status channel.

    Epochs are returned as ``channels x samples x trials``.
    """
    gdf_path = _resolve_gdf_path(
        subject_id=subject_id,
        session=session,
        run=run,
        task="training",
        gdf_path=gdf_path,
        project_root=project_root,
    )
    _validate_paired_training_files(gdf_path, task="training")
    _prepare_plot_environment()
    import mne

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    if not np.isclose(fs, expected_fs):
        print(
            f"WARNING: expected sampling rate {expected_fs} Hz from docs, "
            f"but file reports {fs:g} Hz."
        )

    stimulus_events, task_events, all_events = _extract_training_stimulus_events_from_status(
        raw,
        stim_channel="Status",
    )

    channel_labels = list(raw.ch_names)
    eeg_labels, excluded_found, status_labels = select_analysis_eeg_channels(
        channel_labels,
        excluded_channels=excluded_channels,
    )
    raw_analysis_eeg = raw.copy().pick(eeg_labels)
    print(
        f"Applying zero-phase bandpass filter before epoching: "
        f"{l_freq:g}-{h_freq:g} Hz."
    )
    filtered_analysis_eeg = raw_analysis_eeg.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks="data",
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )
    filtered_data = filtered_analysis_eeg.get_data()
    if not np.isfinite(filtered_data).all():
        bad_count = int(np.size(filtered_data) - np.isfinite(filtered_data).sum())
        raise ValueError(f"Filtered EEG data contains {bad_count} non-finite values.")

    epoch_data, time, kept_event_samples, zero_index = _epoch_channel_time_trials(
        data=filtered_data,
        event_samples=stimulus_events[:, 0],
        fs=fs,
        tmin=tmin,
        tmax=tmax,
    )
    expected_samples = int(round((tmax - tmin) * fs))
    expected_channels = len(eeg_labels)
    expected_trials = TRAINING_TRIALS
    if epoch_data.shape != (expected_channels, expected_samples, expected_trials):
        raise ValueError(
            "Unexpected epoch matrix shape. Expected "
            f"{(expected_channels, expected_samples, expected_trials)}, "
            f"got {epoch_data.shape}."
        )

    print("Epoch validation summary:")
    print(f"  Epoch anchor: stimulus presentation triggers {list(STIMULUS_CODES)}")
    print(f"  Epoch window: {tmin:g} to {tmax:g} s, stop sample exclusive")
    print(f"  Sampling rate: {fs:g} Hz")
    print(f"  Zero-time sample index in epoch: {zero_index}")
    print(f"  Excluded non-analysis channels: {excluded_found}")
    print(f"  Status/trigger channel labels detected: {status_labels or 'none'}")
    print(
        "  Epoch matrix shape (channels x samples x trials): "
        f"{epoch_data.shape}"
    )
    print(
        f"  Dimensions make sense: {expected_channels} analysis EEG channels x "
        f"{expected_samples} samples ({tmax - tmin:g} s at {fs:g} Hz) x "
        f"{expected_trials} stimulus trials."
    )

    epoch_fig, epoch_ax, mean_waveform = _plot_channel_average_epoch(
        epoch_data=epoch_data,
        time=time,
        eeg_labels=eeg_labels,
        channel=plot_channel,
    )

    return {
        "raw": raw,
        "raw_analysis_eeg": raw_analysis_eeg,
        "filtered_analysis_eeg": filtered_analysis_eeg,
        "gdf_path": gdf_path,
        "eeg_labels": eeg_labels,
        "excluded_non_eeg_channels": excluded_found,
        "stimulus_events": stimulus_events,
        "task_events": task_events,
        "all_status_events": all_events,
        "epoch_data_channels_samples_trials": epoch_data,
        "time": time,
        "zero_index": zero_index,
        "kept_event_samples": kept_event_samples,
        "sample_rate": fs,
        "l_freq": l_freq,
        "h_freq": h_freq,
        "tmin": tmin,
        "tmax": tmax,
        "plot_channel": plot_channel,
        "plot_channel_mean": mean_waveform,
        "epoch_fig": epoch_fig,
        "epoch_ax": epoch_ax,
    }


def load_filter_and_plot_eeg_run_segment(
    subject_id=None,
    session=1,
    run=1,
    task="training",
    gdf_path=None,
    project_root=PROJECT_ROOT,
    l_freq=DEFAULT_BANDPASS_L_FREQ,
    h_freq=DEFAULT_BANDPASS_H_FREQ,
    start_sec=30.0,
    duration_sec=10.0,
    excluded_channels=DEFAULT_EXCLUDED_NON_EEG_CHANNELS,
    expected_fs=FS,
    trace_figsize=(14, 18),
    psd_figsize=(9, 5),
):
    """Load one EEG run, apply zero-phase bandpass filtering, and plot checks."""
    if l_freq is None or h_freq is None:
        raise ValueError("Both l_freq and h_freq are required for bandpass filtering.")
    if l_freq <= 0:
        raise ValueError(
            f"l_freq must be positive for high-pass filtering, got {l_freq}."
        )
    if h_freq <= l_freq:
        raise ValueError(
            f"h_freq must be greater than l_freq, got {h_freq} <= {l_freq}."
        )

    gdf_path = _resolve_gdf_path(
        subject_id=subject_id,
        session=session,
        run=run,
        task=task,
        gdf_path=gdf_path,
        project_root=project_root,
    )
    _validate_paired_training_files(gdf_path, task)
    _prepare_plot_environment()
    import mne

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    channel_labels = list(raw.ch_names)
    n_channels = len(channel_labels)
    n_samples = raw.n_times
    duration_total_sec = n_samples / fs
    nyquist = fs / 2.0

    if not np.isclose(fs, expected_fs):
        print(
            f"WARNING: expected sampling rate {expected_fs} Hz from docs, "
            f"but file reports {fs:g} Hz."
        )
    if h_freq >= nyquist:
        raise ValueError(
            f"h_freq must be below Nyquist ({nyquist:g} Hz), got {h_freq:g} Hz."
        )
    if n_channels != EXPECTED_TOTAL_CHANNELS:
        print(
            f"WARNING: docs describe {EXPECTED_TOTAL_CHANNELS} total channels "
            f"(64 EEG + 2 EOG + Status), file has {n_channels}."
        )

    start_sample = int(round(start_sec * fs))
    stop_sample = int(round((start_sec + duration_sec) * fs))
    if start_sample < 0:
        raise ValueError(f"start_sec must be non-negative, got {start_sec}.")
    if stop_sample > n_samples:
        raise ValueError(
            f"Requested {start_sec:g}-{start_sec + duration_sec:g} s, but recording "
            f"is only {duration_total_sec:.2f} s long."
        )

    eeg_labels, excluded_found, status_labels = select_analysis_eeg_channels(
        channel_labels,
        excluded_channels=excluded_channels,
    )
    raw_analysis_eeg = raw.copy().pick(eeg_labels)
    raw_data = raw_analysis_eeg.get_data()
    if not np.isfinite(raw_data).all():
        bad_count = int(np.size(raw_data) - np.isfinite(raw_data).sum())
        raise ValueError(f"Raw EEG data contains {bad_count} non-finite values.")

    print(
        "Analysis EEG data shape before filtering (samples x electrodes): "
        f"{(raw_data.shape[1], raw_data.shape[0])}"
    )
    print(f"Full recording shape (samples x channels): {(n_samples, n_channels)}")
    print(f"Sampling rate: {fs:g} Hz")
    print(f"Recording duration: {duration_total_sec:.2f} s")
    print(f"Excluded non-analysis channels: {excluded_found}")
    print(f"Status/trigger channel labels detected: {status_labels or 'none'}")
    print(
        f"Applying zero-phase bandpass filter: {l_freq:g}-{h_freq:g} Hz "
        "to analysis EEG channels only."
    )

    filtered_analysis_eeg = raw_analysis_eeg.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks="data",
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )
    filtered_data = filtered_analysis_eeg.get_data()
    if not np.isfinite(filtered_data).all():
        bad_count = int(np.size(filtered_data) - np.isfinite(filtered_data).sum())
        raise ValueError(f"Filtered EEG data contains {bad_count} non-finite values.")

    filtered_segment = filtered_data[:, start_sample:stop_sample]
    raw_segment = raw_data[:, start_sample:stop_sample]
    print(
        f"Filtered plot segment: {start_sec:g}-{start_sec + duration_sec:g} s "
        f"({start_sample}:{stop_sample} samples), shape {filtered_segment.T.shape} "
        "(samples x EEG electrodes)."
    )
    print(
        "Filtered segment median by channel, native units: "
        f"median={np.nanmedian(np.nanmedian(filtered_segment, axis=1)):.4g}, "
        f"max_abs={np.nanmax(np.abs(np.nanmedian(filtered_segment, axis=1))):.4g}"
    )

    times = np.arange(start_sample, stop_sample) / fs
    plot_segment, display_unit = _infer_display_scale(filtered_segment)
    trace_fig, trace_ax = _plot_eeg_segment(
        segment=plot_segment,
        times=times,
        eeg_labels=eeg_labels,
        title=(
            f"{gdf_path.stem}: zero-phase {l_freq:g}-{h_freq:g} Hz filtered EEG, "
            f"{duration_sec:g} s from {start_sec:g} s"
        ),
        display_unit=display_unit,
        figsize=trace_figsize,
    )
    psd = _plot_raw_filtered_psd(
        raw_data=raw_data,
        filtered_data=filtered_data,
        fs=fs,
        l_freq=l_freq,
        h_freq=h_freq,
        figsize=psd_figsize,
    )

    return {
        "raw": raw,
        "raw_analysis_eeg": raw_analysis_eeg,
        "filtered_analysis_eeg": filtered_analysis_eeg,
        "gdf_path": gdf_path,
        "eeg_labels": eeg_labels,
        "excluded_non_eeg_channels": excluded_found,
        "raw_segment_samples_by_channels": raw_segment.T,
        "filtered_segment_samples_by_channels": filtered_segment.T,
        "display_unit": display_unit,
        "sample_rate": fs,
        "l_freq": l_freq,
        "h_freq": h_freq,
        "start_sample": start_sample,
        "stop_sample": stop_sample,
        "trace_fig": trace_fig,
        "trace_ax": trace_ax,
        "psd_fig": psd["fig"],
        "psd_ax": psd["ax"],
        "psd_freqs": psd["freqs"],
        "raw_mean_psd": psd["raw_mean_psd"],
        "filtered_mean_psd": psd["filtered_mean_psd"],
    }
