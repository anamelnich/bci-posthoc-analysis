"""EEG preprocessing utilities for distractor decoding.

Pipeline parity target (from project docs):
- bandpass 0.1-20 Hz
- epoch around stimulus trigger
- baseline correction
- posterior ROI + lateralized difference waves
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

# 1-indexed in documentation; convert to 0-indexed Python indices.
EEG_CHANNELS = np.arange(64)
EOG_CHANNELS = np.arange(64, 66)
STATUS_CHANNEL = 66

# Reasonable posterior ROI approximation when labels are not standardized.
DEFAULT_POSTERIOR_LABELS = [
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
]
LEFT_POSTERIOR_LABELS = ["P7", "P5", "P3", "P1", "PO7", "PO3"]
RIGHT_POSTERIOR_LABELS = ["P2", "P4", "P6", "P8", "PO4", "PO8"]


@dataclass(frozen=True)
class EpochConfig:
    """Epoching configuration.

    Times are in seconds; default window captures pre-stim baseline and the
    0.2-0.5 s discriminative period used by decoders.
    """

    sfreq_resample: float = 100.0
    l_freq: float = 0.1
    h_freq: float = 20.0
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: tuple[float, float] = (-0.2, 0.0)
    stimulus_codes: tuple[int, ...] = (8, 32, 44)


def read_gdf_array(path: str | Path) -> tuple[np.ndarray, float, list[str]]:
    """Load GDF as ndarray with shape ``(n_samples, n_channels)``.

    Returns
    -------
    data : np.ndarray
        Samples x channels matrix.
    sfreq : float
        Sampling rate from header.
    channel_names : list[str]
        Channel labels from header.
    """

    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    data = raw.get_data().T
    return data, float(raw.info["sfreq"]), list(raw.ch_names)


def make_epochs_from_gdf(path: str | Path, cfg: EpochConfig = EpochConfig()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter and epoch GDF around stimulus triggers.

    Returns
    -------
    epochs : np.ndarray
        Shape ``(n_trials, n_eeg_channels, n_times)``.
    labels : np.ndarray
        Per-trial condition labels derived from trigger codes:
        ``0=no distractor (8), 1=right (32), 2=left (44)``.
    times : np.ndarray
        Time vector in seconds with shape ``(n_times,)``.
    """

    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, verbose="ERROR")

    # Prefer annotations/events emitted by GDF reader; fallback to STATUS channel
    events, _ = mne.events_from_annotations(raw, verbose="ERROR")
    if events.size == 0:
        status = raw.get_data(picks=[STATUS_CHANNEL])[0]
        changes = np.where(np.diff(status.astype(int), prepend=0) > 0)[0]
        events = np.column_stack([changes, np.zeros_like(changes), status[changes].astype(int)])

    events = events[np.isin(events[:, 2], np.array(cfg.stimulus_codes))]
    event_id = {"no_distractor": 8, "right": 32, "left": 44}
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=cfg.baseline,
        picks=np.arange(64),
        preload=True,
        verbose="ERROR",
    )

    if cfg.sfreq_resample:
        epochs = epochs.copy().resample(cfg.sfreq_resample)

    code_to_label = {8: 0, 32: 1, 44: 2}
    labels = np.array([code_to_label[c] for c in epochs.events[:, 2]], dtype=int)
    return epochs.get_data(), labels, epochs.times


def posterior_and_lateralized_features(
    epochs: np.ndarray,
    channel_names: list[str],
    times: np.ndarray,
    window: tuple[float, float] = (0.2, 0.5),
) -> np.ndarray:
    """Extract posterior ROI and left-right difference-wave features.

    Parameters
    ----------
    epochs : np.ndarray
        Shape ``(n_trials, n_channels, n_times)``.
    channel_names : list[str]
        Length ``n_channels`` channel labels.
    times : np.ndarray
        Time axis (seconds), shape ``(n_times,)``.

    Returns
    -------
    np.ndarray
        Feature matrix with shape ``(n_trials, n_features)`` where features are
        concatenated [posterior channels, lateralized difference].
    """

    tmask = (times >= window[0]) & (times <= window[1])
    if not np.any(tmask):
        raise ValueError(f"Window {window} has no samples in provided times axis")

    ch_to_idx = {c: i for i, c in enumerate(channel_names)}
    posterior_idx = [ch_to_idx[c] for c in DEFAULT_POSTERIOR_LABELS if c in ch_to_idx]
    left_idx = [ch_to_idx[c] for c in LEFT_POSTERIOR_LABELS if c in ch_to_idx]
    right_idx = [ch_to_idx[c] for c in RIGHT_POSTERIOR_LABELS if c in ch_to_idx]

    if not posterior_idx:
        # Fallback when labels differ; use all EEG channels.
        posterior_idx = list(range(min(64, epochs.shape[1])))

    post = epochs[:, posterior_idx][:, :, tmask]
    post_flat = post.reshape(post.shape[0], -1)

    if left_idx and right_idx:
        lat = epochs[:, left_idx][:, :, tmask].mean(axis=1) - epochs[:, right_idx][:, :, tmask].mean(axis=1)
        lat_flat = lat.reshape(lat.shape[0], -1)
        return np.hstack([post_flat, lat_flat])
    return post_flat
