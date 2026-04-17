"""Feature engineering blocks: xDAWN, temporal windowing, and r² ranking."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from mne.preprocessing import Xdawn


@dataclass
class FeatureSelector:
    """Stores selected feature indices for reproducibility."""

    indices: np.ndarray
    scores: np.ndarray


def apply_xdawn(epochs: np.ndarray, labels_binary: np.ndarray, n_components: int = 4) -> np.ndarray:
    """Apply xDAWN spatial filtering.

    Parameters
    ----------
    epochs : np.ndarray
        Shape ``(n_trials, n_channels, n_times)``.
    labels_binary : np.ndarray
        Binary labels used by xDAWN covariance targets.
    n_components : int
        Number of xDAWN components.

    Returns
    -------
    np.ndarray
        Spatially filtered epochs of shape ``(n_trials, n_components, n_times)``.
    """

    if epochs.ndim != 3:
        raise ValueError(f"Expected epochs shape (trials, channels, times), got {epochs.shape}")
    info = mne.create_info(
        ch_names=[f"EEG{i+1}" for i in range(epochs.shape[1])],
        sfreq=100.0,
        ch_types="eeg",
    )
    events = np.column_stack(
        [np.arange(len(labels_binary), dtype=int), np.zeros(len(labels_binary), dtype=int), labels_binary.astype(int)]
    )
    ep = mne.EpochsArray(epochs, info=info, events=events, tmin=0.0, verbose="ERROR")
    xd = Xdawn(n_components=n_components)
    transformed = xd.fit_transform(ep)
    return transformed


def r2_feature_ranking(X: np.ndarray, y_binary: np.ndarray, top_k: int = 30) -> FeatureSelector:
    """Rank features by r² (squared point-biserial/correlation-style effect)."""

    x0 = X[y_binary == 0]
    x1 = X[y_binary == 1]
    if len(x0) == 0 or len(x1) == 0:
        raise ValueError("Both classes must be present for r² ranking")

    mu0, mu1 = x0.mean(axis=0), x1.mean(axis=0)
    var0, var1 = x0.var(axis=0), x1.var(axis=0)
    denom = var0 + var1 + 1e-12
    r2 = (mu1 - mu0) ** 2 / denom

    top_k = min(top_k, X.shape[1])
    idx = np.argsort(r2)[::-1][:top_k]
    return FeatureSelector(indices=idx, scores=r2[idx])


def zscore_from_train(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Z-score using training statistics only (decoder-compatible behavior)."""

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, {"mean": mean, "std": std}
