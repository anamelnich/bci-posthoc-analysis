"""Plots for decoder performance and posterior dynamics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_permutation_histogram(observed_accuracy: float, null_distribution: np.ndarray, output_path: str | Path) -> Path:
    """Save permutation histogram plot."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_distribution, bins=25, alpha=0.7)
    ax.axvline(observed_accuracy, color="crimson", lw=2)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Count")
    ax.set_title("Permutation null distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_online_posteriors(posteriors: np.ndarray, output_path: str | Path) -> Path:
    """Plot posterior trajectories; ``posteriors`` shape is ``(n_trials, 3)``."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(posteriors[:, 0], label="P(no distractor)")
    ax.plot(posteriors[:, 1], label="P(right)")
    ax.plot(posteriors[:, 2], label="P(left)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Posterior")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
