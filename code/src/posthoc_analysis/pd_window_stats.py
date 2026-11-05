"""Alternative-window positive-AUC statistics on the validated filtered Pd data."""

import numpy as np

from . import eeg
from .pd_filtered import _validate_filtered_pd_waveforms


def analyze_filtered_pd_auc_window(filtered_results, time_window=(0.350, 0.450)):
    """Recompute positive AUC and existing tests without changing trial inclusion.

    Windows are in seconds. As in the original analysis, integrate the
    nonnegative part of each participant/session mean waveform at recorded
    sample times inside the window, without interpolating boundary samples.
    """
    window = np.asarray(time_window, dtype=float)
    if window.shape != (2,) or not np.isfinite(window).all() or window[0] >= window[1]:
        raise ValueError("time_window must contain two finite, increasing times in seconds.")
    waves = filtered_results["subject_session_pd"]
    _validate_filtered_pd_waveforms(waves, filtered_results["trials"])
    time = np.sort(waves.time_sec.unique())
    if window[0] < time[0] or window[1] > time[-1]:
        raise ValueError(f"Requested window {tuple(window)} exceeds recorded epoch {time[0], time[-1]}.")
    selected = time[(time >= window[0]) & (time <= window[1])]
    if len(selected) < 2:
        raise ValueError("AUC window must contain at least two recorded samples.")
    print(f"Exploratory positive Pd AUC window: {window[0]*1000:g}–{window[1]*1000:g} ms.")
    print(f"Recorded samples used: {selected[0]*1000:.4f}–{selected[-1]*1000:.4f} ms; "
          f"n={len(selected)}, 512 Hz; no boundary interpolation.")
    print("Reusing the same excluded-trial mask, pooled-trial participant/session "
          "waveforms, Group × Session ANOVA, and four planned contrasts. "
          "P values and confidence intervals are unadjusted, including across windows.")
    auc = eeg.compute_pd_positive_auc(waves, time_window=tuple(window))
    if not auc.n_samples.eq(len(selected)).all():
        raise ValueError("Participant/session AUC sample counts do not match the requested window.")
    summary = eeg.summarize_pd_positive_auc(auc)
    anova = eeg.run_pd_positive_auc_mixed_anova(auc)
    contrasts = eeg.run_pd_positive_auc_planned_contrasts(auc)
    return {"time_window_sec": tuple(window),
            "sample_window_sec": (float(selected[0]), float(selected[-1])),
            "pd_positive_auc": auc, "cell_summary": summary,
            "anova_table": anova["anova_table"], "anova_diagnostics": anova["diagnostics"],
            "planned_contrasts": contrasts}
