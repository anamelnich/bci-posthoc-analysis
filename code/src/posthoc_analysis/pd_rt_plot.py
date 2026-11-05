"""Mean distractor RT reference lines for the behavior-filtered Pd figures."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import eeg
from .pd_filtered import TRIAL_KEYS, _validate_filtered_pd_waveforms


def plot_behavior_filtered_pd_with_mean_rt(filtered_results, output_dir=eeg.FIGURES_DIR):
    """Plot filtered Pd with color-matched dashed participant-mean RT lines.

    Reuse the exact retained distractor mask. First pool RTs within each
    participant/session, then average participants equally, matching the
    behavioral distractor-RT summary and the group-level EEG weighting.
    """
    from matplotlib.lines import Line2D

    trials = filtered_results["trials"]
    required = set(TRIAL_KEYS + ["group", "retained_for_pd", "feedback", "rt_ms",
                               "gdf_rt_ms", "rt_source", "stimulus_trigger"])
    if not isinstance(trials, pd.DataFrame) or trials.empty or required - set(trials):
        raise ValueError(f"Mean RT lines require a nonempty filtered trial table with {sorted(required)}.")
    if trials.duplicated(TRIAL_KEYS).any() or not trials.retained_for_pd.isin([True, False]).all():
        raise ValueError("Duplicate trial keys or invalid retained_for_pd mask.")
    waves = filtered_results["subject_session_pd"]
    _validate_filtered_pd_waveforms(waves, trials)
    retained = trials.loc[trials.retained_for_pd].copy()
    if (retained.empty or not retained.feedback.eq(1).all()
            or not retained.stimulus_trigger.isin([32, 44]).all()
            or not np.isfinite(retained.rt_ms).all() or not retained.rt_ms.ge(150).all()
            or not retained.rt_source.eq("gdf_status").all()
            or not np.allclose(retained.rt_ms, retained.gdf_rt_ms, rtol=0, atol=1e-8)):
        raise ValueError("Retained Pd trials must be correct distractors with finite Status-derived RT >=150 ms.")
    retained["group"] = retained.group.replace({"experimental": "bci"})
    participant_rt = (retained.groupby(["subject_id", "group", "session_id"], observed=True)
                      .agg(mean_rt_ms=("rt_ms", "mean"), n_trials=("rt_ms", "size")).reset_index())
    coverage = ["subject_id", "group", "session_id"]
    if set(map(tuple, participant_rt[coverage].to_numpy())) != set(map(tuple, waves[coverage].drop_duplicates().to_numpy())):
        raise ValueError("RT participant/session coverage does not match the plotted Pd waveforms.")
    rt_summary = (participant_rt.groupby(["group", "session_id"], observed=True)
                  .agg(mean_rt_ms=("mean_rt_ms", "mean"), n_subjects=("subject_id", "nunique"))
                  .reset_index())
    rt_summary["mean_rt_sec"] = rt_summary.mean_rt_ms / 1000
    expected = {("bci", 1), ("bci", 5), ("control", 1), ("control", 5)}
    if set(map(tuple, rt_summary[["group", "session_id"]].to_numpy())) != expected:
        raise ValueError("Mean RT references require both groups in Sessions 1 and 5.")
    if not rt_summary.mean_rt_sec.between(*eeg.DEFAULT_PD_PLOT_TIME_WINDOW).all():
        raise ValueError("A mean RT is outside the current Pd plot window; extend the plot explicitly.")
    group_pd_summary = eeg.summarize_group_pd_pre_post(waves)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures, saved_paths = {}, []
    for group in ["bci", "control"]:
        fig, ax = eeg.plot_group_pd_pre_post(group_pd_summary, group=group)
        ax.set_ylabel("Amplitude (microvolts)")
        for row in rt_summary.loc[rt_summary.group.eq(group)].itertuples(index=False):
            ax.axvline(row.mean_rt_sec, color=eeg.POSTER_PD_SESSION_COLORS[group][row.session_id],
                       linestyle="--", linewidth=1.1, alpha=.95,
                       gid=f"mean_rt_session_{row.session_id}")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.1))
        labels.append("Mean RT")
        # Keep the legend clear of the RT lines in the right-hand part of the plot.
        font = ax.get_legend().get_texts()[0].get_fontproperties()
        ax.legend(handles, labels, frameon=False, loc="upper left", handlelength=1.1,
                  handletextpad=.4, prop=font)
        for extension in ["pdf", "png"]:
            path = output_dir / f"training_pd_{group}_pre_post.{extension}"
            fig.savefig(path, dpi=300)
            saved_paths.append(path)
        figures[group] = fig
    print("Mean distractor RT lines: exact retained Pd trials; Status-derived RT; "
          "equal participant weighting. Dashed lines match each session's waveform hue.")
    print(rt_summary.to_string(index=False))
    print(f"Saved {len(saved_paths)} PDF/PNG files in {output_dir}; Pd waveforms and AUC statistics unchanged.")
    return {"participant_rt_summary": participant_rt, "rt_summary": rt_summary,
            "figures": figures, "saved_paths": saved_paths}
