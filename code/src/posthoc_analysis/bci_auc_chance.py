"""Constrained label-shuffle reference for the existing online ROC-AUC plot.

This tests recorded score/label association, conditional on the saved scores.
It is NOT a routing-preserving null for EEG discrimination: the online task
selected R/L/N using the true condition before saving one posterior per trial.
See doc/bci_auc_chance.md for the source audit and interpretation limits.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .bci import (
    BCI_SESSION_EXCEPTIONS, EXPECTED_REAL_RUNS_BY_SESSION, FIGURES_DIR,
    compute_bci_subject_session_auc,
)
from .config import BCI_GROUP_SUBJECTS, CONTROL_GROUP_SUBJECTS, EXPECTED_SUBJECTS

INTERPRETATION = (
    "Conditional label-shuffle reference for recorded scores; not a null for "
    "the full EEG/decoder-routing system. True stimulus condition selected "
    "R/L/N before the single trial posterior was saved."
)


def validate_bci_auc_reference_input(df):
    """Validate trial identity, alignment, score ranges, and run randomization."""
    required = {"subject_id", "group", "session_id", "run_id", "trial_id",
                "posterior_probability", "decoding_task", "decoding_trial_index"}
    missing = required.difference(df.columns)
    if missing or df.empty:
        raise ValueError(f"AUC reference needs nonempty trial data; missing columns: {sorted(missing)}")
    data = df.copy()
    ids = ["subject_id", "group", "session_id", "run_id", "trial_id"]
    if data[ids].isna().any().any():
        raise ValueError("Missing participant/group/session/run/trial identifiers.")
    for col in ["session_id", "run_id", "trial_id"]:
        values = pd.to_numeric(data[col], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{col} must contain finite integers.")
        data[col] = values.astype(int)
    if not data.session_id.isin(EXPECTED_REAL_RUNS_BY_SESSION).all():
        raise ValueError("Only experimental sessions 1–5 are supported.")
    if (data.run_id < 1).any():
        raise ValueError("Experimental run IDs must be positive.")
    keys = ["subject_id", "session_id", "run_id", "trial_id"]
    duplicates = data.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicate trial keys:\n{data.loc[duplicates, keys].head(10)}")
    groups = {s: "experimental" for s in BCI_GROUP_SUBJECTS}
    groups.update({s: "control" for s in CONTROL_GROUP_SUBJECTS})
    bad = data.group.ne(data.subject_id.map(groups))
    if bad.any():
        raise ValueError(f"Unknown participant or group mismatch:\n{data.loc[bad, ids].head(10)}")
    for col in ["posterior_probability", "decoding_task", "decoding_trial_index"]:
        data[col] = pd.to_numeric(data[col], errors="raise")
    scores = data.posterior_probability.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Every recorded posterior must be finite and in [0, 1].")
    if not data.decoding_task.dropna().isin([0, 1]).all():
        raise ValueError("Ground-truth decoding_task must be exactly 0/1; fractional labels are invalid.")
    data = data.sort_values(keys).reset_index(drop=True)
    audits = []
    for (subject, session, run), cell in data.groupby(keys[:3], sort=True):
        context = f"{subject}, session {session}, run {run}"
        if cell.trial_id.tolist() != list(range(1, 61)):
            raise ValueError(f"{context}: expected exactly 60 unique ordered trials, 1–60.")
        absent = cell.decoding_task.isna()
        if absent.any() and not absent.all():
            raise ValueError(f"{context}: partially missing labels; cannot reproduce full-run randomization.")
        n_missing = int(absent.sum())
        longest = 0
        if not absent.all():
            if not np.array_equal(cell.decoding_trial_index, cell.trial_id):
                raise ValueError(f"{context}: posterior trial IDs do not match analysis trial indices.")
            y = cell.decoding_task.to_numpy(dtype=int)
            if y.sum() != 30:
                raise ValueError(f"{context}: expected 30 distractor and 30 no-distractor trials.")
            changes = np.r_[0, np.flatnonzero(np.diff(y)) + 1, len(y)]
            longest = int(np.diff(changes).max())
            if longest > 3:
                raise ValueError(f"{context}: {longest} consecutive identical labels; source limit is 3.")
            if "decoding_distractor_position" in cell:
                pos = pd.to_numeric(cell.decoding_distractor_position, errors="raise")
                if not pos.isin([0, 2, 4]).all() or not np.array_equal(pos.ne(0), y):
                    raise ValueError(f"{context}: distractor position and task labels disagree.")
                if (pos.eq(2).sum(), pos.eq(4).sum()) != (15, 15):
                    raise ValueError(f"{context}: expected 15 right and 15 left distractors.")
        audits.append(dict(subject_id=subject, session_id=session, run_id=run,
                           n_trials=len(cell), n_missing_labels=n_missing,
                           max_label_streak=longest))
    audit = pd.DataFrame(audits)
    for (subject, session), cell in audit.groupby(["subject_id", "session_id"]):
        expected = BCI_SESSION_EXCEPTIONS.get((subject, session), {}).get(
            "real_runs", EXPECTED_REAL_RUNS_BY_SESSION[session])
        found = cell.run_id.tolist()
        if found != list(range(1, expected + 1)):
            print(f"WARNING: {subject} session {session}: runs {found}, expected 1–{expected}; using available complete runs.")
    missing_subjects = sorted(set(EXPECTED_SUBJECTS) - set(data.subject_id))
    if missing_subjects:
        print(f"WARNING: participants absent from input: {missing_subjects}")
    sessions = data.groupby("subject_id").session_id.nunique()
    if sessions.ne(5).any():
        print(f"WARNING: incomplete session coverage: {sessions[sessions.ne(5)].to_dict()}")
    excluded = audit[audit.n_missing_labels.gt(0)].copy()
    print(f"Validated {len(data):,} rows, {len(audit)} complete runs, {data.subject_id.nunique()} participants.")
    print("Run checks: 60 trials; 30/30 classes; at most 3 consecutive labels; trial/analysis alignment.")
    if not excluded.empty:
        print("Entire runs excluded for unavailable labels (same exclusion as original AUC):")
        print(excluded.to_string(index=False))
    valid = data.dropna(subset=["decoding_task"]).reset_index(drop=True)
    if valid.empty:
        raise ValueError("No labelled trials remain for the AUC reference.")
    valid["decoding_task"] = valid.decoding_task.astype(int)
    return {"dataframe": valid, "run_audit": audit, "excluded_runs": excluded}


@lru_cache(maxsize=None)
def _sequence_count(zeros, ones, last, streak, max_streak):
    """Count legal suffixes exactly, for uniform constrained randomization."""
    if zeros < 0 or ones < 0 or streak > max_streak:
        return 0
    if zeros + ones == 0:
        return 1
    return (
        _sequence_count(zeros - 1, ones, 0, streak + 1 if last == 0 else 1, max_streak)
        + _sequence_count(zeros, ones - 1, 1, streak + 1 if last == 1 else 1, max_streak)
    )


@lru_cache(maxsize=None)
def _transition_probabilities(n_zero, n_one, max_streak):
    """Build validated probability of the next label being 1 in each state."""
    if min(n_zero, n_one) < 0 or max_streak < 1:
        raise ValueError("Counts must be nonnegative and the streak limit positive.")
    if _sequence_count(n_zero, n_one, 2, 0, max_streak) == 0:
        raise ValueError("No legal sequences exist for the requested counts/streak limit.")
    probabilities = np.zeros((n_zero + 1, n_one + 1, 3, max_streak + 1))
    for z in range(n_zero + 1):
        for o in range(n_one + 1):
            for last in range(3):
                for streak in range(max_streak + 1):
                    total = _sequence_count(z, o, last, streak, max_streak)
                    if total and z + o:
                        numerator = _sequence_count(z, o - 1, 1, streak + 1 if last == 1 else 1, max_streak)
                        probabilities[z, o, last, streak] = numerator / total
    print(f"Built uniform constrained sampler: {n_zero}/{n_one} labels, maximum streak {max_streak}.")
    return probabilities


def _sample_labels(n_zero, n_one, n_draws, rng, max_streak=3):
    """Draw uniformly from legal label orders without rejection sampling."""
    if n_draws < 1:
        raise ValueError("At least one label draw is required.")
    p = _transition_probabilities(n_zero, n_one, max_streak)
    z = np.full(n_draws, n_zero, dtype=int)
    o = np.full(n_draws, n_one, dtype=int)
    last = np.full(n_draws, 2, dtype=int)
    streak = np.zeros(n_draws, dtype=int)
    labels = np.empty((n_draws, n_zero + n_one), dtype=bool)
    for trial in range(labels.shape[1]):
        label = rng.random(n_draws) < p[z, o, last, streak]
        labels[:, trial] = label
        streak = np.where(label == last, streak + 1, 1)
        last = label.astype(int)
        o -= label
        z -= ~label
    if np.any(o) or np.any(z) or (streak > max_streak).any():
        raise RuntimeError("Constrained sampler violated label-count or streak invariants.")
    return labels


def compute_bci_auc_shuffle_reference(df, n_permutations=10000, random_seed=20261103, alpha=0.05):
    """Estimate a conditional reference, retaining the existing pooled ROC AUC.

    Labels are redrawn independently within runs, uniformly over balanced
    sequences with maximum streak three, matching Pd_utils.generate_trials.
    Scores remain fixed; this deliberately does not simulate decoder routing.
    One-sided p_label_shuffle values are artificial-reference tail probabilities,
    with Holm adjustment across all returned group/session points. Because
    routing depends on true labels, they must not be used as p-values for
    above-chance EEG discrimination, population generalization, or training.
    """
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, (int, np.integer)) or n_permutations < 100:
        raise ValueError("n_permutations must be an integer >=100 (use 10,000 for analysis).")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5.")
    if isinstance(random_seed, bool) or not isinstance(random_seed, (int, np.integer)) or random_seed < 0:
        raise ValueError("random_seed must be an explicit nonnegative integer.")
    checked = validate_bci_auc_reference_input(df)
    data = checked["dataframe"]
    observed = compute_bci_subject_session_auc(data)["subject_session_auc"].sort_values(
        ["subject_id", "session_id"]).reset_index(drop=True)
    rng = np.random.default_rng(random_seed)
    null = np.empty((n_permutations, len(observed)))
    print(INTERPRETATION)
    print(f"Drawing {n_permutations:,} legal label orders per run; seed={random_seed}; retaining all score ties and ambivalent trials.")
    for j, row in observed.iterrows():
        cell = data[(data.subject_id == row.subject_id) & (data.session_id == row.session_id)].reset_index(drop=True)
        y = cell.decoding_task.to_numpy()
        scores = cell.posterior_probability.to_numpy()
        ranks = rankdata(scores, method="average")
        n1, n0 = int(y.sum()), int((y == 0).sum())
        offset = n1 * (n1 + 1) / 2
        rank_auc = (ranks[y == 1].sum() - offset) / (n1 * n0)
        if not np.isclose(rank_auc, row.auc, atol=1e-12, rtol=0):
            raise RuntimeError(f"Rank AUC disagrees with original sklearn AUC for {row.subject_id}/{row.session_id}.")
        rank_sums = np.zeros(n_permutations)
        for _, run in cell.groupby("run_id", sort=True):
            labels = _sample_labels(30, 30, n_permutations, rng)
            rank_sums += np.sum(labels * ranks[run.index.to_numpy()], axis=1)
        null[:, j] = (rank_sums - offset) / (n1 * n0)
        if (j + 1) % 5 == 0 or j + 1 == len(observed):
            print(f"  Completed {j + 1}/{len(observed)} participant/session cells.")
    if not np.isfinite(null).all() or ((null < 0) | (null > 1)).any():
        raise RuntimeError("Invalid permutation ROC-AUC values.")
    observed["reference_mean"] = null.mean(axis=0)
    observed["reference_lower"] = np.quantile(null, alpha / 2, axis=0)
    observed["reference_upper"] = np.quantile(null, 1 - alpha / 2, axis=0)
    observed["reference_upper_one_sided"] = np.quantile(null, 1 - alpha, axis=0)
    group_rows, group_null = [], []
    for (group, session), cell in observed.groupby(["group", "session_id"], sort=True):
        draws = null[:, cell.index.to_numpy()].mean(axis=1)
        actual = float(cell.auc.mean())
        # Include ties, allowing floating point arithmetic at machine precision.
        exceed = np.count_nonzero(draws >= actual - 1e-12)
        group_rows.append(dict(
            group=group, session_id=session, n_subjects=len(cell), mean_auc=actual,
            sem_auc=float(cell.auc.std(ddof=1) / np.sqrt(len(cell))) if len(cell) > 1 else np.nan,
            reference_mean=float(draws.mean()), reference_sd=float(draws.std(ddof=1)),
            reference_lower=float(np.quantile(draws, alpha / 2)),
            reference_upper=float(np.quantile(draws, 1 - alpha / 2)),
            reference_upper_one_sided=float(np.quantile(draws, 1 - alpha)),
            auc_minus_reference_mean=actual - float(draws.mean()),
            p_label_shuffle=(exceed + 1) / (n_permutations + 1),
        ))
        group_null.append(draws)
    summary = pd.DataFrame(group_rows)
    order = np.argsort(summary.p_label_shuffle.to_numpy())
    adjusted = np.minimum(1, np.maximum.accumulate(
        summary.p_label_shuffle.to_numpy()[order] * np.arange(len(order), 0, -1)))
    summary["p_label_shuffle_holm"] = np.empty(len(summary))
    summary.loc[order, "p_label_shuffle_holm"] = adjusted
    columns = ["subject_id", "group", "session_id", "run_id", "trial_id", "posterior_probability", "decoding_task"]
    fingerprint = hashlib.sha256(data[columns].to_csv(index=False, float_format="%.17g").encode()).hexdigest()
    print("Group/session reference (equal participant weighting; Holm family = all displayed points):")
    print(summary.to_string(index=False))
    print("Intervals describe label-shuffle variation, not SEM or confidence intervals on population means.")
    return {**checked, "subject_session_auc": observed, "group_summary": summary,
            "subject_null_auc": null, "group_null_auc": np.column_stack(group_null),
            "settings": {"n_permutations": int(n_permutations), "random_seed": int(random_seed),
                         "alpha": alpha, "max_label_streak": 3, "numpy_version": np.__version__,
                         "input_sha256": fingerprint, "interpretation": INTERPRETATION,
                         "multiple_testing_family_size": len(summary)}}


def _plot_auc_with_attached_paper_layout(summary, alpha, output_dir, save):
    """Match the measured page/axes geometry of the user's paper-reference PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Measured from figures/for paper/bci_auc_across_sessions.pdf, in PDF points.
    # These intentionally reproduce the exported artwork, not its pre-crop canvas.
    page_width, page_height = 255.4946095, 228.5775
    left, bottom, right, top = 54.848513, 32.35625, 155.442112, 208.09625
    if not 0 < left < right < page_width or not 0 < bottom < top < page_height:
        raise ValueError("Paper-reference axes must lie inside the measured PDF page.")
    colors = {"experimental": "#DD8452", "control": "#4C72B0"}
    names = {"experimental": "BCI", "control": "Mental rehearsal"}
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial"],
             "font.size": 8, "axes.titlesize": 9.6, "axes.labelsize": 8,
             "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": .8,
             "legend.fontsize": 8, "legend.frameon": False,
             "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(page_width / 72, page_height / 72))
        ax = fig.add_axes([left / page_width, bottom / page_height,
                           (right - left) / page_width, (top - bottom) / page_height])
        observed_handles, observed_labels = [], []
        for i, group in enumerate(colors):
            cell = summary[summary.group.eq(group)].sort_values("session_id")
            if cell.empty:
                continue
            x = cell.session_id.to_numpy()
            ax.fill_between(x, cell.reference_lower, cell.reference_upper,
                            color=colors[group], alpha=.12, lw=0, zorder=0)
            ax.plot(x, cell.reference_mean, color=colors[group], lw=.8,
                    ls="--" if i == 0 else ":", zorder=1, label="Mean shuffled AUC")
            handle = ax.errorbar(x, cell.mean_auc, yerr=cell.sem_auc, marker="o",
                                ms=4.5, capsize=3, capthick=.8, lw=1.5,
                                color=colors[group], label=names[group], zorder=3)
            observed_handles.append(handle)
            observed_labels.append(names[group])
        low = min((summary.mean_auc - summary.sem_auc.fillna(0)).min(), .5)
        high = max((summary.mean_auc + summary.sem_auc.fillna(0)).max(), .5)
        pad = max((high - low) * .15, .03)
        ax.set(xlim=(.8, 5.2), ylim=(max(0, low - pad), min(1, high + pad)),
               xticks=[1, 2, 3, 4, 5], xlabel="Session", ylabel="AUC")
        if summary.reference_lower.min() < ax.get_ylim()[0] or summary.reference_upper.max() > ax.get_ylim()[1]:
            raise ValueError("Shuffle range exceeds original plot limits; revise layout explicitly.")
        ax.set_title("Online Classifier AUC Across Sessions (SEM)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", length=3, width=.8)
        original_legend = ax.legend(observed_handles, observed_labels, loc="center left",
                                    bbox_to_anchor=(1.02, .5), handlelength=1.8)
        ax.add_artist(original_legend)
        ax.legend([Line2D([], [], color=".45", ls="--", lw=.8),
                   Patch(facecolor=".7", alpha=.25)],
                  ["Shuffled mean", f"{100 * (1 - alpha):g}% shuffle range"],
                  loc="upper left", bbox_to_anchor=(1.02, .36),
                  fontsize=7, handlelength=1.8, title="Group colors", title_fontsize=7)
        paths = []
        if save:
            directory = Path(FIGURES_DIR if output_dir is None else output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            for suffix in ["pdf", "png"]:
                path = directory / f"bci_auc_across_sessions_with_shuffle_reference.{suffix}"
                # A tight crop changes the page size: preserve the attached PDF's
                # measured media box exactly instead, as explicitly requested.
                fig.savefig(path, dpi=300, bbox_inches=None)
                paths.append(path)
                print(f"Saved {path}")
    print(f"Matched reference PDF: {page_width:.7f} × {page_height:.7f} pt; "
          f"axes ({left}, {bottom})–({right}, {top}) pt; Arial; original title and right legend.")
    print(f"Plotted {len(summary)} observed group/session means ± SEM, actual shuffled means and central 95% ranges.")
    return {"figure": fig, "paths": paths}


def plot_bci_auc_shuffle_reference(results, output_dir=None, save=True, layout="panels"):
    """Plot original means/SEM and shuffle intervals in panels or combined axes."""
    import matplotlib.pyplot as plt

    if layout not in {"panels", "combined"}:
        raise ValueError("layout must be 'panels' or 'combined'.")
    combined = layout == "combined"
    summary = results["group_summary"]
    required = {"group", "session_id", "mean_auc", "sem_auc", "n_subjects",
                "reference_lower", "reference_upper",
                "reference_mean" if combined else "reference_upper_one_sided"}
    if summary.empty or required.difference(summary.columns):
        raise ValueError(f"Plot requires a nonempty reference summary with columns {sorted(required)}.")
    if summary.duplicated(["group", "session_id"]).any():
        raise ValueError("Plot summary has duplicate group/session points.")
    numeric = summary[list(required - {"group", "sem_auc"})].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Plot summary contains non-finite estimates or bounds.")
    if (summary.reference_lower > summary.reference_upper).any():
        raise ValueError("Reference interval bounds are reversed.")
    if combined:
        if not summary.group.isin(["experimental", "control"]).all():
            raise ValueError("Unexpected plot group labels.")
        return _plot_auc_with_attached_paper_layout(summary, results["settings"]["alpha"], output_dir, save)
    groups = [g for g in ["experimental", "control"] if g in set(summary.group)]
    if len(groups) != summary.group.nunique():
        raise ValueError("Unexpected plot group labels.")
    names = {"experimental": "BCI", "control": "Mental rehearsal"}
    colors = {"experimental": "#DD8452", "control": "#4C72B0"}
    alpha = results["settings"]["alpha"]
    coverage = 100 * (1 - alpha)
    style = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
             "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
             "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
             "xtick.direction": "out", "ytick.direction": "out", "legend.fontsize": 6.5,
             "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, len(groups),
                                 figsize=(7, 3.1),
                                 sharey=True, squeeze=False)
        for i, group in enumerate(groups):
            ax = axes[0, i]
            cell = summary[summary.group == group].sort_values("session_id")
            x = cell.session_id.to_numpy()
            ax.fill_between(x, cell.reference_lower, cell.reference_upper,
                            color=".88",
                            linewidth=0, label=f"Central {coverage:g}% label-shuffle range")
            ax.plot(x, cell.reference_upper_one_sided, color=".35", lw=.8, ls=":",
                    label=f"One-sided {coverage:g}th percentile")
            ax.axhline(.5, color=".55", ls="--", lw=.7, label="Label-independent reference (0.5)")
            ax.errorbar(x, cell.mean_auc, yerr=cell.sem_auc, color=colors[group], marker="o",
                        ms=3.5, lw=1, capsize=2, capthick=.6,
                        label="Observed mean ± SEM")
            ax.set(xlabel="Session", xticks=x)
            ax.set_title(names[group], fontsize=8)
            ax.text(-.13, 1.04, chr(97 + i), transform=ax.transAxes, fontsize=9, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            counts = sorted(cell.n_subjects.unique())
            ax.text(.98, .96, "n = " + "/".join(str(int(n)) for n in counts),
                    ha="right", va="top", transform=ax.transAxes, fontsize=7)
        axes[0, 0].set_ylabel("ROC AUC")
        low = min((summary.mean_auc - summary.sem_auc.fillna(0)).min(), summary.reference_lower.min(), .5)
        high = max((summary.mean_auc + summary.sem_auc.fillna(0)).max(), summary.reference_upper.max(), .5)
        axes[0, 0].set_ylim(max(0, low - .015), min(1, high + .02))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .075), ncol=2, frameon=False)
        fig.text(.5, .025, "Fixed-score label-shuffle reference; does not preserve condition-dependent decoder selection.",
                 ha="center", fontsize=6.5)
        fig.tight_layout(rect=[0, .24, 1, 1], w_pad=2)
        paths = []
        if save:
            directory = Path(FIGURES_DIR if output_dir is None else output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            for suffix in ["pdf", "png"]:
                stem = "bci_auc_label_shuffle_reference"
                path = directory / f"{stem}.{suffix}"
                fig.savefig(path, dpi=300, bbox_inches="tight")
                paths.append(path)
                print(f"Saved {path}")
    print(f"Plotted {len(summary)} group/session points ({layout}), original means ± SEM; pointwise reference bands, no significance stars.")
    return {"figure": fig, "paths": paths}
