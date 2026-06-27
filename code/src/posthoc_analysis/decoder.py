"""Load and plot offline decoder performance from saved decoder .mat files."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import EXPECTED_SUBJECTS, PROJECT_ROOT, get_subject_group


REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = REPO_ROOT / "figures"
DECODER_DIRNAME = "decoders"
DECODER_TYPES = ("decoderR", "decoderL", "decoderN")
SUMMARY_DECODER_TYPES = ("decoderR", "decoderL")
EXPECTED_TOP_LEVEL_FIELDS = {
    "Classes",
    "fsamp",
    "epochOnset",
    "numFeatures",
    "classify",
    "resample",
    "features",
    "spatialFilter",
    "leftElectrodeIndices",
    "rightElectrodeIndices",
    "baseline_idx",
    "performance",
    "subjectID",
}
EXPECTED_PERFORMANCE_FIELDS = {
    "tpr",
    "tnr",
    "posteriors",
    "labels",
    "file_id",
    "nTrials",
    "history",
}


def _loadmat(filepath):
    """Load one MATLAB v5 file with an informative dependency error."""
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise ModuleNotFoundError(
            "Reading decoder .mat files requires scipy.io.loadmat in the active "
            f"Python/Jupyter kernel. scipy could not be imported: {exc}"
        ) from exc

    return loadmat(filepath, squeeze_me=True, struct_as_record=False)


def _publication_style_rcparams():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 8,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _decoder_dir(root_path=None):
    root = Path(PROJECT_ROOT if root_path is None else root_path)
    return root / DECODER_DIRNAME


def _mat_fieldnames(obj):
    return set(getattr(obj, "_fieldnames", []) or [])


def _candidate_decoder_structs(mat):
    candidates = []
    for name, value in mat.items():
        if name.startswith("__"):
            continue
        fields = _mat_fieldnames(value)
        if "performance" in fields and "subjectID" in fields:
            candidates.append((name, value))
    return candidates


def _extract_decoder_struct(mat, filepath, decoder_type=None):
    candidates = _candidate_decoder_structs(mat)
    if not candidates:
        public_names = sorted(name for name in mat if not name.startswith("__"))
        raise ValueError(
            f"No decoder-like MATLAB struct found in {filepath}. "
            f"Public variables were: {public_names}."
        )

    if decoder_type is not None:
        matching = [(name, value) for name, value in candidates if name == decoder_type]
        if matching:
            return matching[0]

    if len(candidates) == 1:
        return candidates[0]

    candidate_names = [name for name, _ in candidates]
    raise ValueError(
        f"Multiple decoder-like structs found in {filepath}: {candidate_names}. "
        "Pass decoder_type to disambiguate."
    )


def _as_text(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item()).strip()
    return "".join(str(item) for item in arr.ravel()).strip()


def _as_float(value, field_name, filepath):
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError(
            f"Expected scalar performance.{field_name} in {filepath}, got shape {arr.shape}."
        )
    scalar = float(arr.reshape(-1)[0])
    if not np.isfinite(scalar):
        raise ValueError(f"performance.{field_name} is not finite in {filepath}: {scalar}.")
    return scalar


def _validate_probability(value, field_name, filepath):
    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"performance.{field_name} must be within [0, 1] in {filepath}, got {value}."
        )


def _validate_decoder_struct(decoder, struct_name, filepath, expected_subject_id=None):
    fields = _mat_fieldnames(decoder)
    missing_fields = sorted(EXPECTED_TOP_LEVEL_FIELDS - fields)
    if missing_fields:
        print(
            f"WARNING: {filepath.name} is missing documented top-level field(s): "
            f"{missing_fields}. Continuing with available performance fields."
        )

    performance = getattr(decoder, "performance", None)
    if performance is None:
        raise ValueError(f"{filepath} is missing the required performance struct.")

    performance_fields = _mat_fieldnames(performance)
    missing_performance = sorted(EXPECTED_PERFORMANCE_FIELDS - performance_fields)
    if missing_performance:
        print(
            f"WARNING: {filepath.name} performance is missing documented field(s): "
            f"{missing_performance}. Continuing if tpr/tnr are available."
        )
    for required_field in ("tpr", "tnr"):
        if required_field not in performance_fields:
            raise ValueError(f"{filepath} is missing performance.{required_field}.")

    subject_id = _as_text(getattr(decoder, "subjectID", "")).lower()
    if not re.fullmatch(r"e\d+", subject_id):
        raise ValueError(f"Could not parse subjectID from {filepath}: {subject_id!r}.")
    if expected_subject_id is not None and subject_id != expected_subject_id:
        raise ValueError(
            f"Subject mismatch in {filepath}: expected {expected_subject_id}, "
            f"decoder struct reports {subject_id}."
        )

    if struct_name not in DECODER_TYPES:
        print(
            f"WARNING: decoder struct variable in {filepath.name} is {struct_name!r}, "
            f"not one of {DECODER_TYPES}."
        )

    classes = np.asarray(getattr(decoder, "Classes", []), dtype=int).reshape(-1)
    if classes.size and classes.tolist() != [0, 1]:
        raise ValueError(
            f"{filepath} has unexpected Classes {classes.tolist()}; expected [0, 1]."
        )

    labels = np.asarray(getattr(performance, "labels", []), dtype=float).reshape(-1)
    posteriors = np.asarray(getattr(performance, "posteriors", []), dtype=float).reshape(-1)
    if labels.size != posteriors.size:
        raise ValueError(
            f"performance.labels and performance.posteriors must align in {filepath}; "
            f"got {labels.size} labels and {posteriors.size} posteriors."
        )
    if labels.size == 0:
        raise ValueError(f"performance.labels is empty in {filepath}.")
    invalid_labels = sorted(set(labels.astype(int).tolist()) - {0, 1})
    if invalid_labels:
        raise ValueError(
            f"performance.labels must contain only 0/1 classes in {filepath}; "
            f"found {invalid_labels}."
        )
    if not np.isfinite(posteriors).all():
        raise ValueError(f"performance.posteriors contains non-finite values in {filepath}.")
    if ((posteriors < 0.0) | (posteriors > 1.0)).any():
        raise ValueError(
            f"performance.posteriors must be probabilities within [0, 1] in {filepath}."
        )

    n_trials = int(np.asarray(getattr(performance, "nTrials", labels.size)).reshape(-1)[0])
    if n_trials <= 0 or n_trials > labels.size:
        raise ValueError(
            f"performance.nTrials should be positive and no larger than labels length "
            f"in {filepath}; got nTrials={n_trials}, labels={labels.size}."
        )

    tpr = _as_float(performance.tpr, "tpr", filepath)
    tnr = _as_float(performance.tnr, "tnr", filepath)
    _validate_probability(tpr, "tpr", filepath)
    _validate_probability(tnr, "tnr", filepath)

    return {
        "subject_id": subject_id,
        "decoder_type": struct_name,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_offline_accuracy": (tpr + tnr) / 2.0,
        "n_labels": int(labels.size),
        "n_trials_retained": n_trials,
        "n_positive_labels": int((labels == 1).sum()),
        "n_negative_labels": int((labels == 0).sum()),
    }


def load_decoder_performance_file(filepath, expected_subject_id=None, decoder_type=None):
    """Load one decoder `.mat` file and return validated performance metrics."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Decoder file not found: {filepath}")

    mat = _loadmat(filepath)
    struct_name, decoder = _extract_decoder_struct(mat, filepath, decoder_type=decoder_type)
    return _validate_decoder_struct(
        decoder,
        struct_name,
        filepath,
        expected_subject_id=expected_subject_id,
    )


def collect_decoder_performance(root_path=None, subjects=None, decoder_types=SUMMARY_DECODER_TYPES):
    """Collect validated decoder performance rows across subjects and decoder types."""
    decoder_types = tuple(decoder_types)
    unknown_types = sorted(set(decoder_types) - set(DECODER_TYPES))
    if unknown_types:
        raise ValueError(f"Unknown decoder_types: {unknown_types}. Expected subset of {DECODER_TYPES}.")

    decoders_path = _decoder_dir(root_path)
    if not decoders_path.exists():
        raise FileNotFoundError(f"Decoder directory not found: {decoders_path}")

    subjects = list(EXPECTED_SUBJECTS if subjects is None else subjects)
    print("=" * 80)
    print("DECODER PERFORMANCE FILE VALIDATION")
    print("=" * 80)
    print(f"Decoder directory: {decoders_path}")
    print(f"Subjects requested: {len(subjects)}")
    print(f"Decoder types requested: {decoder_types}")

    rows = []
    issues = []
    for subject_id in subjects:
        normalized_subject = str(subject_id).lower().strip()
        for decoder_type in decoder_types:
            filepath = decoders_path / f"{normalized_subject}_{decoder_type}.mat"
            if not filepath.exists():
                issues.append({
                    "subject_id": normalized_subject,
                    "decoder_type": decoder_type,
                    "file": str(filepath),
                    "issue": "Missing decoder file.",
                })
                print(f"WARNING: missing decoder file: {filepath}")
                continue
            try:
                row = load_decoder_performance_file(
                    filepath,
                    expected_subject_id=normalized_subject,
                    decoder_type=decoder_type,
                )
                row["group"] = get_subject_group(normalized_subject)
                row["file"] = str(filepath)
                rows.append(row)
            except Exception as exc:
                issues.append({
                    "subject_id": normalized_subject,
                    "decoder_type": decoder_type,
                    "file": str(filepath),
                    "issue": str(exc),
                })
                print(f"ERROR: skipping {filepath.name}: {exc}")

    performance = pd.DataFrame(rows)
    issues_df = pd.DataFrame(issues)
    if performance.empty:
        raise ValueError("No decoder performance rows could be loaded.")

    performance = performance.sort_values(["group", "subject_id", "decoder_type"]).reset_index(drop=True)
    expected_rows = len(subjects) * len(decoder_types)
    print("\nDecoder performance rows loaded:")
    print(f"{len(performance)} / {expected_rows} requested subject-decoder rows.")
    print("\nRows by group and decoder type:")
    print(
        performance.groupby(["group", "decoder_type"], observed=False)
        .size()
        .reset_index(name="n_files")
        .to_string(index=False)
    )
    if not issues_df.empty:
        print("\nDecoder file issues:")
        print(issues_df.to_string(index=False))

    return {
        "decoder_performance": performance,
        "issues": issues_df,
    }


def average_decoder_rl_by_subject(decoder_performance):
    """Average decoderR and decoderL TPR/TNR per subject and compute balanced accuracy."""
    required_columns = {"subject_id", "group", "decoder_type", "tpr", "tnr"}
    missing_columns = sorted(required_columns - set(decoder_performance.columns))
    if missing_columns:
        raise ValueError(f"Decoder performance table missing required columns: {missing_columns}.")

    data = decoder_performance.copy()
    data = data[data["decoder_type"].isin(SUMMARY_DECODER_TYPES)].copy()
    if data.empty:
        raise ValueError("No decoderR/decoderL rows are available for subject averaging.")

    counts = (
        data.groupby(["subject_id", "group"], observed=False)["decoder_type"]
        .nunique()
        .reset_index(name="n_decoder_types")
    )
    incomplete = counts[counts["n_decoder_types"] != len(SUMMARY_DECODER_TYPES)]
    if not incomplete.empty:
        raise ValueError(
            "Each subject must have decoderR and decoderL before averaging. "
            f"Incomplete subject rows:\n{incomplete.to_string(index=False)}"
        )

    subject_average = (
        data.groupby(["subject_id", "group"], observed=False)
        .agg(
            tpr=("tpr", "mean"),
            tnr=("tnr", "mean"),
            n_decoder_types=("decoder_type", "nunique"),
            mean_n_labels=("n_labels", "mean"),
            mean_n_trials_retained=("n_trials_retained", "mean"),
        )
        .reset_index()
        .sort_values(["group", "subject_id"])
    )
    subject_average["balanced_offline_accuracy"] = (
        subject_average["tpr"] + subject_average["tnr"]
    ) / 2.0

    print("\nSubject-level decoderR/decoderL averages:")
    print(
        f"{len(subject_average)} subjects; "
        f"TPR range {subject_average['tpr'].min():.3f}-{subject_average['tpr'].max():.3f}; "
        f"TNR range {subject_average['tnr'].min():.3f}-{subject_average['tnr'].max():.3f}; "
        "balanced offline accuracy range "
        f"{subject_average['balanced_offline_accuracy'].min():.3f}-"
        f"{subject_average['balanced_offline_accuracy'].max():.3f}."
    )

    return subject_average


def summarize_decoder_performance_by_group(subject_average, variability="sem"):
    """Summarize subject-averaged offline decoder performance by group."""
    if variability not in {"sem", "sd"}:
        raise ValueError("variability must be 'sem' or 'sd'.")

    metrics = ["tpr", "tnr", "balanced_offline_accuracy"]
    rows = []
    for group, group_df in subject_average.groupby("group", observed=False):
        row = {"group": group, "n_subjects": int(len(group_df)), "variability": variability}
        for metric in metrics:
            row[f"mean_{metric}"] = float(group_df[metric].mean())
            row[f"sd_{metric}"] = float(group_df[metric].std(ddof=1))
            row[f"sem_{metric}"] = row[f"sd_{metric}"] / np.sqrt(row["n_subjects"])
            row[f"error_{metric}"] = row[f"{variability}_{metric}"]
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    print("\nGroup summary of subject-averaged decoderR/decoderL performance:")
    print(summary.to_string(index=False))
    return summary


def _set_axis_padding(ax, values, pad_fraction=0.15, min_pad=0.03, lower_bound=0.0, upper_bound=1.0):
    finite_values = [float(value) for value in values if pd.notna(value) and np.isfinite(value)]
    if not finite_values:
        ax.set_ylim(lower_bound, upper_bound)
        return
    low = min(finite_values)
    high = max(finite_values)
    spread = high - low
    pad = max(spread * pad_fraction, min_pad)
    ax.set_ylim(max(lower_bound, low - pad), min(upper_bound, high + pad))


def plot_decoder_rl_performance_by_group(
    subject_average,
    group_summary,
    variability="sem",
    save=True,
    output_path=None,
):
    """Plot subject-averaged decoderR/decoderL TPR, TNR, and balanced accuracy."""
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = FIGURES_DIR / "decoder_rl_offline_performance_by_group.pdf"

    metrics = [
        ("tpr", "TPR"),
        ("tnr", "TNR"),
        ("balanced_offline_accuracy", "Balanced accuracy"),
    ]
    group_order = ["control", "bci"]
    labels = {
        "control": "Mental rehearsal",
        "bci": "BCI",
    }
    colors = {
        "control": "#4C72B0",
        "bci": "#DD8452",
    }
    x_positions = np.arange(len(metrics), dtype=float)
    offsets = {"control": -0.18, "bci": 0.18}
    jitter_offsets = np.linspace(-0.055, 0.055, max(1, subject_average["subject_id"].nunique()))
    plotted_values = [0.5]

    with plt.rc_context(_publication_style_rcparams()):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))

        for group in group_order:
            group_subjects = (
                subject_average[subject_average["group"] == group]
                .sort_values("subject_id")
                .reset_index(drop=True)
            )
            group_stats = group_summary[group_summary["group"] == group]
            if group_subjects.empty or group_stats.empty:
                continue
            group_stats = group_stats.iloc[0]

            for metric_idx, (metric, _) in enumerate(metrics):
                center = x_positions[metric_idx] + offsets[group]
                values = group_subjects[metric].to_numpy(dtype=float)
                local_jitter = jitter_offsets[: len(values)]
                ax.scatter(
                    np.full(len(values), center) + local_jitter,
                    values,
                    s=16,
                    color=colors[group],
                    alpha=0.65,
                    linewidths=0,
                    zorder=2,
                )
                mean_value = group_stats[f"mean_{metric}"]
                error_value = group_stats[f"error_{metric}"]
                plotted_values.extend(values.tolist())
                plotted_values.extend([mean_value - error_value, mean_value + error_value])
                ax.errorbar(
                    center,
                    mean_value,
                    yerr=error_value,
                    marker="o",
                    markersize=5.5,
                    capsize=3,
                    capthick=0.8,
                    linewidth=1.2,
                    color=colors[group],
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    zorder=4,
                )

        for xpos in x_positions[:-1] + 0.5:
            ax.axvline(xpos, color="#DDDDDD", linewidth=0.5, zorder=0)
        ax.axhline(0.5, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label for _, label in metrics])
        ax.set_ylabel("Offline decoder performance")
        ax.set_title(f"DecoderR/L Offline Performance ({variability.upper()})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)
        ax.tick_params(axis="both", which="both", length=3, width=0.8)
        _set_axis_padding(ax, plotted_values)
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=colors[group], label=labels[group])
            for group in group_order
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout(rect=[0, 0, 0.78, 1])

        if save:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, format="pdf", bbox_inches="tight")
            print(f"Saved figure: {output_path}")
            saved_path = output_path
        else:
            saved_path = None

    return fig, saved_path


def load_compute_and_plot_decoder_rl_performance(root_path=None, variability="sem", save=True, output_path=None):
    """Load decoderR/L .mat files, average metrics by subject, summarize, and plot."""
    collected = collect_decoder_performance(
        root_path=root_path,
        decoder_types=SUMMARY_DECODER_TYPES,
    )
    subject_average = average_decoder_rl_by_subject(collected["decoder_performance"])
    group_summary = summarize_decoder_performance_by_group(
        subject_average,
        variability=variability,
    )
    fig, figure_path = plot_decoder_rl_performance_by_group(
        subject_average,
        group_summary,
        variability=variability,
        save=save,
        output_path=output_path,
    )

    return {
        **collected,
        "subject_average": subject_average,
        "group_summary": group_summary,
        "figure": fig,
        "figure_path": str(figure_path) if figure_path is not None else None,
    }
