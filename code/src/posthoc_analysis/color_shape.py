"""Load and validate Color/Shape task data for pre/post group analyses."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from .config import PROJECT_ROOT


COLOR_SHAPE_DATA_DIR = PROJECT_ROOT / "color_shape_task"
ANALYSES_DIR = PROJECT_ROOT / "analyses"
REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = REPO_ROOT / "figures"

SUBJECT_RUNS_FILENAME = "subject_runs.csv"
SUBJECT_RUNS_FALLBACK_FILENAME = "subj_runs.csv"

SUBJECT_RUNS_REQUIRED_COLUMNS = ["subject_id", "group", "run1", "run2"]
GROUP_LABELS = {"bci", "control"}
DEFAULT_EXCLUDED_SUBJECT_REASONS = {
    "24": "incomplete Color/Shape data: post run 47.csv lacks required task columns",
    "43": "overall Color/Shape accuracy QC outlier: post run 131 >2 SD below mean",
    "46": "overall Color/Shape accuracy QC outlier: post run 136 >2 SD below mean",
    "55": "overall Color/Shape accuracy QC outlier: pre run 100 >2 SD below mean",
}
DEFAULT_EXCLUDED_SUBJECTS = set(DEFAULT_EXCLUDED_SUBJECT_REASONS)

REQUIRED_RUN_COLUMNS = [
    "chose_best",
    "left_color_val",
    "right_color_val",
    "left_shape_val",
    "right_shape_val",
    "block_type",
    "relevant_dimension",
    "color_value_difference",
    "shape_value_difference",
    "rt",
]

NONMISSING_TASK_COLUMNS = [
    "block_type",
    "chose_best",
    "rt",
    "relevant_dimension",
    "left_color_val",
    "right_color_val",
    "left_shape_val",
    "right_shape_val",
    "color_value_difference",
    "shape_value_difference",
]

NUMERIC_RUN_COLUMNS = [
    "chose_best",
    "left_color_val",
    "right_color_val",
    "left_shape_val",
    "right_shape_val",
    "block_type",
    "color_value_difference",
    "shape_value_difference",
    "rt",
]

MAIN_TASK_BLOCK_TYPE = 3


def _warn(message):
    """Emit a Color/Shape data-quality warning."""
    warnings.warn(f"Color/Shape loading: {message}", stacklevel=2)


def _format_run_number(run_number):
    """Return a run number string suitable for matching run CSV filenames."""
    if pd.isna(run_number):
        return None
    numeric = pd.to_numeric(run_number, errors="coerce")
    if pd.isna(numeric):
        return None
    if float(numeric).is_integer():
        return str(int(numeric))
    return str(run_number).strip()


def _normalize_excluded_subjects(excluded_subjects):
    """Return excluded subjects and human-readable exclusion reasons."""
    if excluded_subjects is None:
        return set(DEFAULT_EXCLUDED_SUBJECTS), dict(DEFAULT_EXCLUDED_SUBJECT_REASONS)

    if isinstance(excluded_subjects, dict):
        normalized = {
            str(subject).strip(): str(reason)
            for subject, reason in excluded_subjects.items()
        }
        return set(normalized), normalized

    normalized_subjects = {str(subject).strip() for subject in excluded_subjects}
    reasons = {
        subject: DEFAULT_EXCLUDED_SUBJECT_REASONS.get(
            subject, "excluded from Color/Shape analysis"
        )
        for subject in normalized_subjects
    }
    return normalized_subjects, reasons


def find_color_shape_subject_runs_file(data_dir=None):
    """Find the subject/run mapping CSV, preferring the documented filename.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Folder containing Color/Shape task CSV files.

    Returns
    -------
    pathlib.Path
        Path to the mapping file.
    """
    data_dir = Path(data_dir) if data_dir is not None else COLOR_SHAPE_DATA_DIR
    documented_path = data_dir / SUBJECT_RUNS_FILENAME
    fallback_path = data_dir / SUBJECT_RUNS_FALLBACK_FILENAME

    if documented_path.exists():
        return documented_path
    if fallback_path.exists():
        _warn(
            f"Documented mapping file {documented_path.name} was not found; "
            f"using {fallback_path.name}."
        )
        return fallback_path

    _warn(
        f"No subject/run mapping file found. Checked {documented_path} and "
        f"{fallback_path}."
    )
    return documented_path


def load_color_shape_subject_runs(data_dir=None, excluded_subjects=None):
    """Load and lightly validate the Color/Shape subject/run mapping table.

    Malformed and excluded rows are retained in the returned dataframe with
    validation columns so they can be inspected, but downstream run loading
    skips them.
    """
    data_dir = Path(data_dir) if data_dir is not None else COLOR_SHAPE_DATA_DIR
    excluded_subjects, exclusion_reasons = _normalize_excluded_subjects(
        excluded_subjects
    )
    mapping_path = find_color_shape_subject_runs_file(data_dir)

    if not mapping_path.exists():
        columns = SUBJECT_RUNS_REQUIRED_COLUMNS + [
            "mapping_status",
            "mapping_notes",
            "run1_label",
            "run2_label",
        ]
        return pd.DataFrame(columns=columns), mapping_path

    try:
        subject_runs = pd.read_csv(mapping_path)
    except Exception as exc:
        _warn(f"Could not read {mapping_path}: {exc}.")
        columns = SUBJECT_RUNS_REQUIRED_COLUMNS + [
            "mapping_status",
            "mapping_notes",
            "run1_label",
            "run2_label",
        ]
        return pd.DataFrame(columns=columns), mapping_path

    missing_columns = [
        col for col in SUBJECT_RUNS_REQUIRED_COLUMNS if col not in subject_runs.columns
    ]
    if missing_columns:
        _warn(
            f"{mapping_path} is missing required columns: {missing_columns}. "
            "Downstream loading will use any available required fields."
        )
        for col in missing_columns:
            subject_runs[col] = np.nan

    subject_runs = subject_runs.copy()
    subject_runs["subject_id"] = subject_runs["subject_id"].astype("string").str.strip()
    subject_runs["group"] = subject_runs["group"].astype("string").str.strip().str.lower()
    subject_runs["run1_label"] = subject_runs["run1"].apply(_format_run_number)
    subject_runs["run2_label"] = subject_runs["run2"].apply(_format_run_number)

    statuses = []
    notes = []
    for _, row in subject_runs.iterrows():
        row_notes = []
        if pd.isna(row["subject_id"]) or str(row["subject_id"]).strip() == "":
            row_notes.append("missing subject_id")
        if row["group"] not in GROUP_LABELS:
            row_notes.append(f"unexpected group={row['group']!r}")
        if row["run1_label"] is None:
            row_notes.append("missing or invalid run1")
        if row["run2_label"] is None:
            row_notes.append("missing or invalid run2")

        if row["subject_id"] in excluded_subjects:
            row_notes.append(
                exclusion_reasons.get(
                    row["subject_id"], "excluded from Color/Shape analysis"
                )
            )
            statuses.append("excluded")
            notes.append("; ".join(row_notes))
        elif row_notes:
            statuses.append("malformed")
            notes.append("; ".join(row_notes))
        else:
            statuses.append("valid")
            notes.append("")

    subject_runs["mapping_status"] = statuses
    subject_runs["mapping_notes"] = notes

    malformed_count = int((subject_runs["mapping_status"] == "malformed").sum())
    excluded_count = int((subject_runs["mapping_status"] == "excluded").sum())
    if malformed_count:
        _warn(
            f"Found {malformed_count} malformed row(s) in {mapping_path.name}; "
            "these rows will be skipped during run loading."
        )
    if excluded_count:
        _warn(
            f"Found {excluded_count} excluded subject row(s) in {mapping_path.name}: "
            f"{sorted(excluded_subjects)}."
        )

    print(f"Loaded Color/Shape subject/run mapping from: {mapping_path}")
    print(
        f"Mapping rows: {len(subject_runs)} total, "
        f"{(subject_runs['mapping_status'] == 'valid').sum()} valid, "
        f"{excluded_count} excluded."
    )
    print("Group counts among valid rows:")
    valid_groups = subject_runs.loc[
        subject_runs["mapping_status"] == "valid", "group"
    ].value_counts(dropna=False)
    print(valid_groups.to_string() if not valid_groups.empty else "No valid groups found.")

    return subject_runs, mapping_path


def _load_one_color_shape_run(csv_path, subject_id, group, session, run_number):
    """Load and clean one Color/Shape run CSV, returning data and a log row."""
    base_log = {
        "subject_id": subject_id,
        "group": group,
        "session": session,
        "run_number": run_number,
        "csv_path": str(csv_path),
        "status": "loaded",
        "n_rows_raw": np.nan,
        "n_rows_after_cleaning": 0,
        "notes": "",
    }

    if not csv_path.exists():
        message = f"Missing run CSV for subject {subject_id}, {session}: {csv_path}."
        _warn(message)
        base_log.update({"status": "missing", "notes": message})
        return None, base_log

    try:
        run_df = pd.read_csv(csv_path)
    except Exception as exc:
        message = f"Could not read {csv_path}: {exc}."
        _warn(message)
        base_log.update({"status": "malformed", "notes": message})
        return None, base_log

    base_log["n_rows_raw"] = len(run_df)

    missing_columns = [col for col in REQUIRED_RUN_COLUMNS if col not in run_df.columns]
    if missing_columns:
        message = f"{csv_path.name} missing required columns: {missing_columns}."
        _warn(message)
        base_log.update({"status": "invalid_columns", "notes": message})
        return None, base_log

    cleaned = run_df.copy()
    cleaned = cleaned.dropna(subset=NONMISSING_TASK_COLUMNS)

    for col in NUMERIC_RUN_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned["relevant_dimension"] = (
        cleaned["relevant_dimension"].astype("string").str.strip().str.lower()
    )
    cleaned = cleaned.dropna(subset=NONMISSING_TASK_COLUMNS)
    cleaned = cleaned.reset_index(drop=True)

    unexpected_dimensions = sorted(
        set(cleaned["relevant_dimension"].dropna().unique()) - {"color", "shape"}
    )
    notes = []
    if unexpected_dimensions:
        message = (
            f"{csv_path.name} contains unexpected relevant_dimension values: "
            f"{unexpected_dimensions}."
        )
        _warn(message)
        notes.append(message)

    cleaned = cleaned[cleaned["block_type"] == MAIN_TASK_BLOCK_TYPE].reset_index(drop=True)

    if cleaned.empty:
        message = f"{csv_path.name} had no usable task rows after cleaning."
        _warn(message)
        base_log.update({"status": "empty_after_cleaning", "notes": message})
        return None, base_log

    cleaned.insert(0, "run_number", run_number)
    cleaned.insert(0, "session", session)
    cleaned.insert(0, "group", group)
    cleaned.insert(0, "subject_id", subject_id)

    notes.append(f"kept block_type == {MAIN_TASK_BLOCK_TYPE} main-task rows only")
    base_log["n_rows_after_cleaning"] = len(cleaned)
    base_log["notes"] = "; ".join(notes)
    return cleaned, base_log


def load_color_shape_runs(
    data_dir=None,
    save_outputs=False,
    output_dir=None,
    excluded_subjects=None,
):
    """Load all Color/Shape pre/post run CSVs listed in the mapping file.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Folder containing the mapping file and run CSVs.
    save_outputs : bool, default False
        When True, save the loaded trial table and logs to the analyses folder.
    output_dir : str or pathlib.Path, optional
        Destination folder for CSV outputs. Defaults to PROJECT_ROOT / "analyses".
    excluded_subjects : iterable or dict, optional
        Subject IDs to exclude before run loading, or a dict mapping subject IDs
        to reasons. Defaults to the documented Color/Shape exclusions:
        subjects ``"24"``, ``"43"``, ``"46"``, and ``"55"``.

    Returns
    -------
    dict
        Keys include ``data``, ``subject_runs``, ``loading_log``, ``mapping_log``,
        ``mapping_path``, and ``output_paths``.
    """
    data_dir = Path(data_dir) if data_dir is not None else COLOR_SHAPE_DATA_DIR
    output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR

    subject_runs, mapping_path = load_color_shape_subject_runs(
        data_dir=data_dir,
        excluded_subjects=excluded_subjects,
    )

    loaded_runs = []
    loading_log_rows = []
    mapping_log = subject_runs.copy()

    for _, row in subject_runs.iterrows():
        subject_id = row.get("subject_id")
        group = row.get("group")

        if row.get("mapping_status") != "valid":
            status = row.get("mapping_status", "malformed")
            for session, run_col in [("pre", "run1_label"), ("post", "run2_label")]:
                loading_log_rows.append({
                    "subject_id": subject_id,
                    "group": group,
                    "session": session,
                    "run_number": row.get(run_col),
                    "csv_path": "",
                    "status": status,
                    "n_rows_raw": np.nan,
                    "n_rows_after_cleaning": 0,
                    "notes": row.get("mapping_notes", ""),
                })
            continue

        for session, run_col in [("pre", "run1_label"), ("post", "run2_label")]:
            run_number = row[run_col]
            csv_path = data_dir / f"{run_number}.csv"
            loaded, log_row = _load_one_color_shape_run(
                csv_path=csv_path,
                subject_id=subject_id,
                group=group,
                session=session,
                run_number=run_number,
            )
            loading_log_rows.append(log_row)
            if loaded is not None:
                loaded_runs.append(loaded)

    if loaded_runs:
        data = pd.concat(loaded_runs, ignore_index=True)
    else:
        data = pd.DataFrame(columns=["subject_id", "group", "session", "run_number"])

    loading_log = pd.DataFrame(loading_log_rows)
    if not loading_log.empty:
        loading_log = loading_log.sort_values(
            ["subject_id", "session"], na_position="last"
        ).reset_index(drop=True)

    print_color_shape_loading_summary(data, loading_log)

    output_paths = {}
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = {
            "data": output_dir / "all_subjects_color_shape_loaded_trials.csv",
            "loading_log": output_dir / "color_shape_loading_log.csv",
            "mapping_log": output_dir / "color_shape_subject_runs_mapping_log.csv",
        }
        data.to_csv(output_paths["data"], index=False)
        loading_log.to_csv(output_paths["loading_log"], index=False)
        mapping_log.to_csv(output_paths["mapping_log"], index=False)
        print("Saved Color/Shape loading outputs:")
        for label, path in output_paths.items():
            print(f"  {label}: {path}")

    return {
        "data": data,
        "subject_runs": subject_runs,
        "mapping_log": mapping_log,
        "loading_log": loading_log,
        "mapping_path": mapping_path,
        "output_paths": output_paths,
    }


def print_color_shape_loading_summary(data, loading_log):
    """Print concise validation summaries for loaded Color/Shape runs."""
    if loading_log.empty:
        print("No Color/Shape loading log rows were generated.")
        return

    status_counts = loading_log["status"].value_counts(dropna=False)
    print("Color/Shape run loading status counts:")
    print(status_counts.to_string())

    loaded_log = loading_log[loading_log["status"] == "loaded"]
    print(f"Loaded files: {len(loaded_log)}")
    print(f"Missing files: {(loading_log['status'] == 'missing').sum()}")
    print(f"Invalid-column files: {(loading_log['status'] == 'invalid_columns').sum()}")

    if not loaded_log.empty:
        print("Cleaned trial count distribution across loaded runs:")
        print(loaded_log["n_rows_after_cleaning"].describe().to_string())

    if data.empty:
        _warn("No Color/Shape task rows were loaded.")
        return

    print(f"Loaded task rows: {len(data):,}")
    print(f"Subjects with loaded data: {data['subject_id'].nunique()}")
    print("Rows by group/session:")
    print(
        data.groupby(["group", "session"], dropna=False)
        .size()
        .rename("n_rows")
        .to_string()
    )


def add_color_shape_trial_definitions(data):
    """Add congruency and value-difference columns to loaded main-task rows.

    Parameters
    ----------
    data : pandas.DataFrame
        Loaded Color/Shape trial table from :func:`load_color_shape_runs`.

    Returns
    -------
    pandas.DataFrame
        Copy of ``data`` with ``congruent_ix``, ``incongruent_ix``,
        ``congruency``, ``relevant_val_diff``, and ``irrelevant_val_diff``.
    """
    required_columns = [
        "subject_id",
        "group",
        "session",
        "run_number",
        "block_type",
        "left_color_val",
        "right_color_val",
        "left_shape_val",
        "right_shape_val",
        "relevant_dimension",
        "color_value_difference",
        "shape_value_difference",
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "Cannot compute Color/Shape trial definitions because required "
            f"columns are missing: {missing_columns}."
        )

    annotated = data.copy()
    non_main_rows = annotated["block_type"] != MAIN_TASK_BLOCK_TYPE
    if non_main_rows.any():
        _warn(
            f"Found {int(non_main_rows.sum())} non-main-task row(s) while "
            "computing trial definitions. They will remain in the table but "
            "are marked as not incongruent."
        )

    left_better_both = (
        (annotated["left_color_val"] > annotated["right_color_val"])
        & (annotated["left_shape_val"] > annotated["right_shape_val"])
    )
    right_better_both = (
        (annotated["right_color_val"] > annotated["left_color_val"])
        & (annotated["right_shape_val"] > annotated["left_shape_val"])
    )
    annotated["congruent_ix"] = left_better_both | right_better_both
    annotated["incongruent_ix"] = (
        (annotated["block_type"] == MAIN_TASK_BLOCK_TYPE)
        & ~annotated["congruent_ix"]
    )
    annotated["congruency"] = np.where(
        annotated["congruent_ix"], "congruent", "incongruent"
    )

    annotated["relevant_val_diff"] = np.nan
    annotated["irrelevant_val_diff"] = np.nan

    color_rows = annotated["relevant_dimension"] == "color"
    shape_rows = annotated["relevant_dimension"] == "shape"
    unexpected_rows = ~(color_rows | shape_rows)
    if unexpected_rows.any():
        unexpected_values = sorted(
            annotated.loc[unexpected_rows, "relevant_dimension"].dropna().unique()
        )
        _warn(
            "Found rows with relevant_dimension other than 'color' or 'shape': "
            f"{unexpected_values}. Relevant/irrelevant differences set to NaN "
            "for affected rows."
        )

    annotated.loc[color_rows, "relevant_val_diff"] = annotated.loc[
        color_rows, "color_value_difference"
    ]
    annotated.loc[color_rows, "irrelevant_val_diff"] = annotated.loc[
        color_rows, "shape_value_difference"
    ]
    annotated.loc[shape_rows, "relevant_val_diff"] = annotated.loc[
        shape_rows, "shape_value_difference"
    ]
    annotated.loc[shape_rows, "irrelevant_val_diff"] = annotated.loc[
        shape_rows, "color_value_difference"
    ]

    print_color_shape_trial_definition_summary(annotated)
    return annotated


def print_color_shape_trial_definition_summary(data):
    """Print sanity checks for Color/Shape trial-definition columns."""
    print("Color/Shape trial definition summary:")
    print(f"Rows: {len(data):,}")

    if "block_type" in data.columns:
        print("Block type counts:")
        print(data["block_type"].value_counts(dropna=False).sort_index().to_string())

    if "congruency" in data.columns:
        print("Congruency counts:")
        print(data["congruency"].value_counts(dropna=False).to_string())
        print("Congruency counts by group/session:")
        print(
            data.groupby(["group", "session", "congruency"], dropna=False)
            .size()
            .rename("n_rows")
            .to_string()
        )

    for column in ["relevant_val_diff", "irrelevant_val_diff"]:
        if column in data.columns:
            missing = int(data[column].isna().sum())
            print(f"{column} missing values: {missing}")
            print(f"{column} levels:")
            print(data[column].value_counts(dropna=False).sort_index().to_string())


def load_and_annotate_color_shape_trials(
    data_dir=None,
    save_outputs=False,
    output_dir=None,
    excluded_subjects=None,
):
    """Load main-task Color/Shape rows and compute trial-definition columns."""
    results = load_color_shape_runs(
        data_dir=data_dir,
        save_outputs=save_outputs,
        output_dir=output_dir,
        excluded_subjects=excluded_subjects,
    )
    annotated = add_color_shape_trial_definitions(results["data"])
    results["annotated_data"] = annotated

    if save_outputs:
        output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = output_dir / "all_subjects_color_shape_main_task_annotated.csv"
        annotated.to_csv(annotated_path, index=False)
        results["output_paths"]["annotated_data"] = annotated_path
        print(f"Saved annotated Color/Shape main-task table: {annotated_path}")

    return results


def compute_color_shape_accuracy_summary(data):
    """Compute #13 Color/Shape accuracy outcomes by subject/session.

    Accuracy is the mean of ``chose_best``. Congruent accuracy uses congruent
    main-task trials. Incongruent accuracy uses incongruent main-task trials
    with ``irrelevant_val_diff > 0``, matching the analysis specification.
    """
    required_columns = [
        "subject_id",
        "group",
        "session",
        "run_number",
        "chose_best",
        "congruent_ix",
        "incongruent_ix",
        "irrelevant_val_diff",
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "Cannot compute Color/Shape accuracy because required columns are "
            f"missing: {missing_columns}. Run add_color_shape_trial_definitions first."
        )

    rows = []
    grouped = data.groupby(["subject_id", "group", "session", "run_number"], dropna=False)
    for (subject_id, group, session, run_number), subset in grouped:
        congruent = subset[subset["congruent_ix"]]
        incongruent = subset[
            subset["incongruent_ix"] & (subset["irrelevant_val_diff"] > 0)
        ]

        if congruent.empty:
            _warn(
                f"No congruent trials for subject {subject_id}, session {session}, "
                f"run {run_number}; acc_congruent set to NaN."
            )
        if incongruent.empty:
            _warn(
                f"No incongruent trials with irrelevant_val_diff > 0 for subject "
                f"{subject_id}, session {session}, run {run_number}; "
                "acc_incongruent set to NaN."
            )

        rows.append({
            "subject_id": subject_id,
            "group": group,
            "session": session,
            "run_number": run_number,
            "n_trials_total": len(subset),
            "n_congruent_trials": len(congruent),
            "n_incongruent_trials_irrelevant_gt0": len(incongruent),
            "acc_congruent": congruent["chose_best"].mean() if not congruent.empty else np.nan,
            "acc_incongruent": (
                incongruent["chose_best"].mean() if not incongruent.empty else np.nan
            ),
        })

    accuracy = pd.DataFrame(rows)
    if not accuracy.empty:
        accuracy = accuracy.sort_values(["subject_id", "session"]).reset_index(drop=True)

    print_color_shape_accuracy_summary(accuracy)
    return accuracy


def print_color_shape_accuracy_summary(accuracy):
    """Print sanity checks for the #13 Color/Shape accuracy table."""
    if accuracy.empty:
        _warn("Color/Shape accuracy summary is empty.")
        return

    print("Color/Shape accuracy summary:")
    print(f"Rows: {len(accuracy)}")
    print(f"Subjects: {accuracy['subject_id'].nunique()}")
    print("Rows by group/session:")
    print(
        accuracy.groupby(["group", "session"], dropna=False)
        .size()
        .rename("n_subject_sessions")
        .to_string()
    )
    print("Trial counts per subject/session:")
    print(
        accuracy[
            [
                "n_trials_total",
                "n_congruent_trials",
                "n_incongruent_trials_irrelevant_gt0",
            ]
        ].describe().to_string()
    )
    print("Accuracy distributions:")
    print(accuracy[["acc_congruent", "acc_incongruent"]].describe().to_string())

    missing = accuracy[["acc_congruent", "acc_incongruent"]].isna().sum()
    if missing.any():
        _warn(f"Missing accuracy values: {missing.to_dict()}.")


def load_annotate_and_compute_color_shape_accuracy(
    data_dir=None,
    save_outputs=False,
    output_dir=None,
    excluded_subjects=None,
):
    """Load, annotate, and compute the #13 Color/Shape accuracy summary."""
    results = load_and_annotate_color_shape_trials(
        data_dir=data_dir,
        save_outputs=save_outputs,
        output_dir=output_dir,
        excluded_subjects=excluded_subjects,
    )
    accuracy = compute_color_shape_accuracy_summary(results["annotated_data"])
    results["accuracy_summary"] = accuracy

    if save_outputs:
        output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        accuracy_path = output_dir / "color_shape_accuracy_summary.csv"
        accuracy.to_csv(accuracy_path, index=False)
        results["output_paths"]["accuracy_summary"] = accuracy_path
        print(f"Saved Color/Shape accuracy summary: {accuracy_path}")

    return results


def compute_color_shape_subject_session_overall_accuracy(data):
    """Compute overall mean accuracy for each subject/session."""
    required_columns = ["subject_id", "group", "session", "run_number", "chose_best"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "Cannot compute overall Color/Shape accuracy QC because required "
            f"columns are missing: {missing_columns}."
        )

    summary = (
        data.groupby(["subject_id", "group", "session", "run_number"], dropna=False)
        .agg(
            n_trials=("chose_best", "size"),
            n_nonmissing_accuracy=("chose_best", "count"),
            mean_accuracy=("chose_best", "mean"),
        )
        .reset_index()
        .sort_values(["subject_id", "session"])
        .reset_index(drop=True)
    )

    mean_acc = summary["mean_accuracy"].mean()
    sd_acc = summary["mean_accuracy"].std(ddof=1)
    if pd.isna(sd_acc) or sd_acc == 0:
        summary["accuracy_z"] = np.nan
        summary["accuracy_outlier_2sd"] = False
        _warn("Could not compute useful accuracy z-scores because SD is zero or missing.")
    else:
        summary["accuracy_z"] = (summary["mean_accuracy"] - mean_acc) / sd_acc
        summary["accuracy_outlier_2sd"] = summary["accuracy_z"].abs() > 2

    print("Color/Shape overall accuracy QC:")
    print(f"Rows: {len(summary)}")
    print(f"Mean subject/session accuracy: {mean_acc:.4f}")
    print(f"SD subject/session accuracy: {sd_acc:.4f}")
    print(f"Potential >2 SD outliers: {int(summary['accuracy_outlier_2sd'].sum())}")
    if summary["accuracy_outlier_2sd"].any():
        print(
            summary.loc[
                summary["accuracy_outlier_2sd"],
                ["subject_id", "group", "session", "run_number", "mean_accuracy", "accuracy_z"],
            ].to_string(index=False)
        )

    return summary


def plot_color_shape_overall_accuracy_histogram(
    accuracy_qc,
    output_dir=None,
    filename="color_shape_subject_session_overall_accuracy_histogram.pdf",
):
    """Plot subject/session overall accuracy histogram with SD reference lines."""
    required_columns = ["mean_accuracy"]
    missing_columns = [col for col in required_columns if col not in accuracy_qc.columns]
    if missing_columns:
        raise ValueError(
            "Cannot plot Color/Shape accuracy QC histogram because required "
            f"columns are missing: {missing_columns}."
        )

    values = accuracy_qc["mean_accuracy"].dropna()
    if values.empty:
        raise ValueError("Cannot plot Color/Shape accuracy QC histogram: no values.")

    mean_acc = values.mean()
    sd_acc = values.std(ddof=1)
    output_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(
        values,
        bins=np.linspace(0.4, 1.0, 13),
        color="#8fb3c9",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axvline(mean_acc, color="#222222", linewidth=1.5, label="Mean")
    for sd_multiplier, linestyle in [(1, "--"), (2, ":")]:
        lower = mean_acc - sd_multiplier * sd_acc
        upper = mean_acc + sd_multiplier * sd_acc
        ax.axvline(
            lower,
            color="#666666",
            linewidth=1.1,
            linestyle=linestyle,
            label=f"±{sd_multiplier} SD" if sd_multiplier == 1 else f"±{sd_multiplier} SD",
        )
        ax.axvline(upper, color="#666666", linewidth=1.1, linestyle=linestyle)

    ax.set_xlabel("Mean accuracy")
    ax.set_ylabel("Subject-session count")
    ax.set_title("Color/Shape Overall Accuracy QC")
    ax.set_xlim(0.4, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")

    print(f"Saved Color/Shape overall accuracy QC histogram: {output_path}")
    return fig, ax, output_path


def load_annotate_and_plot_color_shape_accuracy_qc(
    data_dir=None,
    save_outputs=True,
    output_dir=None,
    figure_dir=None,
    excluded_subjects=None,
):
    """Load annotated trials, compute overall accuracy QC, and plot histogram."""
    results = load_and_annotate_color_shape_trials(
        data_dir=data_dir,
        save_outputs=save_outputs,
        output_dir=output_dir,
        excluded_subjects=excluded_subjects,
    )
    accuracy_qc = compute_color_shape_subject_session_overall_accuracy(
        results["annotated_data"]
    )
    results["overall_accuracy_qc"] = accuracy_qc

    if save_outputs:
        output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        qc_path = output_dir / "color_shape_overall_accuracy_qc.csv"
        accuracy_qc.to_csv(qc_path, index=False)
        results["output_paths"]["overall_accuracy_qc"] = qc_path
        print(f"Saved Color/Shape overall accuracy QC table: {qc_path}")

    fig, ax, figure_path = plot_color_shape_overall_accuracy_histogram(
        accuracy_qc,
        output_dir=figure_dir,
    )
    results["overall_accuracy_qc_figure"] = fig
    results["overall_accuracy_qc_axis"] = ax
    results["output_paths"]["overall_accuracy_qc_figure"] = figure_path
    return results


def compute_color_shape_subject_session_reaction_time_qc(data):
    """Compute mean reaction time for each retained subject/session."""
    required_columns = ["subject_id", "group", "session", "run_number", "rt"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "Cannot compute Color/Shape RT QC because required columns are "
            f"missing: {missing_columns}."
        )

    rt_data = data.copy()
    rt_data["rt"] = pd.to_numeric(rt_data["rt"], errors="coerce")
    missing_rt = int(rt_data["rt"].isna().sum())
    if missing_rt:
        _warn(f"Found {missing_rt} missing/non-numeric RT values; ignored for RT QC.")

    summary = (
        rt_data.groupby(["subject_id", "group", "session", "run_number"], dropna=False)
        .agg(
            n_trials=("rt", "size"),
            n_nonmissing_rt=("rt", "count"),
            mean_rt_ms=("rt", "mean"),
            median_rt_ms=("rt", "median"),
            sd_rt_ms=("rt", "std"),
        )
        .reset_index()
        .sort_values(["subject_id", "session"])
        .reset_index(drop=True)
    )

    mean_rt = summary["mean_rt_ms"].mean()
    sd_rt = summary["mean_rt_ms"].std(ddof=1)
    if pd.isna(sd_rt) or sd_rt == 0:
        summary["mean_rt_z"] = np.nan
        summary["mean_rt_outlier_2sd"] = False
        _warn("Could not compute useful RT z-scores because SD is zero or missing.")
    else:
        summary["mean_rt_z"] = (summary["mean_rt_ms"] - mean_rt) / sd_rt
        summary["mean_rt_outlier_2sd"] = summary["mean_rt_z"].abs() > 2

    print("Color/Shape reaction time QC:")
    print(f"Rows: {len(summary)}")
    print(f"Mean subject/session RT: {mean_rt:.2f} ms")
    print(f"SD subject/session RT: {sd_rt:.2f} ms")
    print(f"Potential >2 SD RT outliers: {int(summary['mean_rt_outlier_2sd'].sum())}")
    if summary["mean_rt_outlier_2sd"].any():
        print(
            summary.loc[
                summary["mean_rt_outlier_2sd"],
                [
                    "subject_id",
                    "group",
                    "session",
                    "run_number",
                    "mean_rt_ms",
                    "mean_rt_z",
                ],
            ].to_string(index=False)
        )

    return summary


def plot_color_shape_reaction_time_qc_histogram(
    rt_qc,
    output_dir=None,
    filename="color_shape_subject_session_mean_rt_histogram.pdf",
    bin_width_ms=50,
):
    """Plot retained subject/session mean RT histogram with SD reference lines."""
    required_columns = ["mean_rt_ms"]
    missing_columns = [col for col in required_columns if col not in rt_qc.columns]
    if missing_columns:
        raise ValueError(
            "Cannot plot Color/Shape RT QC histogram because required columns "
            f"are missing: {missing_columns}."
        )

    values = rt_qc["mean_rt_ms"].dropna()
    if values.empty:
        raise ValueError("Cannot plot Color/Shape RT QC histogram: no RT values.")

    mean_rt = values.mean()
    sd_rt = values.std(ddof=1)
    bin_start = np.floor(values.min() / bin_width_ms) * bin_width_ms
    bin_stop = np.ceil(values.max() / bin_width_ms) * bin_width_ms + bin_width_ms
    bins = np.arange(bin_start, bin_stop + bin_width_ms, bin_width_ms)

    output_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(
        values,
        bins=bins,
        color="#b8a07e",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axvline(mean_rt, color="#222222", linewidth=1.5, label="Mean")
    for sd_multiplier, linestyle in [(1, "--"), (2, ":")]:
        lower = mean_rt - sd_multiplier * sd_rt
        upper = mean_rt + sd_multiplier * sd_rt
        ax.axvline(
            lower,
            color="#666666",
            linewidth=1.1,
            linestyle=linestyle,
            label=f"±{sd_multiplier} SD",
        )
        ax.axvline(upper, color="#666666", linewidth=1.1, linestyle=linestyle)

    ax.set_xlabel("Mean RT (ms)")
    ax.set_ylabel("Subject-session count")
    ax.set_title("Color/Shape Mean RT QC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")

    print(f"Saved Color/Shape RT QC histogram: {output_path}")
    return fig, ax, output_path


def load_annotate_and_plot_color_shape_reaction_time_qc(
    data_dir=None,
    save_outputs=True,
    output_dir=None,
    figure_dir=None,
    excluded_subjects=None,
    bin_width_ms=50,
):
    """Load annotated trials, compute retained-subject RT QC, and plot histogram."""
    results = load_and_annotate_color_shape_trials(
        data_dir=data_dir,
        save_outputs=save_outputs,
        output_dir=output_dir,
        excluded_subjects=excluded_subjects,
    )
    rt_qc = compute_color_shape_subject_session_reaction_time_qc(
        results["annotated_data"]
    )
    results["reaction_time_qc"] = rt_qc

    if save_outputs:
        output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        rt_qc_path = output_dir / "color_shape_reaction_time_qc.csv"
        rt_qc.to_csv(rt_qc_path, index=False)
        results["output_paths"]["reaction_time_qc"] = rt_qc_path
        print(f"Saved Color/Shape RT QC table: {rt_qc_path}")

    fig, ax, figure_path = plot_color_shape_reaction_time_qc_histogram(
        rt_qc,
        output_dir=figure_dir,
        bin_width_ms=bin_width_ms,
    )
    results["reaction_time_qc_figure"] = fig
    results["reaction_time_qc_axis"] = ax
    results["output_paths"]["reaction_time_qc_figure"] = figure_path
    return results


def remove_color_shape_rt_outliers_within_subject_session(data, sd_threshold=3):
    """Remove RT outlier trials using subject/session mean ± SD threshold.

    Parameters
    ----------
    data : pandas.DataFrame
        Annotated Color/Shape trial table.
    sd_threshold : float, default 3
        Number of within-subject/session SDs used for the lower and upper
        exclusion bounds.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Cleaned trial table and a subject/session removal summary table.
    """
    required_columns = ["subject_id", "group", "session", "run_number", "rt"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "Cannot remove Color/Shape RT outliers because required columns are "
            f"missing: {missing_columns}."
        )

    if sd_threshold <= 0:
        raise ValueError("sd_threshold must be positive.")

    working = data.copy()
    working["rt"] = pd.to_numeric(working["rt"], errors="coerce")
    working["_rt_outlier"] = False
    working["_rt_missing"] = working["rt"].isna()

    summary_rows = []
    group_cols = ["subject_id", "group", "session", "run_number"]
    for keys, subset in working.groupby(group_cols, dropna=False):
        subject_id, group, session, run_number = keys
        rt_values = subset["rt"].dropna()
        n_trials = len(subset)
        n_missing_rt = int(subset["rt"].isna().sum())
        mean_rt = rt_values.mean() if not rt_values.empty else np.nan
        sd_rt = rt_values.std(ddof=1) if len(rt_values) > 1 else np.nan

        if pd.isna(sd_rt) or sd_rt == 0:
            lower_bound = np.nan
            upper_bound = np.nan
            outlier_index = []
            if rt_values.empty:
                _warn(
                    f"No non-missing RT values for subject {subject_id}, "
                    f"session {session}, run {run_number}."
                )
        else:
            lower_bound = mean_rt - sd_threshold * sd_rt
            upper_bound = mean_rt + sd_threshold * sd_rt
            outlier_mask = (subset["rt"] < lower_bound) | (subset["rt"] > upper_bound)
            outlier_index = subset.index[outlier_mask].tolist()
            working.loc[outlier_index, "_rt_outlier"] = True

        n_removed = len(outlier_index)
        summary_rows.append({
            "subject_id": subject_id,
            "group": group,
            "session": session,
            "run_number": run_number,
            "n_trials": n_trials,
            "n_nonmissing_rt": int(rt_values.size),
            "n_missing_rt": n_missing_rt,
            "rt_mean_before_ms": mean_rt,
            "rt_sd_before_ms": sd_rt,
            "rt_lower_bound_ms": lower_bound,
            "rt_upper_bound_ms": upper_bound,
            "n_rt_outlier_trials_removed": n_removed,
            "pct_rt_outlier_trials_removed": n_removed / n_trials if n_trials else np.nan,
        })

    cleaned = working.loc[~working["_rt_outlier"]].copy()
    cleaned = cleaned.drop(columns=["_rt_outlier", "_rt_missing"])
    removal_summary = pd.DataFrame(summary_rows)
    if not removal_summary.empty:
        removal_summary = removal_summary.sort_values(
            ["subject_id", "session"]
        ).reset_index(drop=True)

    print("Color/Shape within-subject/session RT outlier removal:")
    print(f"Rows before cleaning: {len(data):,}")
    print(f"Rows after cleaning: {len(cleaned):,}")
    print(
        "Total RT outlier trials removed: "
        f"{int(removal_summary['n_rt_outlier_trials_removed'].sum())}"
    )
    print("Removal count distribution:")
    print(removal_summary["n_rt_outlier_trials_removed"].describe().to_string())
    removed = removal_summary[
        removal_summary["n_rt_outlier_trials_removed"] > 0
    ].sort_values("n_rt_outlier_trials_removed", ascending=False)
    if not removed.empty:
        print("Subject/session pairs with removed RT trials:")
        print(
            removed[
                [
                    "subject_id",
                    "group",
                    "session",
                    "run_number",
                    "n_rt_outlier_trials_removed",
                    "pct_rt_outlier_trials_removed",
                    "rt_mean_before_ms",
                    "rt_sd_before_ms",
                    "rt_upper_bound_ms",
                ]
            ].to_string(index=False)
        )

    return cleaned, removal_summary


def load_annotate_and_remove_color_shape_rt_outliers(
    data_dir=None,
    save_outputs=True,
    output_dir=None,
    excluded_subjects=None,
    sd_threshold=3,
):
    """Load annotated trials and remove RT outliers within subject/session."""
    results = load_and_annotate_color_shape_trials(
        data_dir=data_dir,
        save_outputs=save_outputs,
        output_dir=output_dir,
        excluded_subjects=excluded_subjects,
    )
    cleaned, removal_summary = remove_color_shape_rt_outliers_within_subject_session(
        results["annotated_data"],
        sd_threshold=sd_threshold,
    )
    results["rt_cleaned_data"] = cleaned
    results["rt_outlier_removal_summary"] = removal_summary

    if save_outputs:
        output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = output_dir / "all_subjects_color_shape_main_task_rt_cleaned.csv"
        summary_path = output_dir / "color_shape_rt_outlier_removal_summary.csv"
        cleaned.to_csv(cleaned_path, index=False)
        removal_summary.to_csv(summary_path, index=False)
        results["output_paths"]["rt_cleaned_data"] = cleaned_path
        results["output_paths"]["rt_outlier_removal_summary"] = summary_path
        print(f"Saved Color/Shape RT-cleaned trial table: {cleaned_path}")
        print(f"Saved Color/Shape RT outlier removal summary: {summary_path}")

    return results


def load_clean_and_plot_color_shape_reaction_time_distribution(
    data_dir=None,
    save_outputs=True,
    output_dir=None,
    figure_dir=None,
    excluded_subjects=None,
    sd_threshold=3,
    bin_width_ms=50,
):
    """Remove RT outlier trials, then plot cleaned subject/session mean RTs."""
    results = load_annotate_and_remove_color_shape_rt_outliers(
        data_dir=data_dir,
        save_outputs=save_outputs,
        output_dir=output_dir,
        excluded_subjects=excluded_subjects,
        sd_threshold=sd_threshold,
    )
    cleaned_rt_qc = compute_color_shape_subject_session_reaction_time_qc(
        results["rt_cleaned_data"]
    )
    results["cleaned_reaction_time_qc"] = cleaned_rt_qc

    if save_outputs:
        output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        cleaned_rt_qc_path = output_dir / "color_shape_reaction_time_qc_after_outlier_removal.csv"
        cleaned_rt_qc.to_csv(cleaned_rt_qc_path, index=False)
        results["output_paths"]["cleaned_reaction_time_qc"] = cleaned_rt_qc_path
        print(f"Saved cleaned Color/Shape RT QC table: {cleaned_rt_qc_path}")

    fig, ax, figure_path = plot_color_shape_reaction_time_qc_histogram(
        cleaned_rt_qc,
        output_dir=figure_dir,
        filename="color_shape_subject_session_mean_rt_after_outlier_removal_histogram.pdf",
        bin_width_ms=bin_width_ms,
    )
    results["cleaned_reaction_time_qc_figure"] = fig
    results["cleaned_reaction_time_qc_axis"] = ax
    results["output_paths"]["cleaned_reaction_time_qc_figure"] = figure_path
    return results


def remove_color_shape_rt_outliers_mad_within_subject_session(
    data,
    mad_threshold=3,
):
    """Remove RT outlier trials using within-subject/session median ± MAD.

    Outliers are trials where ``abs(rt - median_rt) > mad_threshold * MAD``.
    This function starts from the supplied trial table and does not depend on
    the earlier SD-based cleaning path.
    """
    required_columns = ["subject_id", "group", "session", "run_number", "rt"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "Cannot remove Color/Shape MAD RT outliers because required columns "
            f"are missing: {missing_columns}."
        )

    if mad_threshold <= 0:
        raise ValueError("mad_threshold must be positive.")

    working = data.copy()
    working["rt"] = pd.to_numeric(working["rt"], errors="coerce")
    working["_rt_mad_outlier"] = False

    summary_rows = []
    group_cols = ["subject_id", "group", "session", "run_number"]
    for keys, subset in working.groupby(group_cols, dropna=False):
        subject_id, group, session, run_number = keys
        rt_values = subset["rt"].dropna()
        n_trials = len(subset)
        n_missing_rt = int(subset["rt"].isna().sum())

        median_rt = rt_values.median() if not rt_values.empty else np.nan
        mad_rt = (
            (rt_values - median_rt).abs().median()
            if not rt_values.empty
            else np.nan
        )

        if pd.isna(mad_rt) or mad_rt == 0:
            lower_bound = np.nan
            upper_bound = np.nan
            outlier_index = []
            if rt_values.empty:
                _warn(
                    f"No non-missing RT values for subject {subject_id}, "
                    f"session {session}, run {run_number}."
                )
            else:
                _warn(
                    f"MAD is zero/missing for subject {subject_id}, session "
                    f"{session}, run {run_number}; no MAD RT trials removed."
                )
        else:
            lower_bound = median_rt - mad_threshold * mad_rt
            upper_bound = median_rt + mad_threshold * mad_rt
            outlier_mask = (subset["rt"] < lower_bound) | (subset["rt"] > upper_bound)
            outlier_index = subset.index[outlier_mask].tolist()
            working.loc[outlier_index, "_rt_mad_outlier"] = True

        n_removed = len(outlier_index)
        summary_rows.append({
            "subject_id": subject_id,
            "group": group,
            "session": session,
            "run_number": run_number,
            "n_trials": n_trials,
            "n_nonmissing_rt": int(rt_values.size),
            "n_missing_rt": n_missing_rt,
            "rt_median_before_ms": median_rt,
            "rt_mad_before_ms": mad_rt,
            "rt_mad_lower_bound_ms": lower_bound,
            "rt_mad_upper_bound_ms": upper_bound,
            "n_rt_mad_outlier_trials_removed": n_removed,
            "pct_rt_mad_outlier_trials_removed": (
                n_removed / n_trials if n_trials else np.nan
            ),
        })

    cleaned = working.loc[~working["_rt_mad_outlier"]].copy()
    cleaned = cleaned.drop(columns=["_rt_mad_outlier"])
    removal_summary = pd.DataFrame(summary_rows)
    if not removal_summary.empty:
        removal_summary = removal_summary.sort_values(
            ["subject_id", "session"]
        ).reset_index(drop=True)

    print("Color/Shape within-subject/session MAD RT outlier removal:")
    print(f"Rows before cleaning: {len(data):,}")
    print(f"Rows after cleaning: {len(cleaned):,}")
    print(
        "Total MAD RT outlier trials removed: "
        f"{int(removal_summary['n_rt_mad_outlier_trials_removed'].sum())}"
    )
    print("Percent removed distribution:")
    print(
        removal_summary["pct_rt_mad_outlier_trials_removed"]
        .mul(100)
        .describe()
        .to_string()
    )
    removed = removal_summary[
        removal_summary["n_rt_mad_outlier_trials_removed"] > 0
    ].sort_values("pct_rt_mad_outlier_trials_removed", ascending=False)
    if not removed.empty:
        print("Subject/session pairs with MAD RT trials removed:")
        print(
            removed[
                [
                    "subject_id",
                    "group",
                    "session",
                    "run_number",
                    "n_rt_mad_outlier_trials_removed",
                    "pct_rt_mad_outlier_trials_removed",
                    "rt_median_before_ms",
                    "rt_mad_before_ms",
                    "rt_mad_upper_bound_ms",
                ]
            ].to_string(index=False)
        )

    return cleaned, removal_summary


def load_annotate_and_remove_color_shape_rt_outliers_mad(
    data_dir=None,
    save_outputs=True,
    output_dir=None,
    excluded_subjects=None,
    mad_threshold=3,
):
    """Load annotated trials and remove RT outliers by within-pair MAD."""
    results = load_and_annotate_color_shape_trials(
        data_dir=data_dir,
        save_outputs=save_outputs,
        output_dir=output_dir,
        excluded_subjects=excluded_subjects,
    )
    cleaned, removal_summary = (
        remove_color_shape_rt_outliers_mad_within_subject_session(
            results["annotated_data"],
            mad_threshold=mad_threshold,
        )
    )
    results["rt_mad_cleaned_data"] = cleaned
    results["rt_mad_outlier_removal_summary"] = removal_summary

    if save_outputs:
        output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = output_dir / "all_subjects_color_shape_main_task_rt_mad_cleaned.csv"
        summary_path = output_dir / "color_shape_rt_mad_outlier_removal_summary.csv"
        cleaned.to_csv(cleaned_path, index=False)
        removal_summary.to_csv(summary_path, index=False)
        results["output_paths"]["rt_mad_cleaned_data"] = cleaned_path
        results["output_paths"]["rt_mad_outlier_removal_summary"] = summary_path
        print(f"Saved Color/Shape MAD RT-cleaned trial table: {cleaned_path}")
        print(f"Saved Color/Shape MAD RT outlier removal summary: {summary_path}")

    return results


def load_mad_clean_and_plot_color_shape_reaction_time_distribution(
    data_dir=None,
    save_outputs=True,
    output_dir=None,
    figure_dir=None,
    excluded_subjects=None,
    mad_threshold=3,
    bin_width_ms=200,
):
    """Remove MAD RT outlier trials, then plot subject/session mean RTs."""
    results = load_annotate_and_remove_color_shape_rt_outliers_mad(
        data_dir=data_dir,
        save_outputs=save_outputs,
        output_dir=output_dir,
        excluded_subjects=excluded_subjects,
        mad_threshold=mad_threshold,
    )
    mad_cleaned_rt_qc = compute_color_shape_subject_session_reaction_time_qc(
        results["rt_mad_cleaned_data"]
    )
    results["mad_cleaned_reaction_time_qc"] = mad_cleaned_rt_qc

    if save_outputs:
        output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        qc_path = output_dir / "color_shape_reaction_time_qc_after_mad_outlier_removal.csv"
        mad_cleaned_rt_qc.to_csv(qc_path, index=False)
        results["output_paths"]["mad_cleaned_reaction_time_qc"] = qc_path
        print(f"Saved MAD-cleaned Color/Shape RT QC table: {qc_path}")

    fig, ax, figure_path = plot_color_shape_reaction_time_qc_histogram(
        mad_cleaned_rt_qc,
        output_dir=figure_dir,
        filename="color_shape_subject_session_mean_rt_after_mad_outlier_removal_histogram.pdf",
        bin_width_ms=bin_width_ms,
    )
    results["mad_cleaned_reaction_time_qc_figure"] = fig
    results["mad_cleaned_reaction_time_qc_axis"] = ax
    results["output_paths"]["mad_cleaned_reaction_time_qc_figure"] = figure_path
    return results


def clean_color_shape_rt_correct_trials_mad(
    data,
    min_rt_ms=150,
    mad_threshold=3,
):
    """Clean Color/Shape RT trials using the planned correct-trial pipeline.

    Pipeline:
    1. Start from annotated main-task trials.
    2. Retain only correct response trials (``chose_best == 1``) for RT work.
    3. Remove RT values below ``min_rt_ms``.
    4. Within each subject/session, remove RT outliers where
       ``abs(rt - median_rt) > mad_threshold * MAD``.

    Returns cleaned correct-trial rows plus separate low-RT, MAD, and overall
    removal summaries at the subject/session level.
    """
    required_columns = [
        "subject_id",
        "group",
        "session",
        "run_number",
        "chose_best",
        "rt",
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "Cannot run Color/Shape planned RT cleaning because required columns "
            f"are missing: {missing_columns}."
        )

    if min_rt_ms < 0:
        raise ValueError("min_rt_ms must be non-negative.")
    if mad_threshold <= 0:
        raise ValueError("mad_threshold must be positive.")

    working = data.copy()
    working["rt"] = pd.to_numeric(working["rt"], errors="coerce")
    working["chose_best"] = pd.to_numeric(working["chose_best"], errors="coerce")
    group_cols = ["subject_id", "group", "session", "run_number"]

    overall_rows = []
    low_rt_rows = []
    mad_rows = []
    cleaned_chunks = []

    for keys, subset in working.groupby(group_cols, dropna=False):
        subject_id, group, session, run_number = keys
        n_total_trials = len(subset)
        correct = subset[subset["chose_best"] == 1].copy()
        n_correct_trials = len(correct)
        n_incorrect_trials_removed = n_total_trials - n_correct_trials
        n_missing_rt_correct = int(correct["rt"].isna().sum())

        low_rt_mask = correct["rt"] < min_rt_ms
        n_low_rt_removed = int(low_rt_mask.sum())
        after_low_rt = correct.loc[~low_rt_mask & correct["rt"].notna()].copy()

        low_rt_rows.append({
            "subject_id": subject_id,
            "group": group,
            "session": session,
            "run_number": run_number,
            "n_correct_trials_before_low_rt": n_correct_trials,
            "n_missing_rt_correct": n_missing_rt_correct,
            "min_rt_ms": min_rt_ms,
            "n_low_rt_trials_removed": n_low_rt_removed,
            "pct_low_rt_trials_removed_of_correct": (
                n_low_rt_removed / n_correct_trials if n_correct_trials else np.nan
            ),
            "pct_low_rt_trials_removed_of_all": (
                n_low_rt_removed / n_total_trials if n_total_trials else np.nan
            ),
        })

        rt_values = after_low_rt["rt"].dropna()
        median_rt = rt_values.median() if not rt_values.empty else np.nan
        mad_rt = (
            (rt_values - median_rt).abs().median()
            if not rt_values.empty
            else np.nan
        )

        if pd.isna(mad_rt) or mad_rt == 0:
            lower_bound = np.nan
            upper_bound = np.nan
            mad_outlier_mask = pd.Series(False, index=after_low_rt.index)
            if rt_values.empty:
                _warn(
                    f"No non-missing correct RT values after low-RT removal for "
                    f"subject {subject_id}, session {session}, run {run_number}."
                )
            else:
                _warn(
                    f"MAD is zero/missing after low-RT removal for subject "
                    f"{subject_id}, session {session}, run {run_number}; "
                    "no MAD RT trials removed."
                )
        else:
            lower_bound = median_rt - mad_threshold * mad_rt
            upper_bound = median_rt + mad_threshold * mad_rt
            mad_outlier_mask = (
                (after_low_rt["rt"] < lower_bound)
                | (after_low_rt["rt"] > upper_bound)
            )

        n_mad_removed = int(mad_outlier_mask.sum())
        cleaned = after_low_rt.loc[~mad_outlier_mask].copy()
        cleaned_chunks.append(cleaned)

        mad_rows.append({
            "subject_id": subject_id,
            "group": group,
            "session": session,
            "run_number": run_number,
            "n_trials_before_mad": len(after_low_rt),
            "rt_median_before_mad_ms": median_rt,
            "rt_mad_before_mad_ms": mad_rt,
            "rt_mad_lower_bound_ms": lower_bound,
            "rt_mad_upper_bound_ms": upper_bound,
            "mad_threshold": mad_threshold,
            "n_mad_rt_trials_removed": n_mad_removed,
            "pct_mad_rt_trials_removed_of_pre_mad": (
                n_mad_removed / len(after_low_rt) if len(after_low_rt) else np.nan
            ),
            "pct_mad_rt_trials_removed_of_all": (
                n_mad_removed / n_total_trials if n_total_trials else np.nan
            ),
        })

        n_total_removed = (
            n_incorrect_trials_removed
            + n_missing_rt_correct
            + n_low_rt_removed
            + n_mad_removed
        )
        overall_rows.append({
            "subject_id": subject_id,
            "group": group,
            "session": session,
            "run_number": run_number,
            "n_total_trials": n_total_trials,
            "n_correct_trials": n_correct_trials,
            "n_incorrect_trials_removed": n_incorrect_trials_removed,
            "pct_incorrect_trials_removed": (
                n_incorrect_trials_removed / n_total_trials if n_total_trials else np.nan
            ),
            "n_missing_rt_correct_removed": n_missing_rt_correct,
            "pct_missing_rt_correct_removed": (
                n_missing_rt_correct / n_total_trials if n_total_trials else np.nan
            ),
            "n_low_rt_trials_removed": n_low_rt_removed,
            "pct_low_rt_trials_removed": (
                n_low_rt_removed / n_total_trials if n_total_trials else np.nan
            ),
            "n_mad_rt_trials_removed": n_mad_removed,
            "pct_mad_rt_trials_removed": (
                n_mad_removed / n_total_trials if n_total_trials else np.nan
            ),
            "n_total_trials_removed": n_total_removed,
            "pct_total_trials_removed": (
                n_total_removed / n_total_trials if n_total_trials else np.nan
            ),
            "n_trials_retained_for_rt": len(cleaned),
            "pct_trials_retained_for_rt": (
                len(cleaned) / n_total_trials if n_total_trials else np.nan
            ),
        })

    cleaned_data = (
        pd.concat(cleaned_chunks, ignore_index=True)
        if cleaned_chunks
        else pd.DataFrame(columns=data.columns)
    )
    low_rt_summary = pd.DataFrame(low_rt_rows)
    mad_summary = pd.DataFrame(mad_rows)
    overall_summary = pd.DataFrame(overall_rows)

    for table in [low_rt_summary, mad_summary, overall_summary]:
        if not table.empty:
            table.sort_values(["subject_id", "session"], inplace=True)
            table.reset_index(drop=True, inplace=True)

    print("Color/Shape planned RT cleaning summary:")
    print(f"Rows before RT cleaning: {len(data):,}")
    print(f"Rows retained for RT: {len(cleaned_data):,}")
    print(
        "Total removed for incorrect responses: "
        f"{int(overall_summary['n_incorrect_trials_removed'].sum())}"
    )
    print(
        f"Total removed for RT < {min_rt_ms} ms: "
        f"{int(overall_summary['n_low_rt_trials_removed'].sum())}"
    )
    print(
        f"Total removed for > {mad_threshold} MAD RT outliers: "
        f"{int(overall_summary['n_mad_rt_trials_removed'].sum())}"
    )
    print(
        "Total removed overall: "
        f"{int(overall_summary['n_total_trials_removed'].sum())}"
    )
    print("Overall percent removed distribution:")
    print(overall_summary["pct_total_trials_removed"].mul(100).describe().to_string())

    return cleaned_data, low_rt_summary, mad_summary, overall_summary


def load_clean_and_plot_color_shape_planned_rt_pipeline(
    data_dir=None,
    save_outputs=True,
    output_dir=None,
    figure_dir=None,
    excluded_subjects=None,
    min_rt_ms=150,
    mad_threshold=3,
    bin_width_ms=150,
):
    """Run planned correct-trial RT cleaning and plot subject/session mean RT."""
    results = load_and_annotate_color_shape_trials(
        data_dir=data_dir,
        save_outputs=save_outputs,
        output_dir=output_dir,
        excluded_subjects=excluded_subjects,
    )
    cleaned, low_rt_summary, mad_summary, overall_summary = (
        clean_color_shape_rt_correct_trials_mad(
            results["annotated_data"],
            min_rt_ms=min_rt_ms,
            mad_threshold=mad_threshold,
        )
    )
    results["planned_rt_cleaned_data"] = cleaned
    results["planned_rt_low_rt_summary"] = low_rt_summary
    results["planned_rt_mad_summary"] = mad_summary
    results["planned_rt_overall_removal_summary"] = overall_summary

    planned_rt_qc = compute_color_shape_subject_session_reaction_time_qc(cleaned)
    results["planned_rt_qc"] = planned_rt_qc

    if save_outputs:
        output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "planned_rt_cleaned_data": output_dir
            / "all_subjects_color_shape_main_task_correct_rt_mad_cleaned.csv",
            "planned_rt_low_rt_summary": output_dir
            / "color_shape_planned_rt_low_rt_removal_summary.csv",
            "planned_rt_mad_summary": output_dir
            / "color_shape_planned_rt_mad_removal_summary.csv",
            "planned_rt_overall_removal_summary": output_dir
            / "color_shape_planned_rt_overall_removal_summary.csv",
            "planned_rt_qc": output_dir
            / "color_shape_reaction_time_qc_planned_cleaning.csv",
        }
        cleaned.to_csv(paths["planned_rt_cleaned_data"], index=False)
        low_rt_summary.to_csv(paths["planned_rt_low_rt_summary"], index=False)
        mad_summary.to_csv(paths["planned_rt_mad_summary"], index=False)
        overall_summary.to_csv(
            paths["planned_rt_overall_removal_summary"], index=False
        )
        planned_rt_qc.to_csv(paths["planned_rt_qc"], index=False)
        results["output_paths"].update(paths)
        print("Saved planned Color/Shape RT cleaning outputs:")
        for label, path in paths.items():
            print(f"  {label}: {path}")

    fig, ax, figure_path = plot_color_shape_reaction_time_qc_histogram(
        planned_rt_qc,
        output_dir=figure_dir,
        filename="color_shape_subject_session_mean_rt_planned_cleaning_histogram.pdf",
        bin_width_ms=bin_width_ms,
    )
    results["planned_rt_qc_figure"] = fig
    results["planned_rt_qc_axis"] = ax
    results["output_paths"]["planned_rt_qc_figure"] = figure_path
    return results


def compute_color_shape_reaction_time_summary(rt_cleaned_data):
    """Compute #14 Color/Shape RT outcomes by subject/session.

    The input should be the planned RT-cleaned correct-trial table produced by
    :func:`clean_color_shape_rt_correct_trials_mad`.
    """
    required_columns = [
        "subject_id",
        "group",
        "session",
        "run_number",
        "rt",
        "congruent_ix",
        "incongruent_ix",
        "irrelevant_val_diff",
    ]
    missing_columns = [col for col in required_columns if col not in rt_cleaned_data.columns]
    if missing_columns:
        raise ValueError(
            "Cannot compute Color/Shape RT summary because required columns are "
            f"missing: {missing_columns}."
        )

    rows = []
    group_cols = ["subject_id", "group", "session", "run_number"]
    for (subject_id, group, session, run_number), subset in rt_cleaned_data.groupby(
        group_cols, dropna=False
    ):
        congruent = subset[subset["congruent_ix"]]
        incongruent = subset[
            subset["incongruent_ix"] & (subset["irrelevant_val_diff"] > 0)
        ]

        if congruent.empty:
            _warn(
                f"No RT-cleaned congruent correct trials for subject {subject_id}, "
                f"session {session}, run {run_number}; RT set to NaN."
            )
        if incongruent.empty:
            _warn(
                f"No RT-cleaned incongruent correct trials with "
                f"irrelevant_val_diff > 0 for subject {subject_id}, session "
                f"{session}, run {run_number}; RT set to NaN."
            )

        rows.append({
            "subject_id": subject_id,
            "group": group,
            "session": session,
            "run_number": run_number,
            "n_rt_cleaned_correct_trials": len(subset),
            "n_rt_congruent_correct_trials": len(congruent),
            "n_rt_incongruent_correct_trials": len(incongruent),
            "rt_congruent_correct": (
                congruent["rt"].mean() if not congruent.empty else np.nan
            ),
            "rt_incongruent_correct": (
                incongruent["rt"].mean() if not incongruent.empty else np.nan
            ),
        })

    rt_summary = pd.DataFrame(rows)
    if not rt_summary.empty:
        rt_summary = rt_summary.sort_values(["subject_id", "session"]).reset_index(drop=True)

    print_color_shape_reaction_time_summary(rt_summary)
    return rt_summary


def print_color_shape_reaction_time_summary(rt_summary):
    """Print sanity checks for #14 Color/Shape RT summary."""
    if rt_summary.empty:
        _warn("Color/Shape RT summary is empty.")
        return

    print("Color/Shape RT summary:")
    print(f"Rows: {len(rt_summary)}")
    print(f"Subjects: {rt_summary['subject_id'].nunique()}")
    print("Rows by group/session:")
    print(
        rt_summary.groupby(["group", "session"], dropna=False)
        .size()
        .rename("n_subject_sessions")
        .to_string()
    )
    print("RT-cleaned trial counts per subject/session:")
    print(
        rt_summary[
            [
                "n_rt_cleaned_correct_trials",
                "n_rt_congruent_correct_trials",
                "n_rt_incongruent_correct_trials",
            ]
        ].describe().to_string()
    )
    print("RT distributions:")
    print(
        rt_summary[
            ["rt_congruent_correct", "rt_incongruent_correct"]
        ].describe().to_string()
    )


def create_color_shape_analysis_log(
    annotated_data,
    accuracy_summary,
    rt_summary,
    rt_overall_removal_summary=None,
):
    """Create #15 Color/Shape analysis log by subject/session."""
    required_annotated = [
        "subject_id",
        "group",
        "session",
        "run_number",
        "congruent_ix",
        "incongruent_ix",
        "irrelevant_val_diff",
    ]
    missing_annotated = [col for col in required_annotated if col not in annotated_data.columns]
    if missing_annotated:
        raise ValueError(
            "Cannot create Color/Shape analysis log because annotated data are "
            f"missing columns: {missing_annotated}."
        )

    count_rows = []
    group_cols = ["subject_id", "group", "session", "run_number"]
    for keys, subset in annotated_data.groupby(group_cols, dropna=False):
        subject_id, group, session, run_number = keys
        count_rows.append({
            "subject_id": subject_id,
            "group": group,
            "session": session,
            "run_number": run_number,
            "n_valid_trials_used": len(subset),
            "n_main_task_trials_used": int((subset["block_type"] == MAIN_TASK_BLOCK_TYPE).sum())
            if "block_type" in subset.columns
            else len(subset),
            "n_congruent_trials": int(subset["congruent_ix"].sum()),
            "n_incongruent_trials": int(
                (subset["incongruent_ix"] & (subset["irrelevant_val_diff"] > 0)).sum()
            ),
        })
    counts = pd.DataFrame(count_rows)

    analysis_log = counts.merge(
        accuracy_summary[
            [
                "subject_id",
                "group",
                "session",
                "run_number",
                "acc_congruent",
                "acc_incongruent",
            ]
        ],
        on=["subject_id", "group", "session", "run_number"],
        how="left",
        validate="one_to_one",
    )
    analysis_log = analysis_log.merge(
        rt_summary[
            [
                "subject_id",
                "group",
                "session",
                "run_number",
                "rt_congruent_correct",
                "rt_incongruent_correct",
                "n_rt_cleaned_correct_trials",
                "n_rt_congruent_correct_trials",
                "n_rt_incongruent_correct_trials",
            ]
        ],
        on=["subject_id", "group", "session", "run_number"],
        how="left",
        validate="one_to_one",
    )

    if rt_overall_removal_summary is not None:
        removal_cols = [
            "subject_id",
            "group",
            "session",
            "run_number",
            "n_incorrect_trials_removed",
            "n_low_rt_trials_removed",
            "n_mad_rt_trials_removed",
            "n_total_trials_removed",
            "pct_total_trials_removed",
            "n_trials_retained_for_rt",
        ]
        missing_removal = [
            col for col in removal_cols if col not in rt_overall_removal_summary.columns
        ]
        if missing_removal:
            _warn(
                "RT removal summary was provided but is missing columns: "
                f"{missing_removal}. Removal columns will not be merged."
            )
        else:
            analysis_log = analysis_log.merge(
                rt_overall_removal_summary[removal_cols],
                on=["subject_id", "group", "session", "run_number"],
                how="left",
                validate="one_to_one",
            )

    analysis_log = analysis_log.sort_values(["subject_id", "session"]).reset_index(drop=True)
    print_color_shape_analysis_log_summary(analysis_log)
    return analysis_log


def print_color_shape_analysis_log_summary(analysis_log):
    """Print sanity checks for #15 Color/Shape analysis log."""
    if analysis_log.empty:
        _warn("Color/Shape analysis log is empty.")
        return

    print("Color/Shape analysis log:")
    print(f"Rows: {len(analysis_log)}")
    print(f"Subjects: {analysis_log['subject_id'].nunique()}")
    print("Rows by group/session:")
    print(
        analysis_log.groupby(["group", "session"], dropna=False)
        .size()
        .rename("n_subject_sessions")
        .to_string()
    )
    missing = analysis_log[
        [
            "acc_congruent",
            "acc_incongruent",
            "rt_congruent_correct",
            "rt_incongruent_correct",
        ]
    ].isna().sum()
    print("Missing outcome counts:")
    print(missing.to_string())


def load_compute_color_shape_rt_and_analysis_log(
    data_dir=None,
    save_outputs=True,
    output_dir=None,
    figure_dir=None,
    excluded_subjects=None,
    min_rt_ms=150,
    mad_threshold=3,
    bin_width_ms=150,
):
    """Run #13, #14, and #15 Color/Shape summaries with planned RT cleaning."""
    results = load_annotate_and_compute_color_shape_accuracy(
        data_dir=data_dir,
        save_outputs=save_outputs,
        output_dir=output_dir,
        excluded_subjects=excluded_subjects,
    )

    cleaned, low_rt_summary, mad_summary, overall_summary = (
        clean_color_shape_rt_correct_trials_mad(
            results["annotated_data"],
            min_rt_ms=min_rt_ms,
            mad_threshold=mad_threshold,
        )
    )
    results["planned_rt_cleaned_data"] = cleaned
    results["planned_rt_low_rt_summary"] = low_rt_summary
    results["planned_rt_mad_summary"] = mad_summary
    results["planned_rt_overall_removal_summary"] = overall_summary

    rt_summary = compute_color_shape_reaction_time_summary(cleaned)
    analysis_log = create_color_shape_analysis_log(
        annotated_data=results["annotated_data"],
        accuracy_summary=results["accuracy_summary"],
        rt_summary=rt_summary,
        rt_overall_removal_summary=overall_summary,
    )
    results["reaction_time_summary"] = rt_summary
    results["analysis_log"] = analysis_log

    planned_rt_qc = compute_color_shape_subject_session_reaction_time_qc(cleaned)
    results["planned_rt_qc"] = planned_rt_qc

    if save_outputs:
        output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "planned_rt_cleaned_data": output_dir
            / "all_subjects_color_shape_main_task_correct_rt_mad_cleaned.csv",
            "planned_rt_low_rt_summary": output_dir
            / "color_shape_planned_rt_low_rt_removal_summary.csv",
            "planned_rt_mad_summary": output_dir
            / "color_shape_planned_rt_mad_removal_summary.csv",
            "planned_rt_overall_removal_summary": output_dir
            / "color_shape_planned_rt_overall_removal_summary.csv",
            "reaction_time_summary": output_dir
            / "color_shape_reaction_time_summary.csv",
            "analysis_log": output_dir / "color_shape_analysis_log.csv",
            "planned_rt_qc": output_dir
            / "color_shape_reaction_time_qc_planned_cleaning.csv",
        }
        cleaned.to_csv(paths["planned_rt_cleaned_data"], index=False)
        low_rt_summary.to_csv(paths["planned_rt_low_rt_summary"], index=False)
        mad_summary.to_csv(paths["planned_rt_mad_summary"], index=False)
        overall_summary.to_csv(
            paths["planned_rt_overall_removal_summary"], index=False
        )
        rt_summary.to_csv(paths["reaction_time_summary"], index=False)
        analysis_log.to_csv(paths["analysis_log"], index=False)
        planned_rt_qc.to_csv(paths["planned_rt_qc"], index=False)
        results["output_paths"].update(paths)
        print("Saved Color/Shape RT and analysis log outputs:")
        for label, path in paths.items():
            print(f"  {label}: {path}")

    fig, ax, figure_path = plot_color_shape_reaction_time_qc_histogram(
        planned_rt_qc,
        output_dir=figure_dir,
        filename="color_shape_subject_session_mean_rt_planned_cleaning_histogram.pdf",
        bin_width_ms=bin_width_ms,
    )
    results["planned_rt_qc_figure"] = fig
    results["planned_rt_qc_axis"] = ax
    results["output_paths"]["planned_rt_qc_figure"] = figure_path
    return results


def create_color_shape_summary_long(analysis_log):
    """Create subject/session/congruency long table for #20 plotting."""
    required_columns = [
        "subject_id",
        "group",
        "session",
        "run_number",
        "acc_congruent",
        "acc_incongruent",
        "rt_congruent_correct",
        "rt_incongruent_correct",
    ]
    missing_columns = [col for col in required_columns if col not in analysis_log.columns]
    if missing_columns:
        raise ValueError(
            "Cannot create Color/Shape long summary because analysis log is "
            f"missing columns: {missing_columns}."
        )

    rows = []
    for _, row in analysis_log.iterrows():
        rows.append({
            "subject_id": row["subject_id"],
            "group": row["group"],
            "session": row["session"],
            "run_number": row["run_number"],
            "congruency": "congruent",
            "accuracy": row["acc_congruent"],
            "rt_correct": row["rt_congruent_correct"],
        })
        rows.append({
            "subject_id": row["subject_id"],
            "group": row["group"],
            "session": row["session"],
            "run_number": row["run_number"],
            "congruency": "incongruent",
            "accuracy": row["acc_incongruent"],
            "rt_correct": row["rt_incongruent_correct"],
        })

    summary_long = pd.DataFrame(rows)
    if not summary_long.empty:
        summary_long = summary_long.sort_values(
            ["subject_id", "session", "congruency"]
        ).reset_index(drop=True)

    print("Color/Shape long summary:")
    print(f"Rows: {len(summary_long)}")
    print(
        summary_long.groupby(["group", "session", "congruency"], dropna=False)
        .size()
        .rename("n_subject_values")
        .to_string()
    )
    return summary_long


def summarize_color_shape_group_level(summary_long):
    """Compute group/session/congruency means, SDs, SEMs, and n."""
    required_columns = ["group", "session", "congruency", "accuracy", "rt_correct"]
    missing_columns = [col for col in required_columns if col not in summary_long.columns]
    if missing_columns:
        raise ValueError(
            "Cannot summarize Color/Shape group level because long summary is "
            f"missing columns: {missing_columns}."
        )

    rows = []
    for keys, subset in summary_long.groupby(["group", "session", "congruency"], dropna=False):
        group, session, congruency = keys
        acc = subset["accuracy"].dropna()
        rt = subset["rt_correct"].dropna()
        rows.append({
            "group": group,
            "session": session,
            "congruency": congruency,
            "accuracy_mean": acc.mean() if len(acc) else np.nan,
            "accuracy_sd": acc.std(ddof=1) if len(acc) > 1 else np.nan,
            "accuracy_sem": acc.std(ddof=1) / np.sqrt(len(acc)) if len(acc) > 1 else np.nan,
            "rt_mean": rt.mean() if len(rt) else np.nan,
            "rt_sd": rt.std(ddof=1) if len(rt) > 1 else np.nan,
            "rt_sem": rt.std(ddof=1) / np.sqrt(len(rt)) if len(rt) > 1 else np.nan,
            "n": int(max(len(acc), len(rt))),
        })

    group_summary = pd.DataFrame(rows)
    if not group_summary.empty:
        group_summary = group_summary.sort_values(
            ["group", "session", "congruency"]
        ).reset_index(drop=True)
    print("Color/Shape group-level summary:")
    print(group_summary.to_string(index=False))
    return group_summary


def bootstrap_mean_ci(values, n_boot=5000, ci=95, random_state=7):
    """Bootstrap percentile CI for a subject-level mean."""
    values = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(values) < 2:
        _warn(
            f"Need at least 2 non-missing values for bootstrap CI; got {len(values)}."
        )
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    alpha = (100 - ci) / 2
    return tuple(np.percentile(means, [alpha, 100 - alpha]))


def _color_shape_plot_panel(ax, summary_long, metric, congruency, ylabel, title, n_boot, random_state):
    """Plot one pre/post panel for one metric and congruency."""
    colors = {"bci": "#C44E52", "control": "#4C72B0"}
    labels = {"bci": "BCI", "control": "Control"}
    x_positions = {"pre": 0, "post": 1}
    plotted_values = []

    for group in ["control", "bci"]:
        group_values = []
        lower_errors = []
        upper_errors = []
        x = []
        for session in ["pre", "post"]:
            values = summary_long.loc[
                (summary_long["group"] == group)
                & (summary_long["session"] == session)
                & (summary_long["congruency"] == congruency),
                metric,
            ].dropna()
            if values.empty:
                _warn(
                    f"No values for Color/Shape plot: group={group}, "
                    f"session={session}, congruency={congruency}, metric={metric}."
                )
                mean_value = np.nan
                ci_low, ci_high = np.nan, np.nan
            else:
                mean_value = float(values.mean())
                ci_low, ci_high = bootstrap_mean_ci(
                    values,
                    n_boot=n_boot,
                    random_state=random_state + len(group) + len(session) + len(congruency) + len(metric),
                )
            group_values.append(mean_value)
            lower_errors.append(mean_value - ci_low if np.isfinite(ci_low) else np.nan)
            upper_errors.append(ci_high - mean_value if np.isfinite(ci_high) else np.nan)
            x.append(x_positions[session])
            if np.isfinite(mean_value):
                plotted_values.append(mean_value)
            if np.isfinite(ci_low):
                plotted_values.append(ci_low)
            if np.isfinite(ci_high):
                plotted_values.append(ci_high)

        yerr = np.array([lower_errors, upper_errors], dtype=float)
        ax.errorbar(
            x,
            group_values,
            yerr=yerr,
            marker="o",
            markersize=4.5,
            color=colors[group],
            linewidth=1.2,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            label=labels[group],
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)

    if plotted_values:
        ymin = min(plotted_values)
        ymax = max(plotted_values)
        pad = max((ymax - ymin) * 0.18, 0.02 if metric == "accuracy" else 20)
        ax.set_ylim(ymin - pad, ymax + pad)
    if metric == "accuracy":
        current = ax.get_ylim()
        ax.set_ylim(max(0, current[0]), min(1.02, current[1]))


def plot_color_shape_prepost_accuracy_rt_by_group(
    summary_long,
    output_dir=None,
    filename="color_shape_prepost_accuracy_rt_by_group.pdf",
    n_boot=5000,
    random_state=7,
):
    """Plot #20 pre/post accuracy and RT by group and congruency."""
    required_columns = [
        "subject_id",
        "group",
        "session",
        "congruency",
        "accuracy",
        "rt_correct",
    ]
    missing_columns = [col for col in required_columns if col not in summary_long.columns]
    if missing_columns:
        raise ValueError(
            "Cannot plot Color/Shape pre/post figure because long summary is "
            f"missing columns: {missing_columns}."
        )

    output_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharex=True)
    panel_specs = [
        (0, 0, "accuracy", "congruent", "p(Choose Best)", "Congruent"),
        (0, 1, "accuracy", "incongruent", "p(Choose Best)", "Incongruent"),
        (1, 0, "rt_correct", "congruent", "RT on correct trials (ms)", "Congruent"),
        (1, 1, "rt_correct", "incongruent", "RT on correct trials (ms)", "Incongruent"),
    ]
    for row, col, metric, congruency, ylabel, title in panel_specs:
        _color_shape_plot_panel(
            axes[row, col],
            summary_long,
            metric,
            congruency,
            ylabel,
            title,
            n_boot=n_boot,
            random_state=random_state,
        )

    axes[0, 0].text(
        -0.22,
        1.08,
        "Accuracy",
        transform=axes[0, 0].transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )
    axes[1, 0].text(
        -0.22,
        1.08,
        "Reaction time",
        transform=axes[1, 0].transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )
    axes[0, 0].legend(frameon=False, loc="best")
    axes[0, 1].legend(frameon=False, loc="best")
    fig.suptitle("Pre/post averages by group", y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Saved Color/Shape #20 pre/post figure: {output_path}")
    return fig, axes, output_path


def load_and_plot_color_shape_prepost_accuracy_rt_by_group(
    data_dir=None,
    save_outputs=True,
    output_dir=None,
    figure_dir=None,
    excluded_subjects=None,
    min_rt_ms=150,
    mad_threshold=3,
    bin_width_ms=150,
    n_boot=5000,
):
    """Run final summaries and plot #20 Color/Shape pre/post outcomes."""
    results = load_compute_color_shape_rt_and_analysis_log(
        data_dir=data_dir,
        save_outputs=save_outputs,
        output_dir=output_dir,
        figure_dir=figure_dir,
        excluded_subjects=excluded_subjects,
        min_rt_ms=min_rt_ms,
        mad_threshold=mad_threshold,
        bin_width_ms=bin_width_ms,
    )
    summary_long = create_color_shape_summary_long(results["analysis_log"])
    group_summary = summarize_color_shape_group_level(summary_long)
    results["summary_long"] = summary_long
    results["group_summary"] = group_summary

    if save_outputs:
        output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_long_path = output_dir / "color_shape_summary_long.csv"
        group_summary_path = output_dir / "color_shape_group_summary.csv"
        summary_long.to_csv(summary_long_path, index=False)
        group_summary.to_csv(group_summary_path, index=False)
        results["output_paths"]["summary_long"] = summary_long_path
        results["output_paths"]["group_summary"] = group_summary_path
        print(f"Saved Color/Shape long summary: {summary_long_path}")
        print(f"Saved Color/Shape group summary: {group_summary_path}")

    fig, axes, figure_path = plot_color_shape_prepost_accuracy_rt_by_group(
        summary_long,
        output_dir=figure_dir,
        n_boot=n_boot,
    )
    results["prepost_accuracy_rt_figure"] = fig
    results["prepost_accuracy_rt_axes"] = axes
    results["output_paths"]["prepost_accuracy_rt_figure"] = figure_path
    return results


def _prepare_two_session_anova_data(summary_long, congruency, dv):
    """Prepare complete pre/post subject rows for one congruency and DV."""
    subset = summary_long[
        (summary_long["congruency"] == congruency)
        & summary_long[dv].notna()
        & summary_long["group"].isin(["bci", "control"])
        & summary_long["session"].isin(["pre", "post"])
    ].copy()

    if subset.empty:
        _warn(f"No data available for mixed ANOVA: congruency={congruency}, dv={dv}.")
        return subset

    complete_subjects = (
        subset.groupby("subject_id")["session"]
        .nunique()
        .loc[lambda counts: counts == 2]
        .index
    )
    subset = subset[subset["subject_id"].isin(complete_subjects)].copy()
    represented_groups = set(subset["group"].dropna().unique())

    if len(complete_subjects) == 0:
        _warn(f"No complete pre/post subjects for congruency={congruency}, dv={dv}.")
        return pd.DataFrame(columns=subset.columns)
    if represented_groups != {"bci", "control"}:
        _warn(
            f"Both groups are not represented for congruency={congruency}, dv={dv}; "
            f"found groups={sorted(represented_groups)}."
        )
        return pd.DataFrame(columns=subset.columns)

    print(
        f"Prepared mixed ANOVA data for {dv}, {congruency}: "
        f"{subset['subject_id'].nunique()} subjects, "
        f"{subset.groupby('group')['subject_id'].nunique().to_dict()}."
    )
    return subset


def _mixed_anova_fallback_two_by_two(subset, dv, congruency):
    """Fallback ANOVA-style table for 2-group × 2-session mixed design."""
    wide = (
        subset.pivot_table(
            index=["subject_id", "group"],
            columns="session",
            values=dv,
            aggfunc="mean",
        )
        .dropna(subset=["pre", "post"])
        .reset_index()
    )
    if wide.empty or set(wide["group"].unique()) != {"bci", "control"}:
        _warn(f"Insufficient complete data for fallback ANOVA: {dv}, {congruency}.")
        return pd.DataFrame()

    wide["subject_mean"] = wide[["pre", "post"]].mean(axis=1)
    wide["change"] = wide["post"] - wide["pre"]

    bci_mean = wide.loc[wide["group"] == "bci", "subject_mean"]
    control_mean = wide.loc[wide["group"] == "control", "subject_mean"]
    bci_change = wide.loc[wide["group"] == "bci", "change"]
    control_change = wide.loc[wide["group"] == "control", "change"]

    tests = []

    group_t = stats.ttest_ind(bci_mean, control_mean, equal_var=True, nan_policy="omit")
    group_df2 = len(bci_mean) + len(control_mean) - 2
    group_f = float(group_t.statistic ** 2)
    tests.append({
        "Source": "Group",
        "SS": np.nan,
        "DF1": 1,
        "DF2": group_df2,
        "MS": np.nan,
        "F": group_f,
        "p-unc": float(group_t.pvalue),
        "np2": group_f / (group_f + group_df2) if group_df2 > 0 else np.nan,
    })

    session_t = stats.ttest_1samp(wide["change"], popmean=0, nan_policy="omit")
    session_df2 = len(wide["change"].dropna()) - 1
    session_f = float(session_t.statistic ** 2)
    tests.append({
        "Source": "Session",
        "SS": np.nan,
        "DF1": 1,
        "DF2": session_df2,
        "MS": np.nan,
        "F": session_f,
        "p-unc": float(session_t.pvalue),
        "np2": session_f / (session_f + session_df2) if session_df2 > 0 else np.nan,
    })

    interaction_t = stats.ttest_ind(
        bci_change,
        control_change,
        equal_var=True,
        nan_policy="omit",
    )
    interaction_df2 = len(bci_change) + len(control_change) - 2
    interaction_f = float(interaction_t.statistic ** 2)
    tests.append({
        "Source": "Interaction",
        "SS": np.nan,
        "DF1": 1,
        "DF2": interaction_df2,
        "MS": np.nan,
        "F": interaction_f,
        "p-unc": float(interaction_t.pvalue),
        "np2": (
            interaction_f / (interaction_f + interaction_df2)
            if interaction_df2 > 0
            else np.nan
        ),
    })

    anova = pd.DataFrame(tests)
    anova.insert(0, "congruency", congruency)
    anova.insert(0, "dv", dv)
    anova["method"] = "fallback_2x2_t_equivalent"
    return anova


def run_color_shape_mixed_anova(summary_long, dv, label=None):
    """Run #21/#22 mixed ANOVA separately for congruent and incongruent trials.

    Uses pingouin.mixed_anova when available. If pingouin is unavailable, uses
    the equivalent tests for this 2-group × 2-session design:

    - group effect: independent t-test on each subject's pre/post mean
    - session effect: one-sample t-test on post-pre change scores
    - interaction: independent t-test on post-pre change scores by group
    """
    if label is None:
        label = dv

    all_results = []
    try:
        import pingouin as pg
    except ImportError as exc:
        pg = None
        _warn(
            "pingouin is not installed; using 2x2 t-equivalent fallback for "
            f"Color/Shape {label} mixed ANOVA. Original import error: {exc}."
        )

    for congruency in ["congruent", "incongruent"]:
        subset = _prepare_two_session_anova_data(summary_long, congruency, dv)
        if subset.empty:
            continue

        if pg is not None:
            try:
                anova = pg.mixed_anova(
                    data=subset,
                    dv=dv,
                    within="session",
                    between="group",
                    subject="subject_id",
                )
                anova.insert(0, "congruency", congruency)
                anova.insert(0, "dv", dv)
                anova["method"] = "pingouin.mixed_anova"
            except Exception as exc:
                _warn(
                    f"pingouin mixed_anova failed for {label}, {congruency}: {exc}. "
                    "Using fallback."
                )
                anova = _mixed_anova_fallback_two_by_two(subset, dv, congruency)
        else:
            anova = _mixed_anova_fallback_two_by_two(subset, dv, congruency)

        if not anova.empty:
            all_results.append(anova)
            print(f"Mixed ANOVA results for {label}, {congruency}:")
            print(anova.to_string(index=False))

    if not all_results:
        _warn(f"No mixed ANOVA results were generated for {label}.")
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def load_and_run_color_shape_mixed_anovas(
    data_dir=None,
    save_outputs=True,
    output_dir=None,
    figure_dir=None,
    excluded_subjects=None,
    min_rt_ms=150,
    mad_threshold=3,
    bin_width_ms=150,
):
    """Run #21 accuracy and #22 RT mixed ANOVAs."""
    results = load_and_plot_color_shape_prepost_accuracy_rt_by_group(
        data_dir=data_dir,
        save_outputs=save_outputs,
        output_dir=output_dir,
        figure_dir=figure_dir,
        excluded_subjects=excluded_subjects,
        min_rt_ms=min_rt_ms,
        mad_threshold=mad_threshold,
        bin_width_ms=bin_width_ms,
    )

    accuracy_anova = run_color_shape_mixed_anova(
        results["summary_long"],
        dv="accuracy",
        label="accuracy",
    )
    rt_anova = run_color_shape_mixed_anova(
        results["summary_long"],
        dv="rt_correct",
        label="RT",
    )
    results["accuracy_mixed_anova"] = accuracy_anova
    results["rt_mixed_anova"] = rt_anova

    if save_outputs:
        output_dir = Path(output_dir) if output_dir is not None else ANALYSES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        accuracy_path = output_dir / "color_shape_accuracy_mixed_anova.csv"
        rt_path = output_dir / "color_shape_rt_mixed_anova.csv"
        accuracy_anova.to_csv(accuracy_path, index=False)
        rt_anova.to_csv(rt_path, index=False)
        results["output_paths"]["accuracy_mixed_anova"] = accuracy_path
        results["output_paths"]["rt_mixed_anova"] = rt_path
        print(f"Saved Color/Shape accuracy mixed ANOVA: {accuracy_path}")
        print(f"Saved Color/Shape RT mixed ANOVA: {rt_path}")

    return results
