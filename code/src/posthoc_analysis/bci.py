"""Consolidate BCI threshold-log and online-posterior trial data."""

from collections import Counter, defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, BCI_GROUP_SUBJECTS, EXPECTED_SUBJECTS
from .consolidated import get_session_number, parse_session_folder_name


ONLINE_INFO_DIRNAME = "online_info"
ANALYSES_DIRNAME = "analyses"
REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = REPO_ROOT / "figures"
TRIALS_PER_BCI_RUN = 60
EXPECTED_SESSIONS = 5
EXPECTED_REAL_RUNS_BY_SESSION = {1: 6, 2: 8, 3: 8, 4: 8, 5: 6}
EXPECTED_POSTERIOR_RAW_RUNS_BY_SESSION = {1: 7, 2: 9, 3: 9, 4: 9, 5: 7}
EXPECTED_THRLOG_RAW_RUNS_BY_SESSION = {1: 8, 2: 9, 3: 9, 4: 9, 5: 7}
BCI_SESSION_EXCEPTIONS = {
    ("e27", 1): {
        "real_runs": 5,
        "posterior_raw_runs": 5,
        "posterior_runs_to_drop": 0,
        "posterior_tail_rows_to_drop": 0,
        "threshold_raw_runs": 6,
        "threshold_runs_to_drop": 1,
        "threshold_tail_runs_to_drop": 0,
        "note": (
            "e27 session 1 completed only five decoding runs including practice; "
            "all five posterior runs are retained as experimental runs."
        ),
    },
    ("e42", 2): {
        "real_runs": 8,
        "posterior_raw_runs": 9,
        "posterior_runs_to_drop": 1,
        "posterior_tail_rows_to_drop": 4,
        "threshold_raw_runs": 9,
        "threshold_runs_to_drop": 1,
        "threshold_tail_runs_to_drop": 1,
        "note": (
            "e42 session 2 online posterior has four trailing extra rows and "
            "the threshold log has one extra trailing run row; trim those "
            "tail records before normal practice removal."
        ),
    },
    ("e43", 5): {
        "real_runs": 6,
        "posterior_raw_runs": 7,
        "posterior_runs_to_drop": 1,
        "posterior_rows_to_drop": None,
        "posterior_tail_rows_to_drop": 2,
        "threshold_raw_runs": 7,
        "threshold_runs_to_drop": 1,
        "threshold_tail_runs_to_drop": 1,
        "note": (
            "e43 session 5 online posterior has two trailing extra rows and "
            "the threshold log has one extra trailing run row; trim those "
            "tail records before normal practice removal."
        ),
    },
    ("e46", 4): {
        "real_runs": 8,
        "posterior_raw_runs": None,
        "posterior_runs_to_drop": None,
        "posterior_rows_to_drop": 32,
        "posterior_tail_rows_to_drop": 0,
        "threshold_raw_runs": 9,
        "threshold_runs_to_drop": 1,
        "threshold_tail_runs_to_drop": 0,
        "note": (
            "e46 session 4 has a shortened 32-trial practice run in the online "
            "posterior file; remove only the first 32 posterior rows, while "
            "removing the normal leading threshold-log practice row."
        ),
    },
}
THRLOG_FIELDS = ["subjectID", "timestamp", "margin", "thrR", "thrL", "thrN"]
POSTERIOR_COLUMNS = [
    "posterior_probability",
    "threshold",
    "classification_output",
]
DECODING_ANALYSIS_COLUMNS = [
    "trial_index",
    "task",
    "feedback",
    "target_position",
    "distractor_position",
    "dot_side",
    "intertrial_interval_ms",
    "bci_output",
]
DECODING_ANALYSIS_OUTPUT_COLUMNS = [
    f"decoding_{column}" for column in DECODING_ANALYSIS_COLUMNS
]
DECODING_BCI_OUTPUT_VALUES = {0, 1, 3}


def _loadmat(filepath):
    """Load a MATLAB v5 file with an informative dependency error."""
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise ModuleNotFoundError(
            "Reading BCI .mat files requires scipy.io.loadmat in the active "
            f"Python/Jupyter kernel. scipy could not be imported: {exc}"
        ) from exc

    return loadmat(filepath, squeeze_me=True, struct_as_record=False)


def _normal_group_label(subject_id):
    return "experimental" if subject_id in BCI_GROUP_SUBJECTS else "control"


def _online_info_dir(root_path=None):
    root = Path(PROJECT_ROOT if root_path is None else root_path)
    return root / ONLINE_INFO_DIRNAME


def _analysis_output_path(output_path=None):
    if output_path is not None:
        return Path(output_path)
    return PROJECT_ROOT / ANALYSES_DIRNAME / "all_subjects_bci.csv"


def _project_root(root_path=None):
    return Path(PROJECT_ROOT if root_path is None else root_path)


def _posterior_file_date(filepath):
    match = re.search(r"_OnlinePosteriors_(\d{8})", Path(filepath).name)
    if not match:
        raise ValueError(f"Could not parse session date from posterior filename: {filepath}")
    return match.group(1)


def _session_exception(subject_id, session_id):
    return BCI_SESSION_EXCEPTIONS.get((subject_id, int(session_id)))


def _expected_real_runs(subject_id, session_id):
    exception = _session_exception(subject_id, session_id)
    if exception is not None:
        return exception["real_runs"]
    return EXPECTED_REAL_RUNS_BY_SESSION[int(session_id)]


def _expected_raw_runs(subject_id, session_id, source_name):
    exception = _session_exception(subject_id, session_id)
    if exception is not None:
        expected_raw = exception[f"{source_name}_raw_runs"]
        if expected_raw is not None:
            return expected_raw
    if source_name == "posterior":
        return EXPECTED_POSTERIOR_RAW_RUNS_BY_SESSION[int(session_id)]
    return EXPECTED_THRLOG_RAW_RUNS_BY_SESSION[int(session_id)]


def _forced_practice_drop(subject_id, session_id, source_name):
    exception = _session_exception(subject_id, session_id)
    if exception is None:
        return None
    return exception[f"{source_name}_runs_to_drop"]


def _forced_practice_row_drop(subject_id, session_id, source_name):
    exception = _session_exception(subject_id, session_id)
    if exception is None or source_name != "posterior":
        return None
    return exception.get("posterior_rows_to_drop")


def _tail_drop_count(subject_id, session_id, source_name):
    exception = _session_exception(subject_id, session_id)
    if exception is None:
        return 0
    if source_name == "posterior":
        return exception.get("posterior_tail_rows_to_drop", 0)
    return exception.get("threshold_tail_runs_to_drop", 0)


def _expected_decoding_analysis_runs(subject_id, session_id):
    if (subject_id, int(session_id)) == ("e27", 1):
        return 4
    return EXPECTED_REAL_RUNS_BY_SESSION[int(session_id)]


def _decoding_analysis_run_offset(subject_id, session_id):
    if (subject_id, int(session_id)) == ("e27", 1):
        return 1
    return 0


def _expected_missing_decoding_analysis_run_keys():
    # e27 Session 1 BCI run 1 is retained from the shortened practice-inclusive
    # posterior file, while decoding_practice analysis files are intentionally
    # excluded from this CSV.
    return {("e27", 1, 1)}


def _select_posterior_files(subject_id, online_info_dir):
    """Return one posterior file per date, preferring files with practice rows."""
    all_files = sorted(online_info_dir.glob(f"{subject_id}_OnlinePosteriors_*.mat"))
    by_date = defaultdict(list)
    for filepath in all_files:
        by_date[_posterior_file_date(filepath)].append(filepath)

    selected = []
    duplicate_choices = []
    for date, files in sorted(by_date.items()):
        preferred = sorted(
            files,
            key=lambda p: ("wpractice" not in p.name.lower(), len(p.name), p.name),
        )[0]
        selected.append(preferred)
        if len(files) > 1:
            duplicate_choices.append({
                "date": date,
                "selected": preferred.name,
                "available": [p.name for p in files],
            })

    return selected, duplicate_choices


def load_bci_posterior_file(filepath, leading_rows_to_drop=0, tail_rows_to_drop=0):
    """Load and validate one `*_OnlinePosteriors_*.mat` file.

    Returns a DataFrame with the documented posterior probability, threshold,
    and classification-output columns. Practice-run removal happens later,
    because expected raw row counts depend on session number.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Online posterior file not found: {filepath}")

    mat = _loadmat(filepath)
    if "OnlinePosteriors" not in mat:
        raise ValueError(
            f"Online posterior file {filepath} is missing the 'OnlinePosteriors' variable."
        )

    arr = np.asarray(mat["OnlinePosteriors"], dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Expected OnlinePosteriors shape n_rows × 3 in {filepath}, got {arr.shape}."
        )
    rows_after_documented_trim = (
        arr.shape[0] - int(leading_rows_to_drop) - int(tail_rows_to_drop)
    )
    if rows_after_documented_trim <= 0 or rows_after_documented_trim % TRIALS_PER_BCI_RUN != 0:
        raise ValueError(
            f"Expected OnlinePosteriors rows to be a positive multiple of "
            f"{TRIALS_PER_BCI_RUN} in {filepath}, got {arr.shape[0]} "
            f"after accounting for {leading_rows_to_drop} documented leading row(s) "
            f"and {tail_rows_to_drop} documented trailing row(s)."
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"OnlinePosteriors contains NaN or infinite values in {filepath}.")

    df = pd.DataFrame(arr, columns=POSTERIOR_COLUMNS)
    invalid_classes = sorted(
        set(df.loc[~df["classification_output"].isin([1, 2, 3]), "classification_output"])
    )
    if invalid_classes:
        raise ValueError(
            f"Invalid classification_output values in {filepath}: {invalid_classes}. "
            "Expected only 1, 2, or 3."
        )

    return df


def load_bci_threshold_log_file(filepath, expected_subject_id=None):
    """Load and validate one `*_thrlog.mat` threshold log."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Threshold log file not found: {filepath}")

    mat = _loadmat(filepath)
    if "thrLog" not in mat:
        raise ValueError(f"Threshold log file {filepath} is missing the 'thrLog' variable.")

    entries = np.ravel(mat["thrLog"])
    rows = []
    for index, entry in enumerate(entries, 1):
        missing = [field for field in THRLOG_FIELDS if not hasattr(entry, field)]
        if missing:
            raise ValueError(
                f"Threshold log entry {index} in {filepath} is missing fields: {missing}."
            )
        subject_id = str(entry.subjectID).strip()
        if expected_subject_id is not None and subject_id != expected_subject_id:
            raise ValueError(
                f"Threshold log entry {index} in {filepath} has subjectID={subject_id}; "
                f"expected {expected_subject_id}."
            )
        timestamp = str(entry.timestamp).strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2} ", timestamp):
            raise ValueError(
                f"Threshold log entry {index} in {filepath} has unexpected timestamp: "
                f"{timestamp!r}."
            )
        values = {
            "margin": float(entry.margin),
            "thrR": float(entry.thrR),
            "thrL": float(entry.thrL),
            "thrN": float(entry.thrN),
        }
        if not np.isfinite(list(values.values())).all():
            raise ValueError(
                f"Threshold log entry {index} in {filepath} contains non-finite values."
            )
        rows.append({
            "subject_id": subject_id,
            "timestamp": timestamp,
            "session_date": timestamp[:10].replace("-", ""),
            **values,
        })

    if not rows:
        raise ValueError(f"Threshold log file {filepath} contains no entries.")

    return pd.DataFrame(rows)


def load_decoding_analysis_file(filepath):
    """Load and validate a non-practice decoding `.analysis.txt` file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Decoding analysis file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=DECODING_ANALYSIS_COLUMNS,
        dtype={column: int for column in DECODING_ANALYSIS_COLUMNS},
    )
    validate_decoding_analysis(df, filepath)
    return df


def validate_decoding_analysis(df, filepath=None):
    """Validate that a decoding analysis file matches documented structure."""
    location = f" {filepath}" if filepath else ""

    if list(df.columns) != DECODING_ANALYSIS_COLUMNS:
        raise ValueError(
            f"Unexpected columns in decoding analysis file{location}. "
            f"Expected {DECODING_ANALYSIS_COLUMNS}, got {list(df.columns)}."
        )
    if df.shape[0] != TRIALS_PER_BCI_RUN:
        raise ValueError(
            f"Expected {TRIALS_PER_BCI_RUN} rows in decoding analysis file{location}, "
            f"got {df.shape[0]} rows."
        )

    expected_trials = list(range(1, TRIALS_PER_BCI_RUN + 1))
    if df["trial_index"].tolist() != expected_trials:
        raise ValueError(
            f"Trial indices must be consecutive 1..{TRIALS_PER_BCI_RUN} "
            f"in decoding analysis file{location}."
        )

    checks = {
        "task": {0, 1},
        "feedback": {1, 2, 3},
        "target_position": {1, 2, 3, 4},
        "distractor_position": {0, 2, 4},
        "dot_side": {0, 1},
        "bci_output": DECODING_BCI_OUTPUT_VALUES,
    }
    for column, valid_values in checks.items():
        invalid = sorted(set(df.loc[~df[column].isin(valid_values), column]))
        if invalid:
            raise ValueError(
                f"Invalid {column} values in decoding analysis file{location}: "
                f"{invalid}. Expected only {sorted(valid_values)}."
            )

    if (df["intertrial_interval_ms"] <= 0).any():
        invalid = df.loc[df["intertrial_interval_ms"] <= 0, "intertrial_interval_ms"].tolist()
        raise ValueError(
            f"Intertrial intervals must be positive in decoding analysis file{location}. "
            f"Found: {invalid}."
        )


def _subject_session_folders(root_path, subject_id):
    subject_dir = _project_root(root_path) / subject_id
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    session_folders = sorted([
        folder for folder in subject_dir.iterdir()
        if folder.is_dir()
        and parse_session_folder_name(folder.name) is not None
    ])
    if len(session_folders) != EXPECTED_SESSIONS:
        raise ValueError(
            f"{subject_id}: expected {EXPECTED_SESSIONS} session folders, "
            f"found {len(session_folders)}."
        )
    return session_folders


def _collect_subject_decoding_analysis(subject_id, root_path, issues):
    """Collect non-practice decoding analysis rows keyed by session/run/trial."""
    session_folders = _subject_session_folders(root_path, subject_id)
    rows = []

    for session_index, session_folder in enumerate(session_folders):
        session_id = get_session_number(session_index)
        expected_runs = _expected_decoding_analysis_runs(subject_id, session_id)
        run_offset = _decoding_analysis_run_offset(subject_id, session_id)

        decoding_folders = sorted([
            folder for folder in session_folder.iterdir()
            if folder.is_dir()
            and folder.name.endswith("_decoding")
            and "decoding_practice" not in folder.name
        ])
        practice_folders = sorted([
            folder for folder in session_folder.iterdir()
            if folder.is_dir() and folder.name.endswith("_decoding_practice")
        ])

        print(
            f"  Decoding analysis session {session_id}: found "
            f"{len(decoding_folders)} non-practice decoding folder(s), "
            f"{len(practice_folders)} practice folder(s); expected "
            f"{expected_runs} non-practice analysis run(s)."
        )

        selected_folders = decoding_folders
        if len(decoding_folders) == expected_runs + 1:
            issues["decoding_analysis_warnings"].append({
                "subject_id": subject_id,
                "session_id": session_id,
                "issue": (
                    "Found one extra non-practice-labeled decoding folder; "
                    "skipping the first folder as practice-like for BCI alignment."
                ),
                "skipped_folder": decoding_folders[0].name,
            })
            selected_folders = decoding_folders[1:]
            print(
                "    WARNING: one extra decoding folder found; skipping first "
                f"folder for alignment: {decoding_folders[0].name}"
            )
        elif len(decoding_folders) != expected_runs:
            issues["decoding_analysis_file_count_mismatches"].append({
                "subject_id": subject_id,
                "session_id": session_id,
                "expected_runs": expected_runs,
                "found_runs": len(decoding_folders),
                "run_folders": [folder.name for folder in decoding_folders],
            })
            print(
                f"    WARNING: expected {expected_runs} non-practice decoding "
                f"analysis runs, found {len(decoding_folders)}."
            )

        if len(practice_folders) != 1:
            issues["decoding_analysis_warnings"].append({
                "subject_id": subject_id,
                "session_id": session_id,
                "issue": (
                    f"Expected 1 decoding_practice folder for file-structure checks, "
                    f"found {len(practice_folders)}. Practice analysis files are not merged."
                ),
                "practice_folders": [folder.name for folder in practice_folders],
            })

        for run_index, run_folder in enumerate(selected_folders[:expected_runs], 1):
            analysis_file = run_folder / f"{run_folder.name}.analysis.txt"
            if not analysis_file.exists():
                issues["decoding_analysis_load_errors"].append({
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "run_folder": run_folder.name,
                    "file": str(analysis_file),
                    "issue": "Missing decoding analysis file.",
                })
                continue

            try:
                analysis_df = load_decoding_analysis_file(analysis_file)
            except Exception as exc:
                issues["decoding_analysis_load_errors"].append({
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "run_folder": run_folder.name,
                    "file": str(analysis_file),
                    "issue": str(exc),
                })
                continue

            analysis_df = analysis_df.copy()
            analysis_df["subject_id"] = subject_id
            analysis_df["session_id"] = session_id
            analysis_df["run_id"] = run_index + run_offset
            analysis_df["trial_id"] = analysis_df["trial_index"]
            analysis_df = analysis_df.rename(columns={
                column: f"decoding_{column}" for column in DECODING_ANALYSIS_COLUMNS
            })
            rows.append(
                analysis_df[
                    ["subject_id", "session_id", "run_id", "trial_id"]
                    + DECODING_ANALYSIS_OUTPUT_COLUMNS
                ]
            )

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=[
        "subject_id",
        "session_id",
        "run_id",
        "trial_id",
        *DECODING_ANALYSIS_OUTPUT_COLUMNS,
    ])


def _drop_practice_runs(df, subject_id, session_id, source_name, filepath, issues):
    """Drop leading practice runs while preserving already-practice-free files."""
    expected_real = _expected_real_runs(subject_id, session_id)
    expected_raw = _expected_raw_runs(subject_id, session_id, source_name)
    expected_practice = expected_raw - expected_real
    forced_drop = _forced_practice_drop(subject_id, session_id, source_name)
    forced_row_drop = _forced_practice_row_drop(subject_id, session_id, source_name)
    tail_drop = _tail_drop_count(subject_id, session_id, source_name)

    if tail_drop:
        if len(df) <= tail_drop:
            raise ValueError(
                f"Cannot trim {tail_drop} trailing {source_name} row(s) for "
                f"{subject_id} session {session_id} in {filepath}; only {len(df)} rows found."
            )
        df = df.iloc[:-tail_drop].reset_index(drop=True)
        unit = "posterior row(s)" if source_name == "posterior" else "threshold-log run row(s)"
        print(
            f"  Subject-specific exception: trimming {tail_drop} trailing {unit} "
            f"before practice removal."
        )

    if source_name == "posterior":
        found_runs = len(df) // TRIALS_PER_BCI_RUN
        extra_rows = len(df) % TRIALS_PER_BCI_RUN
        if extra_rows and forced_row_drop is None:
            raise ValueError(
                f"Cannot split posterior rows into 60-trial runs for {filepath}: "
                f"{len(df)} rows."
            )
    else:
        found_runs = len(df)

    if source_name == "posterior" and forced_row_drop is not None:
        print(
            f"  Session {session_id} {source_name}: found {len(df)} row(s) before "
            f"practice removal; expected {expected_real * TRIALS_PER_BCI_RUN} real "
            f"rows after removing {forced_row_drop} practice row(s)."
        )
    else:
        print(
            f"  Session {session_id} {source_name}: found {found_runs} run(s) before "
            f"practice removal; expected {expected_raw} raw / {expected_real} real."
        )

    exception = _session_exception(subject_id, session_id)
    if source_name == "posterior" and forced_row_drop is not None:
        if len(df) - forced_row_drop != expected_real * TRIALS_PER_BCI_RUN:
            raise ValueError(
                f"posterior exception for {subject_id} session {session_id} expected "
                f"{expected_real * TRIALS_PER_BCI_RUN} rows after dropping "
                f"{forced_row_drop} practice row(s), found {len(df) - forced_row_drop}."
            )
        runs_to_drop = None
        print(f"    Subject-specific exception: {exception['note']}")
        print(f"    Practice retained as shortened row block: dropping first {forced_row_drop} row(s).")
    elif forced_drop is not None:
        if found_runs != expected_raw:
            raise ValueError(
                f"{source_name} exception for {subject_id} session {session_id} expected "
                f"{expected_raw} raw run(s), found {found_runs} in {filepath}."
            )
        runs_to_drop = forced_drop
        print(f"    Subject-specific exception: {exception['note']}")
        if runs_to_drop == 0:
            print("    Practice retained: dropping 0 run(s).")
        else:
            print(f"    Dropping first {runs_to_drop} threshold/log run(s) for alignment.")
    elif found_runs == expected_raw:
        runs_to_drop = expected_practice
        print(f"    Practice present: dropping first {runs_to_drop} run(s).")
    elif found_runs == expected_real:
        runs_to_drop = 0
        issues.append({
            "file": str(filepath),
            "issue": (
                f"{source_name} appears already practice-removed for session {session_id}; "
                f"found {found_runs} real runs and no leading practice runs to drop."
            ),
        })
        print("    WARNING: practice run(s) not present; treating file as already cleaned.")
    elif found_runs > expected_real:
        runs_to_drop = found_runs - expected_real
        issues.append({
            "file": str(filepath),
            "issue": (
                f"{source_name} session {session_id} has {found_runs} runs rather than "
                f"documented {expected_raw}; dropping first {runs_to_drop} run(s) to "
                f"retain {expected_real} real runs."
            ),
        })
        print(
            f"    WARNING: nonstandard raw count; dropping first {runs_to_drop} run(s) "
            "to align to documented real runs."
        )
    else:
        raise ValueError(
            f"{source_name} session {session_id} in {filepath} has {found_runs} run(s), "
            f"fewer than the expected {expected_real} real runs."
        )

    if source_name == "posterior":
        if forced_row_drop is not None:
            start = forced_row_drop
        else:
            start = runs_to_drop * TRIALS_PER_BCI_RUN
        cleaned = df.iloc[start:].reset_index(drop=True)
        cleaned_runs = len(cleaned) // TRIALS_PER_BCI_RUN
    else:
        cleaned = df.iloc[runs_to_drop:].reset_index(drop=True)
        cleaned_runs = len(cleaned)

    print(
        f"    After practice removal: {cleaned_runs} real run(s); "
        f"expected {expected_real}."
    )
    if cleaned_runs != expected_real:
        raise ValueError(
            f"{source_name} practice removal for session {session_id} in {filepath} "
            f"left {cleaned_runs} runs; expected {expected_real}."
        )

    return cleaned


def _threshold_groups_by_date(threshold_df):
    return {
        date: group.sort_values("timestamp").reset_index(drop=True)
        for date, group in threshold_df.groupby("session_date", sort=True)
    }


def _build_subject_bci_dataframe(subject_id, online_info_dir, issues):
    threshold_files = sorted(online_info_dir.glob(f"{subject_id}_thrlog.mat"))
    posterior_files, duplicate_choices = _select_posterior_files(subject_id, online_info_dir)

    print(f"\nSubject {subject_id}")
    print(f"  Threshold log files found: {len(threshold_files)}")
    print(f"  Online posterior session files selected: {len(posterior_files)}")

    if duplicate_choices:
        print("  Duplicate posterior files found; selected practice-inclusive candidates:")
        for choice in duplicate_choices:
            print(f"    {choice['date']}: {choice['selected']}")

    if len(threshold_files) != 1:
        issues["missing_or_duplicate_threshold_logs"].append({
            "subject_id": subject_id,
            "found_files": [p.name for p in threshold_files],
            "issue": f"Expected exactly 1 threshold log, found {len(threshold_files)}.",
        })
        return pd.DataFrame()
    if len(posterior_files) != EXPECTED_SESSIONS:
        issues["missing_or_duplicate_posterior_files"].append({
            "subject_id": subject_id,
            "found_files": [p.name for p in posterior_files],
            "issue": f"Expected {EXPECTED_SESSIONS} posterior session files, found {len(posterior_files)}.",
        })
        return pd.DataFrame()

    for choice in duplicate_choices:
        issues["duplicate_posterior_files"].append({
            "subject_id": subject_id,
            **choice,
        })

    threshold_df = load_bci_threshold_log_file(threshold_files[0], subject_id)
    threshold_by_date = _threshold_groups_by_date(threshold_df)
    posterior_dates = [_posterior_file_date(path) for path in posterior_files]

    print(f"  Threshold log entries by date: {dict(Counter(threshold_df['session_date']))}")
    print(f"  Posterior dates: {posterior_dates}")

    missing_threshold_dates = sorted(set(posterior_dates) - set(threshold_by_date))
    if missing_threshold_dates:
        raise ValueError(
            f"Threshold log {threshold_files[0]} is missing dates needed by posterior files: "
            f"{missing_threshold_dates}."
        )

    subject_frames = []
    for session_id, posterior_path in enumerate(posterior_files, 1):
        try:
            session_date = _posterior_file_date(posterior_path)
            posterior_df = load_bci_posterior_file(
                posterior_path,
                leading_rows_to_drop=(
                    _forced_practice_row_drop(subject_id, session_id, "posterior") or 0
                ),
                tail_rows_to_drop=_tail_drop_count(subject_id, session_id, "posterior"),
            )
            threshold_session_df = threshold_by_date[session_date]

            print(f"  Validating session {session_id} ({session_date})")
            posterior_clean = _drop_practice_runs(
                posterior_df,
                subject_id,
                session_id,
                "posterior",
                posterior_path,
                issues["practice_warnings"],
            )
            threshold_clean = _drop_practice_runs(
                threshold_session_df,
                subject_id,
                session_id,
                "threshold",
                threshold_files[0],
                issues["practice_warnings"],
            )

            expected_real = _expected_real_runs(subject_id, session_id)
            posterior_runs = len(posterior_clean) // TRIALS_PER_BCI_RUN
            threshold_runs = len(threshold_clean)
            if posterior_runs != threshold_runs:
                raise ValueError(
                    f"Threshold/posterior run mismatch for {subject_id} session {session_id}: "
                    f"{threshold_runs} threshold runs vs {posterior_runs} posterior runs."
                )
            if posterior_runs != expected_real:
                raise ValueError(
                    f"{subject_id} session {session_id} has {posterior_runs} aligned real runs; "
                    f"expected {expected_real}."
                )

            for run_idx in range(expected_real):
                post_run = posterior_clean.iloc[
                    run_idx * TRIALS_PER_BCI_RUN:(run_idx + 1) * TRIALS_PER_BCI_RUN
                ].reset_index(drop=True)
                if len(post_run) != TRIALS_PER_BCI_RUN:
                    raise ValueError(
                        f"{subject_id} session {session_id} run {run_idx + 1} has "
                        f"{len(post_run)} posterior trials; expected {TRIALS_PER_BCI_RUN}."
                    )

                threshold_row = threshold_clean.iloc[run_idx]
                run_frame = post_run.copy()
                run_frame["subject_id"] = subject_id
                run_frame["group"] = _normal_group_label(subject_id)
                run_frame["session_id"] = session_id
                run_frame["run_id"] = run_idx + 1
                run_frame["trial_id"] = np.arange(1, TRIALS_PER_BCI_RUN + 1)
                for column in ["margin", "thrR", "thrL", "thrN"]:
                    run_frame[column] = threshold_row[column]
                subject_frames.append(run_frame)
        except Exception as exc:
            issues["load_or_alignment_errors"].append({
                "subject_id": subject_id,
                "session_id": session_id,
                "file": str(posterior_path),
                "issue": str(exc),
            })
            print(f"  ERROR: skipping {subject_id} session {session_id} because {exc}")

    return pd.concat(subject_frames, ignore_index=True) if subject_frames else pd.DataFrame()


def validate_bci_consolidated_dataframe(df):
    """Print and return validation checks for the final BCI CSV DataFrame."""
    issues = []
    expected_columns = [
        "subject_id",
        "group",
        "session_id",
        "run_id",
        "trial_id",
        "margin",
        "thrR",
        "thrL",
        "thrN",
        "posterior_probability",
        "threshold",
        "classification_output",
        *DECODING_ANALYSIS_OUTPUT_COLUMNS,
    ]

    print("\n" + "=" * 80)
    print("FINAL BCI CSV VALIDATION")
    print("=" * 80)
    print(f"Rows: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique() if not df.empty else 0}")

    if list(df.columns) != expected_columns:
        issues.append(
            f"Unexpected final columns. Expected {expected_columns}, got {list(df.columns)}."
        )

    if df.empty:
        issues.append("Final BCI DataFrame is empty.")
        print("No rows available for row-count validation.")
        return {"valid": False, "issues": issues}

    missing_subjects = sorted(set(EXPECTED_SUBJECTS) - set(df["subject_id"].unique()))
    if missing_subjects:
        issues.append(f"Expected subjects missing from final CSV: {missing_subjects}.")

    invalid_classes = sorted(
        set(df.loc[~df["classification_output"].isin([1, 2, 3]), "classification_output"])
    )
    if invalid_classes:
        issues.append(f"Invalid classification_output values in final CSV: {invalid_classes}.")

    missing_analysis_mask = df[DECODING_ANALYSIS_OUTPUT_COLUMNS].isna().all(axis=1)
    if missing_analysis_mask.any():
        missing_groups = (
            df.loc[missing_analysis_mask, ["subject_id", "session_id", "run_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        expected_missing = _expected_missing_decoding_analysis_run_keys()
        unexpected_missing = [
            (subject_id, int(session_id), int(run_id))
            for subject_id, session_id, run_id in missing_groups
            if (subject_id, int(session_id), int(run_id)) not in expected_missing
        ]
        if unexpected_missing:
            issues.append(
                "Missing decoding analysis rows for BCI run(s): "
                f"{unexpected_missing[:20]}"
            )
        expected_present = (
            df.loc[
                missing_analysis_mask
                & df[["subject_id", "session_id", "run_id"]].apply(
                    lambda row: (
                        row["subject_id"],
                        int(row["session_id"]),
                        int(row["run_id"]),
                    ) in expected_missing,
                    axis=1,
                )
            ]
        )
        if not expected_present.empty:
            print(
                "\nExpected missing decoding-analysis rows from documented practice exclusion: "
                f"{len(expected_present)}"
            )

    trial_counts = df.groupby(["subject_id", "session_id", "run_id"]).size()
    bad_trial_counts = trial_counts[trial_counts != TRIALS_PER_BCI_RUN]
    if not bad_trial_counts.empty:
        issues.append(
            f"Found {len(bad_trial_counts)} subject/session/run cells without "
            f"{TRIALS_PER_BCI_RUN} trials."
        )

    run_counts = df.groupby(["subject_id", "session_id"])["run_id"].nunique()
    bad_run_counts = []
    missing_subject_sessions = []
    for subject_id in EXPECTED_SUBJECTS:
        subject_sessions = set(
            df.loc[df["subject_id"] == subject_id, "session_id"].astype(int).unique()
        )
        for session_id in EXPECTED_REAL_RUNS_BY_SESSION:
            if session_id not in subject_sessions:
                missing_subject_sessions.append((subject_id, session_id))
    for (subject_id, session_id), found_runs in run_counts.items():
        expected_runs = _expected_real_runs(subject_id, session_id)
        if found_runs != expected_runs:
            bad_run_counts.append((subject_id, int(session_id), int(found_runs), expected_runs))
    if missing_subject_sessions:
        issues.append(
            "Missing complete subject/session cells in final CSV: "
            f"{missing_subject_sessions[:20]}"
        )
    if bad_run_counts:
        issues.append(
            "Found subject/session run-count mismatches after consolidation: "
            f"{bad_run_counts[:20]}"
        )

    expected_rows_by_session = {
        session_id: runs * TRIALS_PER_BCI_RUN
        for session_id, runs in EXPECTED_REAL_RUNS_BY_SESSION.items()
    }
    rows_by_subject_session = df.groupby(["subject_id", "session_id"]).size().unstack(fill_value=0)
    print("\nRows per subject/session:")
    print(rows_by_subject_session)
    print(f"\nExpected rows per complete subject/session: {expected_rows_by_session}")

    rows_by_run = trial_counts.value_counts().sort_index().to_dict()
    print(f"\nRun-level trial-count distribution: {rows_by_run}")

    if issues:
        print("\nValidation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nFinal CSV validation passed.")

    return {"valid": len(issues) == 0, "issues": issues}


def generate_consolidated_bci_csv(output_path=None, root_path=None):
    """Generate one trial-level BCI CSV from threshold logs and online posteriors."""
    output_path = _analysis_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    online_info_dir = _online_info_dir(root_path)
    if not online_info_dir.exists():
        raise FileNotFoundError(f"online_info directory not found: {online_info_dir}")

    print("Generating consolidated BCI CSV from threshold logs and online posteriors")
    print("=" * 80)
    print(f"Online info directory: {online_info_dir}")
    print(f"Output path: {output_path}")
    print(f"Expected subjects from config: {len(EXPECTED_SUBJECTS)}")

    issues = {
        "missing_or_duplicate_threshold_logs": [],
        "missing_or_duplicate_posterior_files": [],
        "duplicate_posterior_files": [],
        "practice_warnings": [],
        "load_or_alignment_errors": [],
        "decoding_analysis_file_count_mismatches": [],
        "decoding_analysis_load_errors": [],
        "decoding_analysis_warnings": [],
    }
    subject_frames = []

    unexpected_subjects = sorted(
        set(p.name.split("_")[0] for p in online_info_dir.glob("e*_OnlinePosteriors_*.mat"))
        - set(EXPECTED_SUBJECTS)
    )
    if unexpected_subjects:
        print(f"Unexpected posterior subjects ignored because they are not in config: {unexpected_subjects}")

    for subject_id in EXPECTED_SUBJECTS:
        try:
            subject_df = _build_subject_bci_dataframe(subject_id, online_info_dir, issues)
            if not subject_df.empty:
                decoding_analysis_df = _collect_subject_decoding_analysis(
                    subject_id, root_path, issues
                )
                subject_df = subject_df.merge(
                    decoding_analysis_df,
                    on=["subject_id", "session_id", "run_id", "trial_id"],
                    how="left",
                    validate="one_to_one",
                )
                subject_frames.append(subject_df)
        except Exception as exc:
            issues["load_or_alignment_errors"].append({
                "subject_id": subject_id,
                "issue": str(exc),
            })
            print(f"  ERROR: {subject_id} skipped because {exc}")

    if subject_frames:
        df = pd.concat(subject_frames, ignore_index=True)
        id_cols = ["subject_id", "group", "session_id", "run_id", "trial_id"]
        threshold_cols = ["margin", "thrR", "thrL", "thrN"]
        df = df[id_cols + threshold_cols + POSTERIOR_COLUMNS + DECODING_ANALYSIS_OUTPUT_COLUMNS]
        df["classification_output"] = df["classification_output"].astype(int)
        for column in DECODING_ANALYSIS_OUTPUT_COLUMNS:
            df[column] = df[column].astype("Int64")
        df.to_csv(output_path, index=False)
        print(f"\nSaved consolidated BCI data to: {output_path}")
    else:
        df = pd.DataFrame(columns=[
            "subject_id",
            "group",
            "session_id",
            "run_id",
            "trial_id",
            "margin",
            "thrR",
            "thrL",
            "thrN",
            *POSTERIOR_COLUMNS,
            *DECODING_ANALYSIS_OUTPUT_COLUMNS,
        ])
        print("\nNo BCI data were saved because no subject/session aligned successfully.")

    validation = validate_bci_consolidated_dataframe(df)

    print("\n" + "=" * 80)
    print("BCI CONSOLIDATION ISSUE SUMMARY")
    print("=" * 80)
    for key, values in issues.items():
        print(f"{key}: {len(values)}")
        for value in values[:20]:
            print(f"  - {value}")
        if len(values) > 20:
            print(f"  ... and {len(values) - 20} more")

    return {
        "dataframe": df,
        "output_path": str(output_path),
        "total_trials": len(df),
        "total_runs": (
            df.groupby(["subject_id", "session_id", "run_id"]).ngroups
            if not df.empty else 0
        ),
        "subjects_present": sorted(df["subject_id"].unique().tolist()) if not df.empty else [],
        "issues": issues,
        "validation": validation,
    }


def compute_bci_subject_session_auc(df):
    """Compute subject-level BCI AUC for each session using posterior probabilities.

    AUC is computed from all available runs within a subject/session. Labels come
    from `decoding_task` where 1 is distractor and 0 is no distractor.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:
        raise ModuleNotFoundError(
            "BCI AUC computation requires scikit-learn in the active environment."
        ) from exc

    required_columns = {
        "subject_id",
        "group",
        "session_id",
        "posterior_probability",
        "decoding_task",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "BCI AUC analysis is missing required columns: "
            f"{missing_columns}. Regenerate all_subjects_bci.csv with decoding "
            "analysis columns before running this analysis."
        )

    data = df.copy()
    print("=" * 80)
    print("BCI AUC INPUT VALIDATION")
    print("=" * 80)
    print(f"Input rows: {len(data)}")
    print(f"Subjects: {data['subject_id'].nunique()}")
    print(f"Sessions: {sorted(data['session_id'].dropna().unique().tolist())}")

    if data["posterior_probability"].isna().any():
        missing_posteriors = int(data["posterior_probability"].isna().sum())
        raise ValueError(f"Found {missing_posteriors} rows with missing posterior probabilities.")

    invalid_posteriors = data[
        (data["posterior_probability"] < 0) | (data["posterior_probability"] > 1)
    ]
    if not invalid_posteriors.empty:
        raise ValueError(
            "Posterior probabilities must be between 0 and 1. "
            f"Found {len(invalid_posteriors)} invalid rows."
        )

    missing_label_rows = data["decoding_task"].isna()
    if missing_label_rows.any():
        missing_summary = (
            data.loc[missing_label_rows]
            .groupby(["subject_id", "group", "session_id", "run_id"], observed=False)
            .size()
            .reset_index(name="n_missing_label_rows")
        )
        print("\nRows excluded because decoding_task is missing:")
        print(missing_summary.to_string(index=False))

    valid = data[
        data["posterior_probability"].notna()
        & data["decoding_task"].notna()
    ].copy()
    valid["decoding_task"] = valid["decoding_task"].astype(int)

    invalid_labels = sorted(set(valid["decoding_task"]) - {0, 1})
    if invalid_labels:
        raise ValueError(
            f"decoding_task must contain only 0/1 labels. Found: {invalid_labels}"
        )

    class_summary = (
        valid.groupby(["subject_id", "group", "session_id", "decoding_task"], observed=False)
        .size()
        .reset_index(name="n_trials")
        .pivot_table(
            index=["subject_id", "group", "session_id"],
            columns="decoding_task",
            values="n_trials",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={0: "n_no_distractor", 1: "n_distractor"})
    )
    for column in ["n_no_distractor", "n_distractor"]:
        if column not in class_summary.columns:
            class_summary[column] = 0

    rows = []
    skipped = []
    for (subject_id, group, session_id), session_df in valid.groupby(
        ["subject_id", "group", "session_id"], observed=False
    ):
        labels = session_df["decoding_task"].to_numpy()
        posteriors = session_df["posterior_probability"].to_numpy()
        class_counts = pd.Series(labels).value_counts().to_dict()
        if len(class_counts) < 2:
            skipped.append({
                "subject_id": subject_id,
                "group": group,
                "session_id": int(session_id),
                "n_trials": len(session_df),
                "n_no_distractor": int(class_counts.get(0, 0)),
                "n_distractor": int(class_counts.get(1, 0)),
                "issue": "Only one class present; AUC undefined.",
            })
            continue

        rows.append({
            "subject_id": subject_id,
            "group": group,
            "session_id": int(session_id),
            "auc": float(roc_auc_score(labels, posteriors)),
            "n_trials": int(len(session_df)),
            "n_no_distractor": int(class_counts.get(0, 0)),
            "n_distractor": int(class_counts.get(1, 0)),
            "n_runs": int(session_df["run_id"].nunique()) if "run_id" in session_df else np.nan,
        })

    subject_session_auc = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)

    print("\nSubject/session class-count summary:")
    print(class_summary.to_string(index=False))
    if not skipped_df.empty:
        print("\nAUC skipped for subject/session cells:")
        print(skipped_df.to_string(index=False))

    expected_subject_sessions = len(EXPECTED_SUBJECTS) * EXPECTED_SESSIONS
    print("\nAUC rows computed:")
    print(
        f"{len(subject_session_auc)} subject/session rows "
        f"(full design has {expected_subject_sessions})."
    )
    print(
        "AUC range: "
        f"{subject_session_auc['auc'].min():.3f} - "
        f"{subject_session_auc['auc'].max():.3f}"
        if not subject_session_auc.empty else "AUC range: no rows"
    )

    return {
        "subject_session_auc": subject_session_auc,
        "class_summary": class_summary,
        "skipped": skipped_df,
        "missing_label_summary": (
            missing_summary if missing_label_rows.any() else pd.DataFrame()
        ),
    }


def summarize_bci_auc_by_group(subject_session_auc, variability="sem"):
    """Summarize subject-level AUC by group and session."""
    if variability not in {"sem", "sd"}:
        raise ValueError("variability must be 'sem' or 'sd'.")
    if subject_session_auc.empty:
        raise ValueError("No subject/session AUC rows are available to summarize.")

    summary = (
        subject_session_auc
        .groupby(["group", "session_id"], observed=False)["auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_auc", "std": "sd_auc", "count": "n_subjects"})
    )
    summary["sem_auc"] = summary["sd_auc"] / np.sqrt(summary["n_subjects"])
    summary["error_auc"] = summary[f"{variability}_auc"]
    summary["variability"] = variability

    print("\nGroup/session AUC summary:")
    print(summary.to_string(index=False))
    return summary


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


def _set_axis_padding(ax, values, pad_fraction=0.12, min_pad=0.02, lower_bound=0.0, upper_bound=1.0):
    finite_values = [float(value) for value in values if pd.notna(value) and np.isfinite(value)]
    if not finite_values:
        ax.set_ylim(lower_bound, upper_bound)
        return
    low = min(finite_values)
    high = max(finite_values)
    spread = high - low
    pad = max(spread * pad_fraction, min_pad)
    ax.set_ylim(max(lower_bound, low - pad), min(upper_bound, high + pad))


def _save_figure_pdf(fig, filename_stem):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"{filename_stem}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    return output_path


def plot_bci_auc_across_sessions(
    group_summary,
    subject_session_auc=None,
    variability="sem",
    save=True,
    filename_stem="bci_auc_across_sessions",
):
    """Plot BCI AUC across five sessions for BCI and mental-rehearsal groups."""
    import matplotlib.pyplot as plt

    colors = {
        "experimental": "#DD8452",
        "control": "#4C72B0",
    }
    labels = {
        "experimental": "BCI",
        "control": "Mental rehearsal",
    }
    group_order = ["experimental", "control"]
    plotted_values = [0.5]

    with plt.rc_context(_publication_style_rcparams()):
        fig, ax = plt.subplots(figsize=(4.4, 3.2))

        for group in group_order:
            group_df = group_summary[group_summary["group"] == group].sort_values("session_id")
            if group_df.empty:
                continue
            yerr = group_df["error_auc"].fillna(0.0)
            plotted_values.extend((group_df["mean_auc"] - yerr).tolist())
            plotted_values.extend((group_df["mean_auc"] + yerr).tolist())
            ax.errorbar(
                group_df["session_id"],
                group_df["mean_auc"],
                yerr=yerr,
                marker="o",
                markersize=4.5,
                capsize=3,
                capthick=0.8,
                linewidth=1.5,
                color=colors[group],
                label=labels[group],
                zorder=3,
            )

        ax.axhline(0.5, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Session")
        ax.set_ylabel("AUC")
        ax.set_title(f"Online Classifier AUC Across Sessions ({variability.upper()})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)
        ax.tick_params(axis="both", which="both", length=3, width=0.8)
        _set_axis_padding(ax, plotted_values, pad_fraction=0.15, min_pad=0.03)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.8)
        fig.tight_layout(rect=[0, 0, 0.78, 1])

        output_path = _save_figure_pdf(fig, filename_stem) if save else None

    return fig, output_path


def load_and_plot_bci_auc(csv_path=None, variability="sem", save=True):
    """Load the consolidated BCI CSV, compute subject/session AUC, and plot by group."""
    if csv_path is None:
        csv_path = PROJECT_ROOT / ANALYSES_DIRNAME / "all_subjects_bci.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Consolidated BCI CSV not found: {csv_path}")

    print("=" * 80)
    print("BCI AUC ACROSS SESSIONS")
    print("=" * 80)
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    auc_results = compute_bci_subject_session_auc(df)
    group_summary = summarize_bci_auc_by_group(
        auc_results["subject_session_auc"],
        variability=variability,
    )
    fig, output_path = plot_bci_auc_across_sessions(
        group_summary,
        subject_session_auc=auc_results["subject_session_auc"],
        variability=variability,
        save=save,
    )

    return {
        "csv_path": str(csv_path),
        "dataframe": df,
        "subject_session_auc": auc_results["subject_session_auc"],
        "class_summary": auc_results["class_summary"],
        "missing_label_summary": auc_results["missing_label_summary"],
        "skipped": auc_results["skipped"],
        "group_summary": group_summary,
        "figure": fig,
        "figure_path": str(output_path) if output_path is not None else None,
    }


def compute_subject_averaged_posterior_distributions(
    df,
    sessions=(1, 5),
    n_bins=20,
    normalize="density",
    min_trials_per_class=1,
):
    """Compute subject-level posterior histograms before averaging by group.

    Histograms are computed separately for each subject, group, session, and
    `decoding_task` class. Summary values are then means and SEM across subjects,
    avoiding direct trial pooling across subjects.
    """
    if normalize not in {"density", "probability"}:
        raise ValueError("normalize must be 'density' or 'probability'.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if min_trials_per_class < 1:
        raise ValueError("min_trials_per_class must be at least 1.")

    required_columns = {
        "subject_id",
        "group",
        "session_id",
        "posterior_probability",
        "decoding_task",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Posterior distribution analysis is missing required columns: "
            f"{missing_columns}. Regenerate all_subjects_bci.csv with decoding "
            "analysis columns before running this analysis."
        )

    sessions = tuple(int(session_id) for session_id in sessions)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    data = df.copy()
    print("=" * 80)
    print("BCI POSTERIOR DISTRIBUTION INPUT VALIDATION")
    print("=" * 80)
    print(f"Input rows: {len(data)}")
    print(f"Requested sessions: {list(sessions)}")
    print(f"Histogram bins: {n_bins} fixed bins from 0 to 1")
    print(f"Normalization: {normalize}")

    if data["posterior_probability"].isna().any():
        missing_posteriors = int(data["posterior_probability"].isna().sum())
        raise ValueError(f"Found {missing_posteriors} rows with missing posterior probabilities.")

    invalid_posteriors = data[
        (data["posterior_probability"] < 0) | (data["posterior_probability"] > 1)
    ]
    if not invalid_posteriors.empty:
        invalid_summary = (
            invalid_posteriors
            .groupby(["subject_id", "group", "session_id"], observed=False)
            .size()
            .reset_index(name="n_invalid_rows")
        )
        raise ValueError(
            "Posterior probabilities must be between 0 and 1. Invalid rows:\n"
            f"{invalid_summary.to_string(index=False)}"
        )
    print("Posterior probability range check: PASS (all values are between 0 and 1).")

    session_data = data[data["session_id"].isin(sessions)].copy()
    if session_data.empty:
        raise ValueError(f"No rows found for requested sessions: {sessions}")

    missing_label_rows = session_data["decoding_task"].isna()
    if missing_label_rows.any():
        missing_label_summary = (
            session_data.loc[missing_label_rows]
            .groupby(["subject_id", "group", "session_id", "run_id"], observed=False)
            .size()
            .reset_index(name="n_missing_label_rows")
            if "run_id" in session_data.columns
            else session_data.loc[missing_label_rows]
            .groupby(["subject_id", "group", "session_id"], observed=False)
            .size()
            .reset_index(name="n_missing_label_rows")
        )
        print("\nRows excluded because decoding_task is missing:")
        print(missing_label_summary.to_string(index=False))
    else:
        missing_label_summary = pd.DataFrame()
        print("decoding_task label availability: PASS (no missing labels in requested sessions).")

    valid = session_data[
        session_data["posterior_probability"].notna()
        & session_data["decoding_task"].notna()
    ].copy()
    valid["decoding_task"] = valid["decoding_task"].astype(int)

    invalid_labels = sorted(set(valid["decoding_task"]) - {0, 1})
    if invalid_labels:
        raise ValueError(
            f"decoding_task must contain only 0/1 labels. Found: {invalid_labels}"
        )
    print("decoding_task class check: PASS (labels are 0/1).")

    requested_groups = ["experimental", "control"]
    subject_counts = (
        valid.groupby(["group", "session_id"], observed=False)["subject_id"]
        .nunique()
        .reset_index(name="n_subjects_with_any_labeled_trials")
        .sort_values(["group", "session_id"])
    )
    print("\nSubjects included per group/session:")
    print(subject_counts.to_string(index=False))

    trial_counts = (
        valid.groupby(["subject_id", "group", "session_id", "decoding_task"], observed=False)
        .size()
        .reset_index(name="n_trials")
        .sort_values(["group", "session_id", "subject_id", "decoding_task"])
    )
    print("\nTrials per subject/session/class:")
    print(trial_counts.to_string(index=False))

    expected_cells = pd.MultiIndex.from_product(
        [valid[["subject_id", "group"]].drop_duplicates().itertuples(index=False, name=None), sessions],
        names=["subject_group", "session_id"],
    )
    observed_subject_sessions = set(
        valid[["subject_id", "group", "session_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    missing_subject_sessions = []
    for (subject_id, group), session_id in expected_cells:
        if (subject_id, group, session_id) not in observed_subject_sessions:
            missing_subject_sessions.append({
                "subject_id": subject_id,
                "group": group,
                "session_id": int(session_id),
                "issue": "Missing requested session with labeled trials.",
            })
    missing_subject_sessions = pd.DataFrame(missing_subject_sessions)
    if not missing_subject_sessions.empty:
        print("\nMissing subject/session cells:")
        print(missing_subject_sessions.to_string(index=False))
    else:
        print("\nMissing subject/session cells: none for requested sessions.")

    complete_subject_class_index = pd.MultiIndex.from_product(
        [
            valid[["subject_id", "group", "session_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None),
            [0, 1],
        ],
        names=["subject_group_session", "decoding_task"],
    )
    observed_subject_class = set(
        trial_counts[["subject_id", "group", "session_id", "decoding_task"]]
        .itertuples(index=False, name=None)
    )
    missing_or_insufficient = []
    for (subject_id, group, session_id), decoding_task in complete_subject_class_index:
        row = trial_counts[
            (trial_counts["subject_id"] == subject_id)
            & (trial_counts["group"] == group)
            & (trial_counts["session_id"] == session_id)
            & (trial_counts["decoding_task"] == decoding_task)
        ]
        n_trials = int(row["n_trials"].iloc[0]) if not row.empty else 0
        if (subject_id, group, session_id, decoding_task) not in observed_subject_class:
            issue = "Missing class."
        elif n_trials < min_trials_per_class:
            issue = f"Fewer than {min_trials_per_class} trials."
        else:
            continue
        missing_or_insufficient.append({
            "subject_id": subject_id,
            "group": group,
            "session_id": int(session_id),
            "decoding_task": int(decoding_task),
            "n_trials": n_trials,
            "issue": issue,
        })
    missing_or_insufficient = pd.DataFrame(missing_or_insufficient)
    if not missing_or_insufficient.empty:
        print("\nMissing classes or subjects with insufficient class data:")
        print(missing_or_insufficient.to_string(index=False))
    else:
        print("\nMissing classes or insufficient class data: none.")

    histogram_rows = []
    skipped_histograms = []
    for (subject_id, group, session_id, decoding_task), class_df in valid.groupby(
        ["subject_id", "group", "session_id", "decoding_task"], observed=False
    ):
        n_trials = len(class_df)
        if n_trials < min_trials_per_class:
            skipped_histograms.append({
                "subject_id": subject_id,
                "group": group,
                "session_id": int(session_id),
                "decoding_task": int(decoding_task),
                "n_trials": int(n_trials),
                "issue": f"Fewer than {min_trials_per_class} trials.",
            })
            continue

        counts, _ = np.histogram(
            class_df["posterior_probability"].to_numpy(dtype=float),
            bins=bin_edges,
            density=(normalize == "density"),
        )
        if normalize == "probability":
            total = counts.sum()
            if total == 0:
                skipped_histograms.append({
                    "subject_id": subject_id,
                    "group": group,
                    "session_id": int(session_id),
                    "decoding_task": int(decoding_task),
                    "n_trials": int(n_trials),
                    "issue": "Histogram count total is zero.",
                })
                continue
            counts = counts / total

        for bin_index, value in enumerate(counts):
            histogram_rows.append({
                "subject_id": subject_id,
                "group": group,
                "session_id": int(session_id),
                "decoding_task": int(decoding_task),
                "bin_index": int(bin_index),
                "bin_left": float(bin_edges[bin_index]),
                "bin_right": float(bin_edges[bin_index + 1]),
                "bin_center": float(bin_centers[bin_index]),
                "histogram_value": float(value),
                "n_trials": int(n_trials),
            })

    subject_histograms = pd.DataFrame(histogram_rows)
    skipped_histograms = pd.DataFrame(skipped_histograms)
    if subject_histograms.empty:
        raise ValueError("No subject-level posterior histograms could be computed.")

    distribution_summary = (
        subject_histograms
        .groupby(["group", "session_id", "decoding_task", "bin_index"], observed=False)
        .agg(
            bin_left=("bin_left", "first"),
            bin_right=("bin_right", "first"),
            bin_center=("bin_center", "first"),
            mean_value=("histogram_value", "mean"),
            sd_value=("histogram_value", "std"),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )
    distribution_summary["sem_value"] = (
        distribution_summary["sd_value"] / np.sqrt(distribution_summary["n_subjects"])
    )
    distribution_summary["sem_value"] = distribution_summary["sem_value"].fillna(0.0)

    subject_class_counts = (
        subject_histograms
        .groupby(["group", "session_id", "decoding_task"], observed=False)["subject_id"]
        .nunique()
        .reset_index(name="n_subjects_in_class_distribution")
        .sort_values(["group", "session_id", "decoding_task"])
    )
    print("\nSubjects contributing to each group/session/class distribution:")
    print(subject_class_counts.to_string(index=False))

    panel_checks = []
    for group in requested_groups:
        for session_id in sessions:
            for decoding_task in [0, 1]:
                panel = distribution_summary[
                    (distribution_summary["group"] == group)
                    & (distribution_summary["session_id"] == session_id)
                    & (distribution_summary["decoding_task"] == decoding_task)
                ]
                panel_checks.append({
                    "group": group,
                    "session_id": int(session_id),
                    "decoding_task": decoding_task,
                    "n_bins": int(panel["bin_index"].nunique()) if not panel.empty else 0,
                    "n_subjects": (
                        int(panel["n_subjects"].max()) if not panel.empty else 0
                    ),
                    "status": "ok" if len(panel) == n_bins else "missing distribution bins",
                })
    panel_checks = pd.DataFrame(panel_checks)
    print("\nPanel/bin completeness checks:")
    print(panel_checks.to_string(index=False))

    if not skipped_histograms.empty:
        print("\nSkipped subject-level histograms:")
        print(skipped_histograms.to_string(index=False))

    return {
        "subject_histograms": subject_histograms,
        "distribution_summary": distribution_summary,
        "trial_counts": trial_counts,
        "subject_counts": subject_counts,
        "subject_class_counts": subject_class_counts,
        "missing_label_summary": missing_label_summary,
        "missing_subject_sessions": missing_subject_sessions,
        "missing_or_insufficient": missing_or_insufficient,
        "skipped_histograms": skipped_histograms,
        "panel_checks": panel_checks,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "normalize": normalize,
    }


def plot_subject_averaged_posterior_distributions(
    distribution_summary,
    normalize="density",
    save=True,
    filename_stem="bci_posterior_distributions_subject_averaged",
):
    """Plot subject-averaged posterior distributions for Sessions 1 and 5."""
    import matplotlib.pyplot as plt

    group_labels = {
        "experimental": "BCI",
        "control": "Mental rehearsal",
    }
    class_labels = {
        1: "Distractor",
        0: "No distractor",
    }
    class_colors = {
        1: "#C44E52",
        0: "#4C72B0",
    }
    panel_order = [
        ("experimental", 1),
        ("experimental", 5),
        ("control", 1),
        ("control", 5),
    ]
    y_label = (
        "Subject-averaged density"
        if normalize == "density"
        else "Subject-averaged probability"
    )

    with plt.rc_context(_publication_style_rcparams()):
        fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8), sharex=True, sharey=True)
        axes = axes.ravel()
        max_y = 0.0

        for ax, (group, session_id) in zip(axes, panel_order):
            panel_df = distribution_summary[
                (distribution_summary["group"] == group)
                & (distribution_summary["session_id"] == session_id)
            ]
            for decoding_task in [1, 0]:
                class_df = panel_df[
                    panel_df["decoding_task"] == decoding_task
                ].sort_values("bin_index")
                if class_df.empty:
                    continue
                x = class_df["bin_center"].to_numpy(dtype=float)
                y = class_df["mean_value"].to_numpy(dtype=float)
                sem = class_df["sem_value"].fillna(0.0).to_numpy(dtype=float)
                max_y = max(max_y, float(np.nanmax(y + sem)))
                ax.plot(
                    x,
                    y,
                    color=class_colors[decoding_task],
                    linewidth=1.6,
                    label=class_labels[decoding_task],
                )
                ax.fill_between(
                    x,
                    np.maximum(0.0, y - sem),
                    y + sem,
                    color=class_colors[decoding_task],
                    alpha=0.18,
                    linewidth=0,
                )

            ax.set_title(f"{group_labels[group]} - Session {session_id}")
            ax.set_xlim(0, 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.8)
            ax.spines["left"].set_linewidth(0.8)
            ax.tick_params(axis="both", which="both", length=3, width=0.8)

        axes[2].set_xlabel("Posterior probability of distractor class")
        axes[3].set_xlabel("Posterior probability of distractor class")
        axes[0].set_ylabel(y_label)
        axes[2].set_ylabel(y_label)
        if max_y > 0:
            for ax in axes:
                ax.set_ylim(0, max_y * 1.12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            handlelength=2.2,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = _save_figure_pdf(fig, filename_stem) if save else None

    return fig, output_path


def load_and_plot_bci_posterior_distributions(
    csv_path=None,
    sessions=(1, 5),
    n_bins=20,
    normalize="density",
    save=True,
):
    """Load the consolidated BCI CSV and plot subject-averaged posterior distributions."""
    if csv_path is None:
        csv_path = PROJECT_ROOT / ANALYSES_DIRNAME / "all_subjects_bci.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Consolidated BCI CSV not found: {csv_path}")

    print("=" * 80)
    print("BCI SUBJECT-AVERAGED POSTERIOR DISTRIBUTIONS")
    print("=" * 80)
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    distribution_results = compute_subject_averaged_posterior_distributions(
        df,
        sessions=sessions,
        n_bins=n_bins,
        normalize=normalize,
    )
    fig, output_path = plot_subject_averaged_posterior_distributions(
        distribution_results["distribution_summary"],
        normalize=normalize,
        save=save,
    )

    return {
        "csv_path": str(csv_path),
        "dataframe": df,
        **distribution_results,
        "figure": fig,
        "figure_path": str(output_path) if output_path is not None else None,
    }


def run_bci_auc_mixed_anova(subject_session_auc):
    """Run a 2 x 5 mixed-design ANOVA on subject-level BCI AUC.

    Between-subject factor: group (`experimental` vs `control`).
    Within-subject factor: session_id (1-5).
    """
    from scipy import stats

    required_columns = {"subject_id", "group", "session_id", "auc"}
    missing_columns = sorted(required_columns - set(subject_session_auc.columns))
    if missing_columns:
        raise ValueError(
            "Mixed ANOVA requires subject-session AUC columns: "
            f"{missing_columns}"
        )

    data = subject_session_auc[list(required_columns)].copy()
    data["session_id"] = data["session_id"].astype(int)
    data = data.dropna(subset=["subject_id", "group", "session_id", "auc"])

    print("=" * 80)
    print("BCI AUC MIXED-DESIGN ANOVA")
    print("=" * 80)
    print("Design: Group (between: BCI vs mental rehearsal) x Session (within: 1-5)")

    groups = sorted(data["group"].unique().tolist())
    sessions = sorted(data["session_id"].unique().tolist())
    if groups != ["control", "experimental"]:
        raise ValueError(f"Expected groups ['control', 'experimental'], found {groups}.")
    if sessions != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected sessions [1, 2, 3, 4, 5], found {sessions}.")

    counts = (
        data.groupby(["subject_id", "group"], observed=False)["session_id"]
        .nunique()
        .reset_index(name="n_sessions")
    )
    incomplete = counts[counts["n_sessions"] != EXPECTED_SESSIONS]
    if not incomplete.empty:
        raise ValueError(
            "Mixed ANOVA requires complete 5-session AUC data for every subject. "
            f"Incomplete subjects:\n{incomplete.to_string(index=False)}"
        )

    duplicate_cells = (
        data.groupby(["subject_id", "session_id"], observed=False)
        .size()
        .reset_index(name="n_rows")
    )
    duplicate_cells = duplicate_cells[duplicate_cells["n_rows"] != 1]
    if not duplicate_cells.empty:
        raise ValueError(
            "Expected exactly one AUC row per subject/session. Problem cells:\n"
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
    print(f"Sessions: {sessions}")

    grand_mean = data["auc"].mean()
    group_means = data.groupby("group", observed=False)["auc"].mean()
    session_means = data.groupby("session_id", observed=False)["auc"].mean()
    group_session_means = (
        data.groupby(["group", "session_id"], observed=False)["auc"].mean()
    )
    subject_means = (
        data.groupby(["group", "subject_id"], observed=False)["auc"].mean()
    )

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
        ss_error += (
            row.auc - subject_mean - group_session_mean + group_mean
        ) ** 2

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
        "grand_mean_auc": float(grand_mean),
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


def load_and_run_bci_auc_mixed_anova(csv_path=None):
    """Load the consolidated BCI CSV, compute AUC, and run the 2 x 5 mixed ANOVA."""
    auc_results = load_and_plot_bci_auc(csv_path=csv_path, variability="sem", save=False)
    anova_results = run_bci_auc_mixed_anova(auc_results["subject_session_auc"])
    return {
        **auc_results,
        "anova_table": anova_results["anova_table"],
        "anova_diagnostics": anova_results["diagnostics"],
    }


def compute_bci_threshold_change_session1_to_session5(df):
    """Compute subject-level Session 5 - Session 1 threshold changes by group.

    Threshold values in the consolidated BCI CSV are run-level values repeated on
    each trial. This function first collapses to unique subject/session/run rows,
    then averages thresholds across runs within each subject/session.
    """
    required_columns = {
        "subject_id",
        "group",
        "session_id",
        "run_id",
        "thrR",
        "thrL",
        "thrN",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Threshold change analysis is missing required columns: "
            f"{missing_columns}."
        )

    data = df.copy()
    print("=" * 80)
    print("BCI THRESHOLD CHANGE INPUT VALIDATION")
    print("=" * 80)
    print(f"Input trial-level rows: {len(data)}")

    invalid_groups = sorted(set(data["group"].dropna()) - {"experimental", "control"})
    if invalid_groups:
        raise ValueError(
            "Expected group values to be 'experimental' or 'control'. "
            f"Found: {invalid_groups}"
        )

    data["session_id"] = data["session_id"].astype(int)
    for column in ["thrR", "thrL", "thrN"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    missing_thresholds = (
        data[["thrR", "thrL", "thrN"]]
        .isna()
        .sum()
        .rename_axis("threshold")
        .reset_index(name="n_missing_trial_rows")
    )
    print("\nMissing threshold values in trial-level CSV:")
    print(missing_thresholds.to_string(index=False))

    per_run_nunique = (
        data.groupby(["subject_id", "group", "session_id", "run_id"], observed=False)[
            ["thrR", "thrL", "thrN"]
        ]
        .nunique(dropna=False)
        .reset_index()
    )
    inconsistent_runs = per_run_nunique[
        (per_run_nunique[["thrR", "thrL", "thrN"]] > 1).any(axis=1)
    ]
    if not inconsistent_runs.empty:
        raise ValueError(
            "Threshold values should be constant within each subject/session/run "
            "because they are run-level values repeated across trials. Problem runs:\n"
            f"{inconsistent_runs.to_string(index=False)}"
        )

    trial_counts_per_run = (
        data.groupby(["subject_id", "group", "session_id", "run_id"], observed=False)
        .size()
        .reset_index(name="n_trial_rows")
    )
    run_level = (
        data[
            ["subject_id", "group", "session_id", "run_id", "thrR", "thrL", "thrN"]
        ]
        .drop_duplicates()
        .merge(
            trial_counts_per_run,
            on=["subject_id", "group", "session_id", "run_id"],
            how="left",
        )
    )
    print(
        "\nThreshold aggregation level check: "
        f"{len(data)} trial rows collapsed to {len(run_level)} unique run rows."
    )
    print("Thresholds will be averaged across run rows, not trial rows.")

    sessions = [1, 5]
    run_level = run_level[run_level["session_id"].isin(sessions)].copy()
    if run_level.empty:
        raise ValueError("No Session 1 or Session 5 threshold rows found.")

    subjects_by_group = (
        run_level[["subject_id", "group"]]
        .drop_duplicates()
        .groupby("group", observed=False)
        .size()
        .reset_index(name="n_subjects_with_session1_or_session5")
        .sort_values("group")
    )
    print("\nNumber of subjects per group:")
    print(subjects_by_group.to_string(index=False))

    session_presence = (
        run_level.groupby(["subject_id", "group"], observed=False)["session_id"]
        .nunique()
        .reset_index(name="n_prepost_sessions")
    )
    incomplete_subjects = session_presence[session_presence["n_prepost_sessions"] != 2]
    if not incomplete_subjects.empty:
        print("\nSubjects missing Session 1 or Session 5:")
        print(incomplete_subjects.to_string(index=False))
    else:
        print("\nSession presence check: PASS (each subject has both Session 1 and Session 5).")

    run_counts = (
        run_level.groupby(["subject_id", "group", "session_id"], observed=False)["run_id"]
        .nunique()
        .reset_index(name="n_run_rows_averaged")
        .sort_values(["group", "subject_id", "session_id"])
    )
    print("\nRun rows averaged per subject/session:")
    print(run_counts.to_string(index=False))

    run_level["thrN"] = 1.0 - run_level["thrN"]
    session_means = (
        run_level.groupby(["subject_id", "group", "session_id"], observed=False)[
            ["thrR", "thrL", "thrN"]
        ]
        .mean()
        .reset_index()
    )

    long_means = session_means.melt(
        id_vars=["subject_id", "group", "session_id"],
        value_vars=["thrR", "thrL", "thrN"],
        var_name="threshold_type",
        value_name="session_mean",
    )
    wide = (
        long_means.pivot_table(
            index=["subject_id", "group", "threshold_type"],
            columns="session_id",
            values="session_mean",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={1: "session_1_mean", 5: "session_5_mean"})
    )
    complete = wide.dropna(subset=["session_1_mean", "session_5_mean"]).copy()
    complete["delta"] = complete["session_5_mean"] - complete["session_1_mean"]

    missing_change = wide[
        wide[["session_1_mean", "session_5_mean"]].isna().any(axis=1)
    ].copy()
    if not missing_change.empty:
        print("\nSubject/threshold rows excluded from change scores due to missing pre/post means:")
        print(missing_change.to_string(index=False))

    summary = (
        complete.groupby(["group", "threshold_type"], observed=False)["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_delta", "std": "sd_delta", "count": "n_subjects"})
    )
    summary["SEM"] = summary["sd_delta"] / np.sqrt(summary["n_subjects"])
    summary = summary[["group", "threshold_type", "mean_delta", "SEM", "n_subjects"]]
    summary["threshold_type"] = pd.Categorical(
        summary["threshold_type"],
        categories=["thrR", "thrL", "thrN"],
        ordered=True,
    )
    summary = summary.sort_values(["threshold_type", "group"]).reset_index(drop=True)

    print("\nThreshold change summary:")
    print(summary.to_string(index=False))

    return {
        "run_level_thresholds": run_level,
        "run_counts": run_counts,
        "session_means": session_means,
        "subject_threshold_change": complete,
        "summary": summary,
        "missing_thresholds": missing_thresholds,
        "missing_change": missing_change,
        "incomplete_subjects": incomplete_subjects,
    }


def _save_figure_pdf_to_path(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    return output_path


def _summarize_threshold_delta_from_session_means(
    session_means,
    threshold_columns,
    threshold_order,
):
    long_means = session_means.melt(
        id_vars=["subject_id", "group", "session_id"],
        value_vars=threshold_columns,
        var_name="threshold_type",
        value_name="session_mean",
    )
    wide = (
        long_means.pivot_table(
            index=["subject_id", "group", "threshold_type"],
            columns="session_id",
            values="session_mean",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={1: "session_1_mean", 5: "session_5_mean"})
    )
    complete = wide.dropna(subset=["session_1_mean", "session_5_mean"]).copy()
    complete["delta"] = complete["session_5_mean"] - complete["session_1_mean"]

    missing_change = wide[
        wide[["session_1_mean", "session_5_mean"]].isna().any(axis=1)
    ].copy()

    summary = (
        complete.groupby(["group", "threshold_type"], observed=False)["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_delta", "std": "sd_delta", "count": "n_subjects"})
    )
    summary["SEM"] = summary["sd_delta"] / np.sqrt(summary["n_subjects"])
    summary = summary[["group", "threshold_type", "mean_delta", "SEM", "n_subjects"]]
    summary["threshold_type"] = pd.Categorical(
        summary["threshold_type"],
        categories=threshold_order,
        ordered=True,
    )
    summary = summary.sort_values(["threshold_type", "group"]).reset_index(drop=True)

    return complete, summary, missing_change


def compute_bci_combined_threshold_change_session1_to_session5(df):
    """Compute pre/post threshold change with left/right distractor thresholds combined."""
    threshold_results = compute_bci_threshold_change_session1_to_session5(df)
    session_means = threshold_results["session_means"].copy()
    session_means["distractor"] = session_means[["thrR", "thrL"]].mean(axis=1)
    session_means["no_distractor"] = session_means["thrN"]

    print("\nCombined threshold definition:")
    print("- distractor = mean(thrR, thrL) after run-averaging within subject/session")
    print("- no_distractor = transformed thrN after run-averaging within subject/session")

    combined_change, combined_summary, combined_missing_change = (
        _summarize_threshold_delta_from_session_means(
            session_means,
            threshold_columns=["distractor", "no_distractor"],
            threshold_order=["distractor", "no_distractor"],
        )
    )

    if not combined_missing_change.empty:
        print("\nCombined subject/threshold rows excluded due to missing pre/post means:")
        print(combined_missing_change.to_string(index=False))

    print("\nCombined threshold change summary:")
    print(combined_summary.to_string(index=False))

    return {
        **threshold_results,
        "combined_session_means": session_means,
        "combined_subject_threshold_change": combined_change,
        "combined_summary": combined_summary,
        "combined_missing_change": combined_missing_change,
    }


def plot_bci_threshold_change_session1_to_session5(
    summary,
    save=True,
    output_path=None,
):
    """Plot Session 5 - Session 1 threshold change by group."""
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = FIGURES_DIR / "threshold_change_session1_to_session5_by_group.pdf"

    group_order = ["experimental", "control"]
    threshold_order = ["thrR", "thrL", "thrN"]
    threshold_labels = {
        "thrR": "Distractor\nRight",
        "thrL": "Distractor\nLeft",
        "thrN": "No\nDistractor",
    }
    colors = {
        "experimental": "#DD8452",
        "control": "#4C72B0",
    }
    labels = {
        "experimental": "BCI",
        "control": "Mental rehearsal",
    }

    summary = summary.copy()
    summary["threshold_type"] = summary["threshold_type"].astype(str)

    with plt.rc_context(_publication_style_rcparams()):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        x = np.arange(len(threshold_order))
        width = 0.34
        offsets = {
            "experimental": -width / 2,
            "control": width / 2,
        }

        plotted_values = [0.0]
        for group in group_order:
            group_summary = (
                summary[summary["group"] == group]
                .set_index("threshold_type")
                .reindex(threshold_order)
            )
            means = group_summary["mean_delta"].to_numpy(dtype=float)
            sem = group_summary["SEM"].fillna(0.0).to_numpy(dtype=float)
            plotted_values.extend((means - sem).tolist())
            plotted_values.extend((means + sem).tolist())
            ax.bar(
                x + offsets[group],
                means,
                width=width,
                color=colors[group],
                edgecolor="black",
                linewidth=0.5,
                label=labels[group],
                zorder=3,
            )
            ax.errorbar(
                x + offsets[group],
                means,
                yerr=sem,
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=3,
                capthick=0.8,
                zorder=4,
            )

        ax.axhline(0, color="#444444", linewidth=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([threshold_labels[value] for value in threshold_order])
        ax.set_xlabel("Threshold type")
        ax.set_ylabel("Session 5 - Session 1 threshold change")
        ax.set_title("Threshold Change From Session 1 to Session 5")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)
        ax.tick_params(axis="both", which="both", length=3, width=0.8)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.6)

        finite_values = [
            float(value) for value in plotted_values
            if pd.notna(value) and np.isfinite(value)
        ]
        if finite_values:
            low = min(finite_values)
            high = max(finite_values)
            spread = high - low
            pad = max(spread * 0.16, 0.02)
            ax.set_ylim(low - pad, high + pad)
        fig.tight_layout(rect=[0, 0, 0.78, 1])

        saved_path = _save_figure_pdf_to_path(fig, output_path) if save else None

    return fig, saved_path


def plot_bci_combined_threshold_change_session1_to_session5(
    summary,
    save=True,
    output_path=None,
):
    """Plot Session 5 - Session 1 threshold change with distractor sides combined."""
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = (
            FIGURES_DIR
            / "threshold_change_session1_to_session5_combined_distractor_by_group.pdf"
        )

    group_order = ["experimental", "control"]
    threshold_order = ["distractor", "no_distractor"]
    threshold_labels = {
        "distractor": "Distractor",
        "no_distractor": "No\nDistractor",
    }
    colors = {
        "experimental": "#DD8452",
        "control": "#4C72B0",
    }
    labels = {
        "experimental": "BCI",
        "control": "Mental rehearsal",
    }

    summary = summary.copy()
    summary["threshold_type"] = summary["threshold_type"].astype(str)

    with plt.rc_context(_publication_style_rcparams()):
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        x = np.arange(len(threshold_order))
        width = 0.34
        offsets = {
            "experimental": -width / 2,
            "control": width / 2,
        }

        plotted_values = [0.0]
        for group in group_order:
            group_summary = (
                summary[summary["group"] == group]
                .set_index("threshold_type")
                .reindex(threshold_order)
            )
            means = group_summary["mean_delta"].to_numpy(dtype=float)
            sem = group_summary["SEM"].fillna(0.0).to_numpy(dtype=float)
            plotted_values.extend((means - sem).tolist())
            plotted_values.extend((means + sem).tolist())
            ax.bar(
                x + offsets[group],
                means,
                width=width,
                color=colors[group],
                edgecolor="black",
                linewidth=0.5,
                label=labels[group],
                zorder=3,
            )
            ax.errorbar(
                x + offsets[group],
                means,
                yerr=sem,
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=3,
                capthick=0.8,
                zorder=4,
            )

        ax.axhline(0, color="#444444", linewidth=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([threshold_labels[value] for value in threshold_order])
        ax.set_xlabel("Threshold type")
        ax.set_ylabel("Session 5 - Session 1 threshold change")
        ax.set_title("Threshold Change From Session 1 to Session 5")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)
        ax.tick_params(axis="both", which="both", length=3, width=0.8)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.6)

        finite_values = [
            float(value) for value in plotted_values
            if pd.notna(value) and np.isfinite(value)
        ]
        if finite_values:
            low = min(finite_values)
            high = max(finite_values)
            spread = high - low
            pad = max(spread * 0.16, 0.02)
            ax.set_ylim(low - pad, high + pad)
        fig.tight_layout(rect=[0, 0, 0.76, 1])

        saved_path = _save_figure_pdf_to_path(fig, output_path) if save else None

    return fig, saved_path


def load_and_plot_bci_threshold_change_session1_to_session5(
    csv_path=None,
    save=True,
    output_path=None,
):
    """Load consolidated BCI CSV, compute pre/post threshold change, and plot it."""
    if csv_path is None:
        csv_path = PROJECT_ROOT / ANALYSES_DIRNAME / "all_subjects_bci.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Consolidated BCI CSV not found: {csv_path}")

    print("=" * 80)
    print("BCI THRESHOLD CHANGE SESSION 1 TO SESSION 5")
    print("=" * 80)
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    threshold_results = compute_bci_threshold_change_session1_to_session5(df)
    fig, figure_path = plot_bci_threshold_change_session1_to_session5(
        threshold_results["summary"],
        save=save,
        output_path=output_path,
    )

    return {
        "csv_path": str(csv_path),
        "dataframe": df,
        **threshold_results,
        "figure": fig,
        "figure_path": str(figure_path) if figure_path is not None else None,
    }


def load_and_plot_bci_combined_threshold_change_session1_to_session5(
    csv_path=None,
    save=True,
    output_path=None,
):
    """Load BCI CSV and plot pre/post threshold change with distractor sides combined."""
    if csv_path is None:
        csv_path = PROJECT_ROOT / ANALYSES_DIRNAME / "all_subjects_bci.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Consolidated BCI CSV not found: {csv_path}")

    print("=" * 80)
    print("BCI COMBINED DISTRACTOR THRESHOLD CHANGE SESSION 1 TO SESSION 5")
    print("=" * 80)
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    threshold_results = compute_bci_combined_threshold_change_session1_to_session5(df)
    fig, figure_path = plot_bci_combined_threshold_change_session1_to_session5(
        threshold_results["combined_summary"],
        save=save,
        output_path=output_path,
    )

    return {
        "csv_path": str(csv_path),
        "dataframe": df,
        **threshold_results,
        "figure": fig,
        "figure_path": str(figure_path) if figure_path is not None else None,
    }


def run_bci_combined_threshold_change_mixed_anova(combined_subject_threshold_change):
    """Run Group x Threshold Type mixed ANOVA on combined threshold-change scores.

    Dependent variable: Session 5 - Session 1 threshold change.
    Between-subject factor: group (`experimental` vs `control`).
    Within-subject factor: threshold_type (`distractor` vs `no_distractor`).
    """
    from scipy import stats

    required_columns = {"subject_id", "group", "threshold_type", "delta"}
    missing_columns = sorted(required_columns - set(combined_subject_threshold_change.columns))
    if missing_columns:
        raise ValueError(
            "Combined threshold mixed ANOVA requires columns: "
            f"{missing_columns}"
        )

    data = combined_subject_threshold_change[list(required_columns)].copy()
    data = data.dropna(subset=["subject_id", "group", "threshold_type", "delta"])
    data["threshold_type"] = data["threshold_type"].astype(str)

    print("=" * 80)
    print("BCI COMBINED THRESHOLD CHANGE MIXED-DESIGN ANOVA")
    print("=" * 80)
    print("Design: Group (between: BCI vs mental rehearsal) x Threshold type (within: distractor vs no distractor)")
    print("Dependent variable: Session 5 - Session 1 threshold change")

    groups = sorted(data["group"].unique().tolist())
    threshold_types = sorted(data["threshold_type"].unique().tolist())
    expected_groups = ["control", "experimental"]
    expected_threshold_types = ["distractor", "no_distractor"]
    if groups != expected_groups:
        raise ValueError(f"Expected groups {expected_groups}, found {groups}.")
    if threshold_types != expected_threshold_types:
        raise ValueError(
            f"Expected threshold types {expected_threshold_types}, found {threshold_types}."
        )

    duplicate_cells = (
        data.groupby(["subject_id", "threshold_type"], observed=False)
        .size()
        .reset_index(name="n_rows")
    )
    duplicate_cells = duplicate_cells[duplicate_cells["n_rows"] != 1]
    if not duplicate_cells.empty:
        raise ValueError(
            "Expected exactly one delta row per subject/threshold type. Problem cells:\n"
            f"{duplicate_cells.to_string(index=False)}"
        )

    counts = (
        data.groupby(["subject_id", "group"], observed=False)["threshold_type"]
        .nunique()
        .reset_index(name="n_threshold_types")
    )
    incomplete = counts[counts["n_threshold_types"] != len(expected_threshold_types)]
    if not incomplete.empty:
        raise ValueError(
            "Mixed ANOVA requires complete distractor/no-distractor deltas for each subject. "
            f"Incomplete subjects:\n{incomplete.to_string(index=False)}"
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
    print(f"Threshold types: {expected_threshold_types}")

    grand_mean = data["delta"].mean()
    group_means = data.groupby("group", observed=False)["delta"].mean()
    threshold_means = data.groupby("threshold_type", observed=False)["delta"].mean()
    group_threshold_means = (
        data.groupby(["group", "threshold_type"], observed=False)["delta"].mean()
    )
    subject_means = (
        data.groupby(["group", "subject_id"], observed=False)["delta"].mean()
    )

    n_thresholds = len(expected_threshold_types)
    n_total_subjects = data["subject_id"].nunique()
    n_by_group = (
        data[["subject_id", "group"]]
        .drop_duplicates()
        .groupby("group", observed=False)
        .size()
    )

    ss_group = n_thresholds * sum(
        n_by_group[group] * (group_means[group] - grand_mean) ** 2
        for group in expected_groups
    )
    ss_subject_group = n_thresholds * sum(
        (subject_means[(group, subject_id)] - group_means[group]) ** 2
        for group in expected_groups
        for subject_id in data.loc[data["group"] == group, "subject_id"].unique()
    )
    ss_threshold = n_total_subjects * sum(
        (threshold_means[threshold_type] - grand_mean) ** 2
        for threshold_type in expected_threshold_types
    )
    ss_group_threshold = sum(
        n_by_group[group] * (
            group_threshold_means[(group, threshold_type)]
            - group_means[group]
            - threshold_means[threshold_type]
            + grand_mean
        ) ** 2
        for group in expected_groups
        for threshold_type in expected_threshold_types
    )
    ss_error = 0.0
    for row in data.itertuples(index=False):
        subject_mean = subject_means[(row.group, row.subject_id)]
        group_threshold_mean = group_threshold_means[(row.group, row.threshold_type)]
        group_mean = group_means[row.group]
        ss_error += (
            row.delta - subject_mean - group_threshold_mean + group_mean
        ) ** 2

    df_group = len(expected_groups) - 1
    df_subject_group = n_total_subjects - len(expected_groups)
    df_threshold = len(expected_threshold_types) - 1
    df_group_threshold = df_group * df_threshold
    df_error = df_subject_group * df_threshold

    ms_group = ss_group / df_group
    ms_subject_group = ss_subject_group / df_subject_group
    ms_threshold = ss_threshold / df_threshold
    ms_group_threshold = ss_group_threshold / df_group_threshold
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
            "effect": "Threshold type",
            "ss": ss_threshold,
            "df": df_threshold,
            "ms": ms_threshold,
            "error_term": "Threshold type x Subject(Group)",
            "error_df": df_error,
            "F": ms_threshold / ms_error,
            "p_value": stats.f.sf(ms_threshold / ms_error, df_threshold, df_error),
            "partial_eta_sq": ss_threshold / (ss_threshold + ss_error),
        },
        {
            "effect": "Group x Threshold type",
            "ss": ss_group_threshold,
            "df": df_group_threshold,
            "ms": ms_group_threshold,
            "error_term": "Threshold type x Subject(Group)",
            "error_df": df_error,
            "F": ms_group_threshold / ms_error,
            "p_value": stats.f.sf(ms_group_threshold / ms_error, df_group_threshold, df_error),
            "partial_eta_sq": ss_group_threshold / (ss_group_threshold + ss_error),
        },
    ]
    anova_table = pd.DataFrame(rows)

    cell_summary = (
        data.groupby(["group", "threshold_type"], observed=False)["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_delta", "std": "sd_delta", "count": "n_subjects"})
    )
    cell_summary["SEM"] = cell_summary["sd_delta"] / np.sqrt(cell_summary["n_subjects"])

    print("\nCell summary:")
    print(cell_summary.to_string(index=False))
    print("\nMixed-design ANOVA table:")
    print(anova_table.to_string(index=False))

    diagnostics = {
        "grand_mean_delta": float(grand_mean),
        "subjects_by_group": subjects_by_group,
        "n_subjects": int(n_total_subjects),
        "threshold_types": expected_threshold_types,
        "ss_subject_group": float(ss_subject_group),
        "df_subject_group": int(df_subject_group),
        "ss_error": float(ss_error),
        "df_error": int(df_error),
    }

    return {
        "anova_table": anova_table,
        "cell_summary": cell_summary,
        "diagnostics": diagnostics,
        "input_data": data,
    }


def load_and_run_bci_combined_threshold_change_mixed_anova(csv_path=None):
    """Load BCI CSV, compute combined threshold changes, and run mixed ANOVA."""
    threshold_results = load_and_plot_bci_combined_threshold_change_session1_to_session5(
        csv_path=csv_path,
        save=False,
    )
    anova_results = run_bci_combined_threshold_change_mixed_anova(
        threshold_results["combined_subject_threshold_change"]
    )
    return {
        **threshold_results,
        "anova_table": anova_results["anova_table"],
        "anova_cell_summary": anova_results["cell_summary"],
        "anova_diagnostics": anova_results["diagnostics"],
        "anova_input_data": anova_results["input_data"],
    }
