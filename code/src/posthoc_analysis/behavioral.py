"""Behavioral analyses for training-task reaction time, accuracy, and timeout outcomes."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from scipy import stats
except ImportError as exc:
    stats = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

try:
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
except ImportError as exc:
    ols = None
    anova_lm = None
    _STATSMODELS_IMPORT_ERROR = exc
else:
    _STATSMODELS_IMPORT_ERROR = None

from .config import PROJECT_ROOT, BCI_GROUP_SUBJECTS


REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = REPO_ROOT / "figures"


def _print_warning(message):
    """Print a standardized warning message."""
    print(f"WARNING: {message}")


def _require_statistical_dependencies(require_scipy=True, require_statsmodels=True):
    """Raise an informative error if optional stats dependencies are unavailable."""
    missing = []
    if require_scipy and stats is None:
        missing.append(f"scipy ({_SCIPY_IMPORT_ERROR})")
    if require_statsmodels and ols is None:
        missing.append(f"statsmodels ({_STATSMODELS_IMPORT_ERROR})")

    if missing:
        raise ModuleNotFoundError(
            "This analysis requires unavailable statistical dependencies: "
            + ", ".join(missing)
        )


def create_behavioral_summary_table(df):
    """Create subject-level behavioral summary table.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level DataFrame from consolidated CSV

    Returns
    -------
    pd.DataFrame
        Subject-level summary with one row per subject × group × session × trial_type
        Columns: subject_id, group, session, trial_type, n_total_trials,
                 n_correct, n_incorrect, n_timeouts, accuracy, timeout_rate
    """
    # Copy and prepare data
    df = df.copy()

    # Add group column
    df['group'] = df['subject_id'].apply(
        lambda x: 'experimental' if x in BCI_GROUP_SUBJECTS else 'control'
    )

    # Map session_id to pre/post
    df['session'] = df['session_id'].map({1: 'pre', 5: 'post'})

    # Map task to trial_type
    df['trial_type'] = df['task'].map({0: 'no_distractor', 1: 'distractor'})

    # Initialize results list
    results = []

    # Get unique combinations
    subjects = df['subject_id'].unique()
    groups = df['group'].unique()
    sessions = df['session'].unique()
    trial_types = ['distractor', 'no_distractor', 'combined']

    for subject in subjects:
        for group in groups:
            if df[(df['subject_id'] == subject) & (df['group'] == group)].empty:
                continue

            for session in sessions:
                # First compute distractor and no_distractor separately
                for trial_type in ['distractor', 'no_distractor']:
                    subset = df[
                        (df['subject_id'] == subject) &
                        (df['group'] == group) &
                        (df['session'] == session) &
                        (df['trial_type'] == trial_type)
                    ]

                    if subset.empty:
                        continue

                    # All trials for timeout analysis
                    n_total_trials = len(subset)
                    n_timeouts = (subset['feedback'] == 3).sum()
                    timeout_rate = n_timeouts / n_total_trials if n_total_trials > 0 else None

                    # Non-timeout trials for accuracy analysis
                    non_timeout = subset[subset['feedback'] != 3]
                    n_correct = (non_timeout['feedback'] == 1).sum()
                    n_incorrect = (non_timeout['feedback'] == 2).sum()
                    accuracy = n_correct / (n_correct + n_incorrect) if (n_correct + n_incorrect) > 0 else None

                    results.append({
                        'subject_id': subject,
                        'group': group,
                        'session': session,
                        'trial_type': trial_type,
                        'n_total_trials': n_total_trials,
                        'n_correct': n_correct,
                        'n_incorrect': n_incorrect,
                        'n_timeouts': n_timeouts,
                        'accuracy': accuracy,
                        'timeout_rate': timeout_rate
                    })

                # Now compute combined accuracy (across both trial types)
                combined_subset = df[
                    (df['subject_id'] == subject) &
                    (df['group'] == group) &
                    (df['session'] == session)
                ]

                if not combined_subset.empty:
                    # For combined: use all trials for timeout, non-timeout for accuracy
                    n_total_trials_combined = len(combined_subset)
                    n_timeouts_combined = (combined_subset['feedback'] == 3).sum()
                    timeout_rate_combined = n_timeouts_combined / n_total_trials_combined if n_total_trials_combined > 0 else None

                    # Accuracy across all non-timeout trials
                    combined_non_timeout = combined_subset[combined_subset['feedback'] != 3]
                    n_correct_combined = (combined_non_timeout['feedback'] == 1).sum()
                    n_incorrect_combined = (combined_non_timeout['feedback'] == 2).sum()
                    accuracy_combined = n_correct_combined / (n_correct_combined + n_incorrect_combined) if (n_correct_combined + n_incorrect_combined) > 0 else None

                    results.append({
                        'subject_id': subject,
                        'group': group,
                        'session': session,
                        'trial_type': 'combined',
                        'n_total_trials': n_total_trials_combined,
                        'n_correct': n_correct_combined,
                        'n_incorrect': n_incorrect_combined,
                        'n_timeouts': n_timeouts_combined,
                        'accuracy': accuracy_combined,
                        'timeout_rate': timeout_rate_combined
                    })

    return pd.DataFrame(results)


def validate_behavioral_summary(summary_df, original_df):
    """Validate the behavioral summary table.

    Parameters
    ----------
    summary_df : pd.DataFrame
        The summary table to validate
    original_df : pd.DataFrame
        Original trial-level data

    Returns
    -------
    dict
        Validation results with issues and summary stats
    """
    issues = []

    # Check for missing denominators
    zero_denominator_accuracy = summary_df[
        (summary_df['trial_type'] != 'combined') &
        ((summary_df['n_correct'] + summary_df['n_incorrect']) == 0)
    ]
    if not zero_denominator_accuracy.empty:
        issues.append(f"Found {len(zero_denominator_accuracy)} rows with zero accuracy denominator")

    zero_denominator_timeout = summary_df[summary_df['n_total_trials'] == 0]
    if not zero_denominator_timeout.empty:
        issues.append(f"Found {len(zero_denominator_timeout)} rows with zero timeout denominator")

    # Check consistency: n_correct + n_incorrect should equal n_total_trials - n_timeouts
    inconsistent_rows = summary_df[
        (summary_df['n_correct'] + summary_df['n_incorrect']) != (summary_df['n_total_trials'] - summary_df['n_timeouts'])
    ]
    if not inconsistent_rows.empty:
        issues.append(f"Found {len(inconsistent_rows)} rows with inconsistent trial counts")

    # Check expected structure
    expected_subjects = original_df['subject_id'].nunique()
    expected_sessions = original_df['session_id'].nunique()
    expected_groups = original_df['group'].nunique() if 'group' in original_df.columns else summary_df['group'].nunique()

    actual_subjects = summary_df['subject_id'].nunique()
    actual_sessions = summary_df['session'].nunique()
    actual_groups = summary_df['group'].nunique()

    if actual_subjects != expected_subjects:
        issues.append(f"Expected {expected_subjects} subjects, found {actual_subjects}")

    if actual_sessions != expected_sessions:
        issues.append(f"Expected {expected_sessions} sessions, found {actual_sessions}")

    if actual_groups != expected_groups:
        issues.append(f"Expected {expected_groups} groups, found {actual_groups}")

    # Check trial type distribution
    trial_type_counts = summary_df['trial_type'].value_counts()
    expected_trial_types = {'distractor': expected_subjects * expected_sessions,
                           'no_distractor': expected_subjects * expected_sessions,
                           'combined': expected_subjects * expected_sessions}

    for trial_type, expected_count in expected_trial_types.items():
        actual_count = trial_type_counts.get(trial_type, 0)
        if actual_count != expected_count:
            issues.append(f"Expected {expected_count} {trial_type} rows, found {actual_count}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'summary': {
            'total_rows': len(summary_df),
            'subjects': actual_subjects,
            'groups': actual_groups,
            'sessions': actual_sessions,
            'trial_types': dict(trial_type_counts),
            'accuracy_range': (summary_df['accuracy'].min(), summary_df['accuracy'].max()),
            'timeout_rate_range': (summary_df['timeout_rate'].min(), summary_df['timeout_rate'].max())
        }
    }


def print_behavioral_summary_checks(summary_df, original_df):
    """Print sanity checks for the behavioral summary.

    Parameters
    ----------
    summary_df : pd.DataFrame
        The behavioral summary table
    original_df : pd.DataFrame
        Original trial-level data
    """
    print("=" * 80)
    print("BEHAVIORAL SUMMARY SANITY CHECKS")
    print("=" * 80)

    # Counts per subject/session/trial_type
    print("\nCounts per subject/session/trial_type:")
    count_summary = summary_df.groupby(['subject_id', 'session', 'trial_type']).size().reset_index(name='count')
    print(count_summary.pivot_table(
        index=['subject_id', 'session'],
        columns='trial_type',
        values='count',
        fill_value=0
    ))

    # Check for missing denominators
    print("\nRows with zero accuracy denominators:")
    zero_acc = summary_df[
        (summary_df['trial_type'] != 'combined') &
        ((summary_df['n_correct'] + summary_df['n_incorrect']) == 0)
    ]
    if zero_acc.empty:
        print("None found")
    else:
        print(zero_acc[['subject_id', 'session', 'trial_type', 'n_total_trials', 'n_timeouts']])

    print("\nRows with zero timeout denominators:")
    zero_timeout = summary_df[summary_df['n_total_trials'] == 0]
    if zero_timeout.empty:
        print("None found")
    else:
        print(zero_timeout[['subject_id', 'session', 'trial_type']])

    # Check pre/post existence
    print("\nPre/post session coverage:")
    session_coverage = summary_df.groupby(['subject_id', 'group', 'trial_type'])['session'].nunique()
    incomplete_coverage = session_coverage[session_coverage < 2]
    if incomplete_coverage.empty:
        print("All subjects have both pre and post sessions for all trial types")
    else:
        print("Subjects missing pre or post sessions:")
        print(incomplete_coverage.reset_index())

    # Summary stats
    print("\nSummary statistics:")
    print(f"Total rows: {len(summary_df)}")
    print(f"Subjects: {summary_df['subject_id'].nunique()}")
    print(f"Groups: {summary_df['group'].nunique()}")
    print(f"Sessions: {summary_df['session'].nunique()}")
    print(f"Accuracy range: {summary_df['accuracy'].min():.3f} - {summary_df['accuracy'].max():.3f}")
    print(f"Timeout rate range: {summary_df['timeout_rate'].min():.3f} - {summary_df['timeout_rate'].max():.3f}")

    print("=" * 80)


def load_and_summarize_behavioral_data(csv_path=None):
    """Load consolidated CSV and create behavioral summary table.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to consolidated CSV. Defaults to standard location.

    Returns
    -------
    dict
        Keys: 'summary_table', 'validation', 'original_data'
    """
    if csv_path is None:
        csv_path = Path(
            '/Users/hililbby/Library/CloudStorage/Box-Box/CNBI/Attention_distraction'
            '/project_healthy/analyses/all_subjects_training.csv'
        )

    # Load data
    original_df = pd.read_csv(csv_path)

    # Create summary
    summary_table = create_behavioral_summary_table(original_df)

    # Validate
    validation = validate_behavioral_summary(summary_table, original_df)

    return {
        'summary_table': summary_table,
        'validation': validation,
        'original_data': original_df
    }


def analyze_stroop_timeout_exclusions(df):
    """ANALYSIS 1: Summarize Stroop timeout exclusions.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level consolidated Stroop DataFrame.

    Returns
    -------
    dict
        Keys: 'trial_data', 'run_level_summary', 'subject_session_summary',
        'overall_summary', 'warnings'
    """
    print("=" * 80)
    print("STROOP ANALYSIS 1: TIMEOUT SUMMARY")
    print("=" * 80)

    data, warnings_list = _prepare_stroop_timeout_data(df)
    run_level_summary = _build_stroop_timeout_run_summary(data)
    subject_session_summary = _build_stroop_timeout_subject_session_summary(run_level_summary)
    overall_summary = _build_stroop_timeout_overall_summary(subject_session_summary)

    _print_stroop_timeout_checks(
        data,
        run_level_summary,
        subject_session_summary,
        overall_summary,
    )

    print("\nStroop Analysis 1 complete!")
    print("=" * 80)

    return {
        "trial_data": data,
        "run_level_summary": run_level_summary,
        "subject_session_summary": subject_session_summary,
        "overall_summary": overall_summary,
        "warnings": warnings_list,
    }


def load_and_analyze_stroop_timeout_data(csv_path=None):
    """Load the consolidated Stroop CSV and run Analysis 1 timeout summaries."""
    if csv_path is None:
        csv_path = Path(
            '/Users/hililbby/Library/CloudStorage/Box-Box/CNBI/Attention_distraction'
            '/project_healthy/analyses/all_subjects_stroop.csv'
        )

    original_df = pd.read_csv(csv_path)
    results = analyze_stroop_timeout_exclusions(original_df)
    results["original_data"] = original_df
    results["csv_path"] = str(csv_path)
    return results


def create_stroop_accuracy_summary_table(df):
    """Create subject-level Stroop accuracy summaries after excluding timeout trials."""
    required_columns = {
        "subject_id",
        "group",
        "session_id",
        "trial_type",
        "response",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Stroop accuracy summary requires these columns: "
            f"{missing_columns}"
        )

    data = df.copy()
    data["session"] = data["session_id"].map({1: "pre", 5: "post"})
    data["trial_type"] = data["trial_type"].astype(str).str.lower().str.strip()

    # Keep all trials for count reporting, but exclude timeout trials from accuracy denominator.
    results = []

    for (subject_id, group, session, trial_type), subset in data.groupby(
        ["subject_id", "group", "session", "trial_type"],
        observed=False,
    ):
        n_total_trials = int(len(subset))
        n_timeouts = int((subset["response"] == 3).sum())
        non_timeout = subset[subset["response"] != 3]
        n_correct = int((non_timeout["response"] == 1).sum())
        n_incorrect = int((non_timeout["response"] == 2).sum())
        denominator = n_correct + n_incorrect
        accuracy = n_correct / denominator if denominator > 0 else np.nan

        results.append({
            "subject_id": subject_id,
            "group": group,
            "session": session,
            "trial_type": trial_type,
            "n_total_trials": n_total_trials,
            "n_timeouts": n_timeouts,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "accuracy": accuracy,
        })

    # Add overall subject-session rows collapsed across congruent/incongruent.
    for (subject_id, group, session), subset in data.groupby(
        ["subject_id", "group", "session"],
        observed=False,
    ):
        n_total_trials = int(len(subset))
        n_timeouts = int((subset["response"] == 3).sum())
        non_timeout = subset[subset["response"] != 3]
        n_correct = int((non_timeout["response"] == 1).sum())
        n_incorrect = int((non_timeout["response"] == 2).sum())
        denominator = n_correct + n_incorrect
        accuracy = n_correct / denominator if denominator > 0 else np.nan

        results.append({
            "subject_id": subject_id,
            "group": group,
            "session": session,
            "trial_type": "overall",
            "n_total_trials": n_total_trials,
            "n_timeouts": n_timeouts,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "accuracy": accuracy,
        })

    return pd.DataFrame(results)


def validate_stroop_accuracy_summary(summary_df, original_df):
    """Validate the Stroop subject-level accuracy summary."""
    issues = []

    expected_subjects = original_df["subject_id"].nunique()
    actual_subjects = summary_df["subject_id"].nunique()
    if actual_subjects != expected_subjects:
        issues.append(f"Expected {expected_subjects} subjects, found {actual_subjects}")

    expected_sessions = set(original_df["session_id"].map({1: "pre", 5: "post"}).dropna())
    actual_sessions = set(summary_df["session"].dropna())
    if actual_sessions != expected_sessions:
        issues.append(
            f"Expected sessions {sorted(expected_sessions)}, found {sorted(actual_sessions)}"
        )

    expected_trial_types = {"congruent", "incongruent", "overall"}
    actual_trial_types = set(summary_df["trial_type"].dropna())
    if actual_trial_types != expected_trial_types:
        issues.append(
            f"Expected trial types {sorted(expected_trial_types)}, found {sorted(actual_trial_types)}"
        )

    inconsistent_rows = summary_df[
        (summary_df["n_correct"] + summary_df["n_incorrect"]) !=
        (summary_df["n_total_trials"] - summary_df["n_timeouts"])
    ]
    if not inconsistent_rows.empty:
        issues.append(f"Found {len(inconsistent_rows)} rows with inconsistent counts")

    missing_accuracy = summary_df[summary_df["accuracy"].isna()]
    if not missing_accuracy.empty:
        issues.append(f"Found {len(missing_accuracy)} rows with missing accuracy")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "summary": {
            "total_rows": len(summary_df),
            "subjects": actual_subjects,
            "sessions": sorted(actual_sessions),
            "trial_types": sorted(actual_trial_types),
            "accuracy_range": (
                float(summary_df["accuracy"].min()),
                float(summary_df["accuracy"].max()),
            ),
        },
    }


def analyze_stroop_accuracy(df):
    """ANALYSIS 2: Compute Stroop accuracy summaries, plots, and statistics."""
    print("=" * 80)
    print("STROOP ANALYSIS 2: ACCURACY")
    print("=" * 80)

    data, warnings_list = _prepare_stroop_accuracy_data(df)
    summary_table = create_stroop_accuracy_summary_table(data)
    validation = validate_stroop_accuracy_summary(summary_table, data)
    if not validation["valid"]:
        for issue in validation["issues"]:
            _print_warning(issue)

    _print_stroop_accuracy_checks(summary_table)

    congruent_accuracy = summary_table[summary_table["trial_type"] == "congruent"].copy()
    incongruent_accuracy = summary_table[summary_table["trial_type"] == "incongruent"].copy()
    overall_accuracy = summary_table[summary_table["trial_type"] == "overall"].copy()

    stats_results = {}
    if ols is None or anova_lm is None:
        warning = (
            "Statsmodels/SciPy are unavailable in the current environment, so "
            "Stroop accuracy ANOVAs were not executed. Summary tables and figures "
            "were still generated."
        )
        warnings_list.append(warning)
        _print_warning(warning)
        stats_results = {
            "congruent": None,
            "incongruent": None,
            "overall": None,
        }
    else:
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSES")
        print("=" * 80)
        stats_results = {
            "congruent": _run_stroop_accuracy_anova(congruent_accuracy, "congruent"),
            "incongruent": _run_stroop_accuracy_anova(incongruent_accuracy, "incongruent"),
            "overall": _run_stroop_accuracy_anova(overall_accuracy, "overall"),
        }

    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    figures = _plot_stroop_accuracy_figures(
        congruent_accuracy,
        incongruent_accuracy,
        overall_accuracy,
    )

    print("\nStroop Analysis 2 complete!")
    print("=" * 80)

    return {
        "summary_table": summary_table,
        "validation": validation,
        "congruent_accuracy": congruent_accuracy,
        "incongruent_accuracy": incongruent_accuracy,
        "overall_accuracy": overall_accuracy,
        "stats": stats_results,
        "figures": figures,
        "warnings": warnings_list,
    }


def load_and_analyze_stroop_accuracy_data(csv_path=None):
    """Load the consolidated Stroop CSV and run Analysis 2 accuracy summaries."""
    if csv_path is None:
        csv_path = Path(
            '/Users/hililbby/Library/CloudStorage/Box-Box/CNBI/Attention_distraction'
            '/project_healthy/analyses/all_subjects_stroop.csv'
        )

    original_df = pd.read_csv(csv_path)
    results = analyze_stroop_accuracy(original_df)
    results["original_data"] = original_df
    results["csv_path"] = str(csv_path)
    return results


def analyze_stroop_reaction_time(df):
    """ANALYSIS 3: Compute Stroop reaction-time summaries, plots, and statistics."""
    print("=" * 80)
    print("STROOP ANALYSIS 3: REACTION TIME")
    print("=" * 80)

    data, warnings_list, incorrect_trial_summary = _prepare_stroop_reaction_time_data(df)
    trial_data_clean, outlier_report = _remove_stroop_rt_outliers(data)
    subject_exclusion_summary = _summarize_stroop_rt_exclusions_by_subject(
        incorrect_trial_summary,
        outlier_report,
    )
    overall_exclusion_summary = _summarize_stroop_rt_exclusions_overall(
        subject_exclusion_summary
    )
    subject_session_summary = _aggregate_stroop_reaction_times(trial_data_clean)

    _print_stroop_reaction_time_checks(
        data,
        trial_data_clean,
        incorrect_trial_summary,
        outlier_report,
        subject_exclusion_summary,
        overall_exclusion_summary,
        subject_session_summary,
    )

    congruent_rt = subject_session_summary[
        subject_session_summary["trial_type"] == "congruent"
    ].copy()
    incongruent_rt = subject_session_summary[
        subject_session_summary["trial_type"] == "incongruent"
    ].copy()
    overall_rt = subject_session_summary[
        subject_session_summary["trial_type"] == "overall"
    ].copy()

    if ols is None or anova_lm is None or stats is None:
        warning = (
            "Statsmodels/SciPy are unavailable in the current environment, so "
            "Stroop RT ANOVAs were not executed. Summary tables and figures were "
            "still generated."
        )
        warnings_list.append(warning)
        _print_warning(warning)
        stats_results = {
            "congruent": None,
            "incongruent": None,
            "overall": None,
        }
    else:
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSES")
        print("=" * 80)
        stats_results = {
            "congruent": _run_stroop_rt_anova(congruent_rt, "congruent"),
            "incongruent": _run_stroop_rt_anova(incongruent_rt, "incongruent"),
            "overall": _run_stroop_rt_anova(overall_rt, "overall"),
        }

    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    figures = _plot_stroop_reaction_time_figures(subject_session_summary, data)

    print("\nStroop Analysis 3 complete!")
    print("=" * 80)

    return {
        "trial_data_before_outlier_removal": data,
        "trial_data_clean": trial_data_clean,
        "incorrect_trial_summary": incorrect_trial_summary,
        "outlier_report": outlier_report,
        "subject_exclusion_summary": subject_exclusion_summary,
        "overall_exclusion_summary": overall_exclusion_summary,
        "subject_session_summary": subject_session_summary,
        "congruent_rt": congruent_rt,
        "incongruent_rt": incongruent_rt,
        "overall_rt": overall_rt,
        "stats": stats_results,
        "figures": figures,
        "warnings": warnings_list,
    }


def load_and_analyze_stroop_reaction_time_data(csv_path=None):
    """Load the consolidated Stroop CSV and run Analysis 3 RT summaries."""
    if csv_path is None:
        csv_path = Path(
            '/Users/hililbby/Library/CloudStorage/Box-Box/CNBI/Attention_distraction'
            '/project_healthy/analyses/all_subjects_stroop.csv'
        )

    original_df = pd.read_csv(csv_path)
    results = analyze_stroop_reaction_time(original_df)
    results["original_data"] = original_df
    results["csv_path"] = str(csv_path)
    return results


def analyze_stroop_effect(df=None, rt_results=None):
    """ANALYSIS 4: Compute Stroop-effect RT cost with plots and statistics.

    Parameters
    ----------
    df : pandas.DataFrame or None
        Raw consolidated Stroop trial-level data. Used only when `rt_results`
        is not supplied.
    rt_results : dict or None
        Previously computed results from `analyze_stroop_reaction_time`.
        When provided, Analysis 4 reuses the already cleaned RT data and
        outlier-removal summaries rather than rerunning Analysis 3.
    """
    print("=" * 80)
    print("STROOP ANALYSIS 4: STROOP EFFECT")
    print("=" * 80)

    if rt_results is not None:
        required_keys = {
            "trial_data_before_outlier_removal",
            "trial_data_clean",
            "incorrect_trial_summary",
            "outlier_report",
            "subject_exclusion_summary",
            "overall_exclusion_summary",
            "subject_session_summary",
            "warnings",
        }
        missing_keys = sorted(required_keys - set(rt_results.keys()))
        if missing_keys:
            raise ValueError(
                "Provided Stroop RT results are missing required keys for "
                f"Stroop-effect analysis: {missing_keys}"
            )
        print("Reusing cleaned Stroop RT data from Analysis 3; SD-based outliers are not removed again.")
    else:
        if df is None:
            raise ValueError(
                "Stroop-effect analysis requires either raw `df` input or "
                "previous `rt_results` from analyze_stroop_reaction_time."
            )
        rt_results = analyze_stroop_reaction_time(df)

    subject_session_summary = rt_results["subject_session_summary"]
    subject_session_stroop_effect = _compute_subject_stroop_effect(subject_session_summary)

    _print_stroop_effect_checks(
        rt_results["incorrect_trial_summary"],
        rt_results["subject_exclusion_summary"],
        rt_results["overall_exclusion_summary"],
        subject_session_stroop_effect,
    )

    cell_summary = (
        subject_session_stroop_effect
        .groupby(["group", "session"], observed=False)["stroop_effect_ms"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    cell_summary["sem"] = cell_summary["std"] / np.sqrt(cell_summary["count"])
    if stats is not None:
        cell_summary["ci95_low"] = (
            cell_summary["mean"] -
            stats.t.ppf(0.975, cell_summary["count"] - 1) * cell_summary["sem"]
        )
        cell_summary["ci95_high"] = (
            cell_summary["mean"] +
            stats.t.ppf(0.975, cell_summary["count"] - 1) * cell_summary["sem"]
        )
    else:
        cell_summary["ci95_low"] = np.nan
        cell_summary["ci95_high"] = np.nan

    print("\nStroop-effect cell means (ms):")
    print(cell_summary.to_string(index=False))

    if ols is None or anova_lm is None or stats is None:
        warning = (
            "Statsmodels/SciPy are unavailable in the current environment, so "
            "Stroop-effect inferential statistics were not executed. Summary tables "
            "and figures were still generated."
        )
        warnings = list(rt_results["warnings"]) + [warning]
        _print_warning(warning)
        stats_anova = None
        posthoc_tests = None
    else:
        warnings = list(rt_results["warnings"])
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSES")
        print("=" * 80)
        stats_anova = _run_stroop_effect_anova(subject_session_stroop_effect)
        posthoc_tests = _run_stroop_effect_posthocs(subject_session_stroop_effect)

    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    figures = _plot_stroop_effect_figures(subject_session_stroop_effect)

    print("\nStroop Analysis 4 complete!")
    print("=" * 80)

    return {
        "trial_data_before_outlier_removal": rt_results["trial_data_before_outlier_removal"],
        "trial_data_clean": rt_results["trial_data_clean"],
        "incorrect_trial_summary": rt_results["incorrect_trial_summary"],
        "outlier_report": rt_results["outlier_report"],
        "subject_exclusion_summary": rt_results["subject_exclusion_summary"],
        "overall_exclusion_summary": rt_results["overall_exclusion_summary"],
        "subject_session_rt_summary": subject_session_summary,
        "subject_session_stroop_effect": subject_session_stroop_effect,
        "cell_summary": cell_summary,
        "stats_anova": stats_anova,
        "posthoc_tests": posthoc_tests,
        "figures": figures,
        "warnings": warnings,
    }


def load_and_analyze_stroop_effect_data(csv_path=None, rt_results=None):
    """Load the consolidated Stroop CSV and run Analysis 4 Stroop-effect summaries.

    If `rt_results` is supplied, the function reuses the cleaned Analysis 3
    output and skips reloading/recleaning the raw CSV.
    """
    if rt_results is not None:
        results = analyze_stroop_effect(rt_results=rt_results)
        results["csv_path"] = str(csv_path) if csv_path is not None else None
        return results

    if csv_path is None:
        csv_path = Path(
            '/Users/hililbby/Library/CloudStorage/Box-Box/CNBI/Attention_distraction'
            '/project_healthy/analyses/all_subjects_stroop.csv'
        )

    original_df = pd.read_csv(csv_path)
    results = analyze_stroop_effect(df=original_df)
    results["original_data"] = original_df
    results["csv_path"] = str(csv_path)
    return results


def _prepare_stroop_timeout_data(df):
    """Validate and prepare Stroop trial-level data for timeout analysis."""
    required_columns = {
        "subject_id",
        "group",
        "session_id",
        "run_id",
        "trial_number",
        "trial_type",
        "response",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Stroop timeout analysis is missing required columns: "
            f"{missing_columns}"
        )

    data = df.copy()
    warnings_list = []

    session_map = {1: "pre", 5: "post"}
    invalid_sessions = sorted(set(data["session_id"].dropna()) - set(session_map))
    if invalid_sessions:
        warning = (
            "Found session IDs outside the expected Stroop pre/post mapping: "
            f"{invalid_sessions}"
        )
        warnings_list.append(warning)
        _print_warning(warning)

    valid_trial_types = {"congruent", "incongruent"}
    observed_trial_types = set(data["trial_type"].dropna().astype(str).str.lower().str.strip())
    invalid_trial_types = sorted(observed_trial_types - valid_trial_types)
    if invalid_trial_types:
        warning = f"Found unexpected Stroop trial types: {invalid_trial_types}"
        warnings_list.append(warning)
        _print_warning(warning)

    valid_responses = {1, 2, 3}
    invalid_responses = sorted(set(data["response"].dropna()) - valid_responses)
    if invalid_responses:
        warning = f"Found unexpected Stroop response codes: {invalid_responses}"
        warnings_list.append(warning)
        _print_warning(warning)

    subject_count = int(data["subject_id"].nunique())
    if subject_count != 32:
        warning = (
            f"Consolidated Stroop CSV contains {subject_count} subjects rather "
            "than the full 32-subject study roster. Proceeding with available data."
        )
        warnings_list.append(warning)
        _print_warning(warning)

    data["session"] = data["session_id"].map(session_map)
    data["trial_type"] = data["trial_type"].astype(str).str.lower().str.strip()

    duplicate_rows = data.duplicated(
        subset=["subject_id", "session_id", "run_id", "trial_number"]
    )
    if duplicate_rows.any():
        raise ValueError(
            "Found duplicated Stroop rows for subject/session/run/trial combinations."
        )

    run_counts = (
        data.groupby(["subject_id", "session_id"], observed=False)["run_id"]
        .nunique()
        .reset_index(name="n_runs")
    )
    run_count_issues = run_counts[run_counts["n_runs"] != 2]
    if not run_count_issues.empty:
        warning = (
            "Some subject/session combinations do not have the documented 2 Stroop runs."
        )
        warnings_list.append(warning)
        _print_warning(warning)
        print(run_count_issues.to_string(index=False))

    trial_counts = (
        data.groupby(["subject_id", "session_id", "run_id"], observed=False)
        .size()
        .reset_index(name="n_trials")
    )
    invalid_trial_counts = trial_counts[trial_counts["n_trials"] != 60]
    if not invalid_trial_counts.empty:
        warning = "Some included Stroop runs do not contain 60 trials."
        warnings_list.append(warning)
        _print_warning(warning)
        print(invalid_trial_counts.to_string(index=False))

    print("\nStroop timeout-analysis input summary:")
    print(f"Total rows: {len(data)}")
    print(f"Subjects: {data['subject_id'].nunique()}")
    print(f"Sessions: {sorted(data['session_id'].dropna().unique().tolist())}")
    print(
        "Response-code counts: "
        f"{data['response'].value_counts(dropna=False).sort_index().to_dict()}"
    )

    return data, warnings_list


def _build_stroop_timeout_run_summary(data):
    """Summarize timeout counts and percentages for each Stroop run."""
    run_level_summary = (
        data.groupby(
            ["subject_id", "group", "session_id", "session", "run_id"],
            observed=False,
        )
        .agg(
            n_total_trials=("trial_number", "size"),
            n_timeout_trials=("response", lambda x: int((x == 3).sum())),
        )
        .reset_index()
    )
    run_level_summary["percent_timeout_trials"] = (
        run_level_summary["n_timeout_trials"] /
        run_level_summary["n_total_trials"] * 100.0
    )
    return run_level_summary


def _build_stroop_timeout_subject_session_summary(run_level_summary):
    """Average timeout percentages across runs within each subject-session."""
    subject_session_summary = (
        run_level_summary.groupby(
            ["subject_id", "group", "session_id", "session"],
            observed=False,
        )
        .agg(
            n_runs=("run_id", "nunique"),
            mean_timeout_percent_across_runs=("percent_timeout_trials", "mean"),
            std_timeout_percent_across_runs=("percent_timeout_trials", "std"),
            mean_timeout_trials_across_runs=("n_timeout_trials", "mean"),
            total_timeout_trials=("n_timeout_trials", "sum"),
            total_trials=("n_total_trials", "sum"),
        )
        .reset_index()
    )
    subject_session_summary["std_timeout_percent_across_runs"] = (
        subject_session_summary["std_timeout_percent_across_runs"].fillna(0.0)
    )
    subject_session_summary["pooled_timeout_percent"] = (
        subject_session_summary["total_timeout_trials"] /
        subject_session_summary["total_trials"] * 100.0
    )
    return subject_session_summary


def _build_stroop_timeout_overall_summary(subject_session_summary):
    """Compute the overall average timeout percentage across subject-sessions."""
    overall_mean = float(
        subject_session_summary["mean_timeout_percent_across_runs"].mean()
    )
    overall_std = float(
        subject_session_summary["mean_timeout_percent_across_runs"].std(ddof=1)
    )
    n_subject_sessions = int(len(subject_session_summary))
    overall_sem = (
        overall_std / np.sqrt(n_subject_sessions)
        if n_subject_sessions > 1 else np.nan
    )

    by_group_session = (
        subject_session_summary.groupby(["group", "session"], observed=False)
        ["mean_timeout_percent_across_runs"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "mean_timeout_percent",
            "std": "std_timeout_percent",
            "count": "n_subject_sessions",
        })
    )
    by_group_session["sem_timeout_percent"] = (
        by_group_session["std_timeout_percent"] /
        np.sqrt(by_group_session["n_subject_sessions"])
    )

    return {
        "overall_average_timeout_percent": overall_mean,
        "overall_std_timeout_percent": overall_std,
        "overall_sem_timeout_percent": overall_sem,
        "n_subject_sessions": n_subject_sessions,
        "by_group_session": by_group_session,
    }


def _print_stroop_timeout_checks(
    data,
    run_level_summary,
    subject_session_summary,
    overall_summary,
):
    """Print sanity checks for Stroop timeout analysis."""
    print("\nRun-level trial counts:")
    print(
        run_level_summary[
            [
                "subject_id",
                "session",
                "run_id",
                "n_total_trials",
                "n_timeout_trials",
                "percent_timeout_trials",
            ]
        ].to_string(index=False)
    )

    print("\nSubject-session timeout summary (averaged across runs):")
    print(
        subject_session_summary[
            [
                "subject_id",
                "group",
                "session",
                "n_runs",
                "mean_timeout_percent_across_runs",
                "pooled_timeout_percent",
            ]
        ].to_string(index=False)
    )

    print("\nOverall timeout exclusion summary:")
    print(
        "Average percent timeout trials removed per subject-session "
        f"(averaged across runs): "
        f"{overall_summary['overall_average_timeout_percent']:.3f}%"
    )
    print(
        "SEM across subject-sessions: "
        f"{overall_summary['overall_sem_timeout_percent']:.3f}%"
    )
    print(f"Subject-session count: {overall_summary['n_subject_sessions']}")

    print("\nGrouped timeout summary by group and session:")
    print(overall_summary["by_group_session"].to_string(index=False))


def _prepare_stroop_accuracy_data(df):
    """Validate and prepare Stroop trial-level data for accuracy analysis."""
    required_columns = {
        "subject_id",
        "group",
        "session_id",
        "run_id",
        "trial_number",
        "trial_type",
        "response",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Stroop accuracy analysis is missing required columns: "
            f"{missing_columns}"
        )

    data = df.copy()
    warnings_list = []

    subject_count = int(data["subject_id"].nunique())
    if subject_count != 32:
        warning = (
            f"Consolidated Stroop CSV contains {subject_count} subjects rather "
            "than the full 32-subject study roster. Proceeding with available data."
        )
        warnings_list.append(warning)
        _print_warning(warning)

    data["session"] = data["session_id"].map({1: "pre", 5: "post"})
    data["trial_type"] = data["trial_type"].astype(str).str.lower().str.strip()

    invalid_trial_types = sorted(
        set(data["trial_type"].dropna()) - {"congruent", "incongruent"}
    )
    if invalid_trial_types:
        warning = f"Found unexpected Stroop trial types: {invalid_trial_types}"
        warnings_list.append(warning)
        _print_warning(warning)

    invalid_responses = sorted(set(data["response"].dropna()) - {1, 2, 3})
    if invalid_responses:
        warning = f"Found unexpected Stroop response codes: {invalid_responses}"
        warnings_list.append(warning)
        _print_warning(warning)

    timeout_count = int((data["response"] == 3).sum())
    print("\nStroop accuracy-analysis input summary:")
    print(f"Total rows: {len(data)}")
    print(f"Timeout rows excluded from accuracy denominator: {timeout_count}")

    return data, warnings_list


def _print_stroop_accuracy_checks(summary_table):
    """Print sanity checks for the Stroop accuracy summary."""
    print("\nAccuracy summary by group, session, and trial type:")
    grouped = (
        summary_table.groupby(["group", "session", "trial_type"], observed=False)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    print(grouped.to_string(index=False))

    print("\nSubject-session accuracy summary:")
    print(
        summary_table[
            [
                "subject_id",
                "group",
                "session",
                "trial_type",
                "n_correct",
                "n_incorrect",
                "n_timeouts",
                "accuracy",
            ]
        ].to_string(index=False)
    )


def _run_stroop_accuracy_anova(summary_df, label):
    """Run a subject-adjusted two-factor ANOVA on Stroop accuracy."""
    anova_data = summary_df.copy()
    formula = "accuracy ~ C(subject_id) + C(session) * C(group)"
    model = ols(formula, data=anova_data).fit()
    anova_table = anova_lm(model, typ=2)

    residual_ss = anova_table.loc["Residual", "sum_sq"]
    anova_report = anova_table.reset_index().rename(columns={"index": "effect"})
    anova_report["partial_eta_sq"] = np.where(
        anova_report["effect"] != "Residual",
        anova_report["sum_sq"] / (anova_report["sum_sq"] + residual_ss),
        np.nan,
    )

    print(f"\n--- {label.title()} Accuracy: 2-Way ANOVA (Time × Group) ---")
    print(anova_report.to_string(index=False))

    return {
        "model": model,
        "anova_table": anova_table,
        "anova_report": anova_report,
        "r_squared": model.rsquared,
    }


def _plot_stroop_accuracy_figures(
    congruent_accuracy,
    incongruent_accuracy,
    overall_accuracy,
):
    """Generate publication-quality Stroop accuracy plots matching training style."""
    figures = {}

    with plt.rc_context(_publication_style_rcparams()):
        figures["accuracy_congruent"] = _plot_stroop_accuracy_panel(
            congruent_accuracy,
            title="Congruent Stroop Accuracy",
            y_label="Accuracy",
            filename_stem="stroop_accuracy_congruent",
        )
        figures["accuracy_incongruent"] = _plot_stroop_accuracy_panel(
            incongruent_accuracy,
            title="Incongruent Stroop Accuracy",
            y_label="Accuracy",
            filename_stem="stroop_accuracy_incongruent",
        )
        figures["accuracy_overall"] = _plot_stroop_accuracy_panel(
            overall_accuracy,
            title="Overall Stroop Accuracy",
            y_label="Overall Accuracy",
            filename_stem="stroop_accuracy_overall",
        )

    return figures


def _plot_stroop_accuracy_panel(data, title, y_label, filename_stem):
    """Plot a single pre/post Stroop accuracy panel in the training-analysis style."""
    colors = {'control': '#4C72B0', 'experimental': '#DD8452'}
    marker_color = '#333333'

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    panel_values = []

    for group in ['control', 'experimental']:
        group_data = data[data['group'] == group]
        pre_data = group_data[group_data['session'] == 'pre']['accuracy'].values
        post_data = group_data[group_data['session'] == 'post']['accuracy'].values

        if len(pre_data) == 0 or len(post_data) == 0:
            continue

        pre_mean = np.mean(pre_data)
        post_mean = np.mean(post_data)
        pre_sem = np.std(pre_data, ddof=1) / np.sqrt(len(pre_data))
        post_sem = np.std(post_data, ddof=1) / np.sqrt(len(post_data))
        panel_values.extend([
            pre_mean - pre_sem,
            pre_mean + pre_sem,
            post_mean - post_sem,
            post_mean + post_sem,
        ])

        ax.errorbar(
            [0, 1],
            [pre_mean, post_mean],
            yerr=[pre_sem, post_sem],
            marker='o',
            markersize=4,
            color=colors[group],
            capsize=3,
            capthick=1,
            linewidth=1.0,
            label=group.capitalize(),
        )

        for subj_id in group_data['subject_id'].unique():
            subj_pre = group_data[
                (group_data['subject_id'] == subj_id) &
                (group_data['session'] == 'pre')
            ]['accuracy'].values
            subj_post = group_data[
                (group_data['subject_id'] == subj_id) &
                (group_data['session'] == 'post')
            ]['accuracy'].values
            if len(subj_pre) > 0 and len(subj_post) > 0:
                panel_values.extend([subj_pre[0], subj_post[0]])
                ax.plot(
                    [0, 1],
                    [subj_pre[0], subj_post[0]],
                    color=colors[group],
                    alpha=0.25,
                    linewidth=0.8,
                )
                ax.scatter(
                    [0, 1],
                    [subj_pre[0], subj_post[0]],
                    color=marker_color,
                    s=8,
                    alpha=0.5,
                    zorder=2,
                )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre', 'Post'])
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='both', length=3, width=0.8)
    _set_y_limits_with_padding(ax, panel_values, pad_fraction=0.12, min_pad=0.003)
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        handlelength=1.8,
    )
    fig.tight_layout(rect=[0, 0, 0.76, 1])
    _save_figure_pdf(fig, filename_stem)
    return fig


def _prepare_stroop_reaction_time_data(df):
    """Validate and prepare Stroop trial-level data for RT analysis."""
    required_columns = {
        "subject_id",
        "group",
        "session_id",
        "run_id",
        "trial_number",
        "trial_type",
        "response",
        "reaction_time_ms",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Stroop RT analysis is missing required columns: "
            f"{missing_columns}"
        )

    data = df.copy()
    warnings_list = []
    data["session"] = data["session_id"].map({1: "pre", 5: "post"})
    data["trial_type"] = data["trial_type"].astype(str).str.lower().str.strip()

    subject_count = int(data["subject_id"].nunique())
    if subject_count != 32:
        warning = (
            f"Consolidated Stroop CSV contains {subject_count} subjects rather "
            "than the full 32-subject study roster. Proceeding with available data."
        )
        warnings_list.append(warning)
        _print_warning(warning)

    invalid_trial_types = sorted(
        set(data["trial_type"].dropna()) - {"congruent", "incongruent"}
    )
    if invalid_trial_types:
        warning = f"Found unexpected Stroop trial types: {invalid_trial_types}"
        warnings_list.append(warning)
        _print_warning(warning)

    invalid_responses = sorted(set(data["response"].dropna()) - {1, 2, 3})
    if invalid_responses:
        warning = f"Found unexpected Stroop response codes: {invalid_responses}"
        warnings_list.append(warning)
        _print_warning(warning)

    incorrect_trials = int((data["response"] == 2).sum())
    total_trials = int(len(data))
    incorrect_trial_summary = {
        "n_total_trials": total_trials,
        "n_incorrect_trials": incorrect_trials,
        "percent_incorrect_trials": incorrect_trials / total_trials * 100.0,
    }

    included = data[
        (data["response"] == 1) &
        (data["reaction_time_ms"].notna()) &
        (data["session"].notna()) &
        (data["trial_type"].isin(["congruent", "incongruent"]))
    ].copy()

    print("\nStroop RT-analysis input summary:")
    print(f"Total rows: {len(data)}")
    print(f"Correct-response rows retained before outlier removal: {len(included)}")
    print(
        "Overall incorrect-trial percent across all subjects: "
        f"{incorrect_trial_summary['percent_incorrect_trials']:.3f}%"
    )

    return included, warnings_list, incorrect_trial_summary


def _remove_stroop_rt_outliers(trial_data):
    """Remove Stroop RT outliers using mean +/- 3 SD within each run."""
    grouped = (
        trial_data
        .groupby(
            ["subject_id", "group", "session", "session_id", "run_id"],
            observed=False,
        )["reaction_time_ms"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "run_mean_rt_ms",
            "std": "run_sd_rt_ms",
            "count": "n_trials_before",
        })
    )

    merged = trial_data.merge(
        grouped,
        on=["subject_id", "group", "session", "session_id", "run_id"],
        how="left",
    )

    merged["is_outlier"] = False
    valid_sd_mask = merged["run_sd_rt_ms"].notna() & (merged["run_sd_rt_ms"] > 0)
    merged.loc[valid_sd_mask, "is_outlier"] = (
        np.abs(
            merged.loc[valid_sd_mask, "reaction_time_ms"] -
            merged.loc[valid_sd_mask, "run_mean_rt_ms"]
        ) > (3 * merged.loc[valid_sd_mask, "run_sd_rt_ms"])
    )

    outlier_report = (
        merged.groupby(
            ["subject_id", "group", "session", "session_id", "run_id"],
            observed=False,
        )
        .agg(
            n_trials_before=("reaction_time_ms", "size"),
            n_outliers_removed=("is_outlier", "sum"),
            mean_before_ms=("reaction_time_ms", "mean"),
            sd_before_ms=("reaction_time_ms", "std"),
        )
        .reset_index()
    )
    outlier_report["percent_removed"] = (
        outlier_report["n_outliers_removed"] / outlier_report["n_trials_before"] * 100.0
    )
    outlier_report["n_trials_after"] = (
        outlier_report["n_trials_before"] - outlier_report["n_outliers_removed"]
    )

    cleaned = merged.loc[~merged["is_outlier"]].copy()
    return cleaned, outlier_report


def _summarize_stroop_rt_exclusions_by_subject(incorrect_trial_summary, outlier_report):
    """Summarize Stroop RT exclusions by subject across all runs and sessions."""
    subject_summary = (
        outlier_report.groupby(["subject_id", "group"], observed=False)
        .agg(
            n_runs=("run_id", "nunique"),
            n_correct_trials_before=("n_trials_before", "sum"),
            n_outliers_removed=("n_outliers_removed", "sum"),
            mean_percent_removed_across_runs=("percent_removed", "mean"),
        )
        .reset_index()
    )
    subject_summary["percent_removed_total_correct_trials"] = (
        subject_summary["n_outliers_removed"] /
        subject_summary["n_correct_trials_before"] * 100.0
    )
    return subject_summary


def _summarize_stroop_rt_exclusions_overall(subject_exclusion_summary):
    """Compute overall average RT exclusion percentages across subjects."""
    overall_mean = float(
        subject_exclusion_summary["percent_removed_total_correct_trials"].mean()
    )
    overall_std = float(
        subject_exclusion_summary["percent_removed_total_correct_trials"].std(ddof=1)
    )
    n_subjects = int(len(subject_exclusion_summary))
    overall_sem = overall_std / np.sqrt(n_subjects) if n_subjects > 1 else np.nan

    by_group = (
        subject_exclusion_summary.groupby("group", observed=False)
        ["percent_removed_total_correct_trials"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "mean_percent_removed",
            "std": "std_percent_removed",
            "count": "n_subjects",
        })
    )
    by_group["sem_percent_removed"] = (
        by_group["std_percent_removed"] / np.sqrt(by_group["n_subjects"])
    )

    return {
        "overall_average_percent_removed": overall_mean,
        "overall_std_percent_removed": overall_std,
        "overall_sem_percent_removed": overall_sem,
        "n_subjects": n_subjects,
        "by_group": by_group,
    }


def _aggregate_stroop_reaction_times(cleaned_trial_data):
    """Aggregate cleaned Stroop RTs to subject-session summaries."""
    by_type = (
        cleaned_trial_data.groupby(
            ["subject_id", "group", "session", "session_id", "trial_type"],
            observed=False,
        )
        .agg(
            mean_rt_ms=("reaction_time_ms", "mean"),
            n_trials=("reaction_time_ms", "size"),
        )
        .reset_index()
    )

    overall = (
        cleaned_trial_data.groupby(
            ["subject_id", "group", "session", "session_id"],
            observed=False,
        )
        .agg(
            mean_rt_ms=("reaction_time_ms", "mean"),
            n_trials=("reaction_time_ms", "size"),
        )
        .reset_index()
    )
    overall["trial_type"] = "overall"

    return pd.concat([by_type, overall], ignore_index=True)


def _print_stroop_reaction_time_checks(
    trial_data,
    trial_data_clean,
    incorrect_trial_summary,
    outlier_report,
    subject_exclusion_summary,
    overall_exclusion_summary,
    subject_session_summary,
):
    """Print sanity checks for Stroop RT analysis."""
    print("\nIncorrect-trial summary:")
    print(
        f"Incorrect trials: {incorrect_trial_summary['n_incorrect_trials']} / "
        f"{incorrect_trial_summary['n_total_trials']} "
        f"({incorrect_trial_summary['percent_incorrect_trials']:.3f}%)"
    )

    print("\nRun-level outlier summary:")
    print(
        outlier_report[
            [
                "subject_id",
                "session",
                "run_id",
                "n_trials_before",
                "n_outliers_removed",
                "percent_removed",
            ]
        ].to_string(index=False)
    )

    print("\nSubject-level RT exclusion summary:")
    print(subject_exclusion_summary.to_string(index=False))

    print("\nOverall RT exclusion summary:")
    print(
        "Average percent of RT trials removed across subjects: "
        f"{overall_exclusion_summary['overall_average_percent_removed']:.3f}%"
    )
    print(
        f"SEM across subjects: "
        f"{overall_exclusion_summary['overall_sem_percent_removed']:.3f}%"
    )
    print(overall_exclusion_summary["by_group"].to_string(index=False))

    print("\nCleaned Stroop RT summary by group, session, and trial type:")
    grouped = (
        subject_session_summary.groupby(
            ["group", "session", "trial_type"], observed=False
        )["mean_rt_ms"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    print(grouped.to_string(index=False))


def _run_stroop_rt_anova(summary_df, label):
    """Run a subject-adjusted two-factor ANOVA on Stroop RT."""
    anova_data = summary_df.copy()
    formula = "mean_rt_ms ~ C(subject_id) + C(session) * C(group)"
    model = ols(formula, data=anova_data).fit()
    anova_table = anova_lm(model, typ=2)

    residual_ss = anova_table.loc["Residual", "sum_sq"]
    anova_report = anova_table.reset_index().rename(columns={"index": "effect"})
    anova_report["partial_eta_sq"] = np.where(
        anova_report["effect"] != "Residual",
        anova_report["sum_sq"] / (anova_report["sum_sq"] + residual_ss),
        np.nan,
    )

    print(f"\n--- {label.title()} RT: 2-Way ANOVA (Time × Group) ---")
    print(anova_report.to_string(index=False))

    return {
        "model": model,
        "anova_table": anova_table,
        "anova_report": anova_report,
        "r_squared": model.rsquared,
    }


def _plot_stroop_reaction_time_figures(subject_session_summary, pre_outlier_trial_data):
    """Generate publication-quality Stroop RT figures matching training style."""
    figures = {}

    with plt.rc_context(_publication_style_rcparams()):
        fig_congruent = _plot_rt_prepost_panel(
            subject_session_summary,
            "congruent",
            "Reaction Time: Congruent Stroop Trials",
        )
        _save_figure_pdf(fig_congruent, "stroop_rt_congruent")
        figures["rt_congruent"] = fig_congruent

        fig_incongruent = _plot_rt_prepost_panel(
            subject_session_summary,
            "incongruent",
            "Reaction Time: Incongruent Stroop Trials",
        )
        _save_figure_pdf(fig_incongruent, "stroop_rt_incongruent")
        figures["rt_incongruent"] = fig_incongruent

        fig_overall = _plot_stroop_rt_overall_barplot(subject_session_summary)
        _save_figure_pdf(fig_overall, "stroop_rt_overall")
        figures["rt_overall"] = fig_overall

        fig_hist = _plot_stroop_rt_distribution_histogram(pre_outlier_trial_data)
        _save_figure_pdf(fig_hist, "stroop_rt_histogram_before_outlier_removal")
        figures["rt_histogram_before_outlier_removal"] = fig_hist

    return figures


def _plot_stroop_rt_overall_barplot(subject_session_summary):
    """Plot overall Stroop RT collapsed across congruent/incongruent trials."""
    overall_summary = subject_session_summary[
        subject_session_summary["trial_type"] == "overall"
    ].copy()
    overall_summary = overall_summary.rename(columns={"mean_rt_ms": "rt_ms"})
    return _plot_rt_combined_barplot(overall_summary)


def _plot_stroop_rt_distribution_histogram(pre_outlier_trial_data):
    """Plot Stroop RT distribution before outlier removal, stratified by trial type."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)

    for ax, trial_type in zip(axes, ["congruent", "incongruent"]):
        subset = pre_outlier_trial_data[
            pre_outlier_trial_data["trial_type"] == trial_type
        ]["reaction_time_ms"].dropna()
        mean_rt = float(subset.mean())
        sd_rt = float(subset.std(ddof=1))

        ax.hist(subset, bins=28, color="#BFBFBF", edgecolor="white")
        ax.axvline(mean_rt, color="#222222", linewidth=1.0, label="Mean")
        ax.axvline(
            mean_rt - 3 * sd_rt,
            color="#C44E52",
            linewidth=0.9,
            linestyle="--",
            label="Mean ± 3 SD",
        )
        ax.axvline(mean_rt + 3 * sd_rt, color="#C44E52", linewidth=0.9, linestyle="--")

        ax.set_title(trial_type.title())
        ax.set_xlabel("RT (ms)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)
        ax.tick_params(axis="both", which="both", length=3, width=0.8)

    axes[0].set_ylabel("Trial count")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        handlelength=1.8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _compute_subject_stroop_effect(subject_session_summary):
    """Compute Stroop effect per subject x session from cleaned RT summaries."""
    wide = (
        subject_session_summary
        .pivot_table(
            index=["subject_id", "group", "session", "session_id"],
            columns="trial_type",
            values="mean_rt_ms",
        )
        .reset_index()
    )

    required_columns = ["congruent", "incongruent"]
    missing_required = [col for col in required_columns if col not in wide.columns]
    if missing_required:
        raise ValueError(
            "Stroop-effect computation requires both congruent and incongruent RT "
            f"summaries, but missing columns were found: {missing_required}"
        )

    incomplete = wide[wide[required_columns].isna().any(axis=1)]
    if not incomplete.empty:
        _print_warning(
            "Some subject x session rows are missing one Stroop RT condition and will "
            "be excluded from Stroop-effect computation."
        )
        print(
            incomplete[
                ["subject_id", "group", "session", "congruent", "incongruent"]
            ].to_string(index=False)
        )

    effect_df = wide.dropna(subset=required_columns).copy()
    effect_df["stroop_effect_ms"] = (
        effect_df["incongruent"] - effect_df["congruent"]
    )
    effect_df["stroop_effect_s"] = effect_df["stroop_effect_ms"] / 1000.0

    return effect_df[
        [
            "subject_id",
            "group",
            "session",
            "session_id",
            "congruent",
            "incongruent",
            "stroop_effect_ms",
            "stroop_effect_s",
        ]
    ].rename(columns={
        "congruent": "congruent_rt_ms",
        "incongruent": "incongruent_rt_ms",
    })


def _print_stroop_effect_checks(
    incorrect_trial_summary,
    subject_exclusion_summary,
    overall_exclusion_summary,
    subject_session_stroop_effect,
):
    """Print sanity checks for Stroop-effect analysis."""
    print("\nIncorrect-trial summary reused from Stroop Analysis 3:")
    print(
        f"Incorrect trials: {incorrect_trial_summary['n_incorrect_trials']} / "
        f"{incorrect_trial_summary['n_total_trials']} "
        f"({incorrect_trial_summary['percent_incorrect_trials']:.3f}%)"
    )

    print("\nRT exclusion summary reused from Stroop Analysis 3:")
    print(subject_exclusion_summary.to_string(index=False))
    print(overall_exclusion_summary["by_group"].to_string(index=False))

    print("\nSubject-session Stroop-effect summary:")
    print(subject_session_stroop_effect.to_string(index=False))


def _run_stroop_effect_anova(subject_session_stroop_effect):
    """Run a subject-adjusted two-factor ANOVA on Stroop effect."""
    anova_data = subject_session_stroop_effect.copy()
    formula = "stroop_effect_ms ~ C(subject_id) + C(group) * C(session)"
    model = ols(formula, data=anova_data).fit()
    anova_table = anova_lm(model, typ=2)

    residual_ss = anova_table.loc["Residual", "sum_sq"]
    anova_report = anova_table.reset_index().rename(columns={"index": "effect"})
    anova_report["partial_eta_sq"] = np.where(
        anova_report["effect"] != "Residual",
        anova_report["sum_sq"] / (anova_report["sum_sq"] + residual_ss),
        np.nan,
    )

    print("Stroop-effect ANOVA (subject-adjusted Time × Group):")
    print(anova_report.to_string(index=False))

    return {
        "model": model,
        "anova_table": anova_table,
        "anova_report": anova_report,
        "r_squared": model.rsquared,
    }


def _run_stroop_effect_posthocs(subject_session_stroop_effect):
    """Run planned post hoc contrasts for Stroop effect."""
    results = []

    for group in ["control", "experimental"]:
        wide = (
            subject_session_stroop_effect[subject_session_stroop_effect["group"] == group]
            .pivot_table(index="subject_id", columns="session", values="stroop_effect_ms")
            .dropna(subset=["pre", "post"])
        )
        if wide.empty:
            continue

        diff = wide["post"] - wide["pre"]
        t_stat, p_value = stats.ttest_rel(wide["post"], wide["pre"])
        effect_size = _cohens_dz(diff.to_numpy())
        ci_low, ci_high = _paired_mean_difference_ci(diff.to_numpy())

        results.append({
            "contrast_type": "paired_pre_post",
            "group": group,
            "n_subjects": len(wide),
            "estimate_ms": diff.mean(),
            "statistic": t_stat,
            "df": len(wide) - 1,
            "p_value": p_value,
            "effect_size": effect_size,
            "effect_size_label": "cohens_dz",
            "ci95_low_ms": ci_low,
            "ci95_high_ms": ci_high,
        })

    wide = (
        subject_session_stroop_effect
        .pivot_table(
            index=["subject_id", "group"],
            columns="session",
            values="stroop_effect_ms",
        )
        .dropna(subset=["pre", "post"])
        .reset_index()
    )
    wide["change_ms"] = wide["post"] - wide["pre"]
    control_change = wide.loc[wide["group"] == "control", "change_ms"].to_numpy()
    experimental_change = wide.loc[
        wide["group"] == "experimental", "change_ms"
    ].to_numpy()

    if len(control_change) and len(experimental_change):
        t_stat, p_value = stats.ttest_ind(
            experimental_change, control_change, equal_var=False
        )
        effect_size = _cohens_d_independent(
            experimental_change, control_change
        )
        ci_low, ci_high = _welch_mean_difference_ci(
            experimental_change, control_change
        )

        results.append({
            "contrast_type": "between_group_change",
            "group": "experimental-control",
            "n_subjects": len(control_change) + len(experimental_change),
            "estimate_ms": experimental_change.mean() - control_change.mean(),
            "statistic": t_stat,
            "df": _welch_df(experimental_change, control_change),
            "p_value": p_value,
            "effect_size": effect_size,
            "effect_size_label": "cohens_d",
            "ci95_low_ms": ci_low,
            "ci95_high_ms": ci_high,
        })

    posthoc_df = pd.DataFrame(results)
    if not posthoc_df.empty:
        print("\nPlanned Stroop-effect contrasts:")
        print(posthoc_df.to_string(index=False))
    else:
        _print_warning(
            "No Stroop-effect post hoc contrasts could be computed from the available summaries."
        )

    return posthoc_df


def _plot_stroop_effect_figures(subject_session_stroop_effect):
    """Generate publication-quality Stroop-effect figures."""
    figures = {}

    with plt.rc_context(_publication_style_rcparams()):
        fig = _plot_stroop_effect_barplot(subject_session_stroop_effect)
        _save_figure_pdf(fig, "stroop_effect")
        figures["stroop_effect"] = fig

    return figures


def _plot_stroop_effect_barplot(subject_session_stroop_effect):
    """Plot Stroop effect with group mean lines and faint subject trajectories."""
    colors = {"control": "#4C72B0", "experimental": "#DD8452"}
    marker_color = "#333333"
    plotted_values = []

    fig, ax = plt.subplots(figsize=(4.8, 4.0))

    for group in ["control", "experimental"]:
        group_data = subject_session_stroop_effect[
            subject_session_stroop_effect["group"] == group
        ].copy()
        wide = group_data.pivot_table(
            index="subject_id", columns="session", values="stroop_effect_ms"
        )
        wide = wide.dropna(subset=["pre", "post"])
        if wide.empty:
            continue

        pre_values = wide["pre"].to_numpy()
        post_values = wide["post"].to_numpy()
        pre_mean = float(np.mean(pre_values))
        post_mean = float(np.mean(post_values))
        pre_sem = (
            float(np.std(pre_values, ddof=1) / np.sqrt(len(pre_values)))
            if len(pre_values) > 1 else 0.0
        )
        post_sem = (
            float(np.std(post_values, ddof=1) / np.sqrt(len(post_values)))
            if len(post_values) > 1 else 0.0
        )

        plotted_values.extend([
            pre_mean - pre_sem,
            pre_mean + pre_sem,
            post_mean - post_sem,
            post_mean + post_sem,
        ])

        ax.errorbar(
            [0, 1],
            [pre_mean, post_mean],
            yerr=[pre_sem, post_sem],
            marker="o",
            markersize=6,
            color=colors[group],
            capsize=3,
            capthick=1,
            linewidth=1.6,
            label=group.capitalize(),
            zorder=3,
        )

        for _, row in wide.iterrows():
            plotted_values.extend([row["pre"], row["post"]])
            ax.plot(
                [0, 1],
                [row["pre"], row["post"]],
                color=colors[group],
                alpha=0.22,
                linewidth=0.8,
                zorder=1,
            )
            ax.scatter(
                [0, 1],
                [row["pre"], row["post"]],
                color=marker_color,
                s=10,
                alpha=0.5,
                zorder=2,
            )

    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel("Stroop Effect (ms)")
    ax.set_title("Stroop Effect Across Sessions")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=3, width=0.8)
    _set_y_limits_with_padding(ax, plotted_values, pad_fraction=0.15, min_pad=15.0)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        handlelength=1.8,
    )
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    return fig


def analyze_reaction_time(df):
    """ANALYSIS 1: Compute reaction-time metrics, figures, and statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level consolidated training DataFrame.

    Returns
    -------
    dict
        Keys: 'trial_data_before_outlier_removal', 'trial_data_clean',
        'subject_session_exclusion_report', 'exclusion_summary_by_session',
        'outlier_report', 'subject_session_summary', 'cell_summary',
        'stats_anova', 'posthoc_tests', 'figures', 'warnings'
    """
    _require_statistical_dependencies(require_scipy=True, require_statsmodels=True)

    print("=" * 80)
    print("ANALYSIS 1: Reaction Time")
    print("=" * 80)

    (
        trial_data,
        warnings_list,
        subject_session_exclusion_report,
    ) = _prepare_reaction_time_trial_data(df)
    trial_data_clean, outlier_report = _remove_rt_outliers(trial_data)
    subject_session_exclusion_report = subject_session_exclusion_report.merge(
        outlier_report[
            [
                "subject_id",
                "group",
                "session",
                "n_outliers_removed",
                "percent_removed",
            ]
        ],
        on=["subject_id", "group", "session"],
        how="left",
    )
    subject_session_exclusion_report["n_outliers_removed"] = (
        subject_session_exclusion_report["n_outliers_removed"].fillna(0).astype(int)
    )
    subject_session_exclusion_report["percent_outliers_removed"] = (
        subject_session_exclusion_report["n_outliers_removed"] /
        subject_session_exclusion_report["n_total_trials"] * 100.0
    )
    subject_session_exclusion_report["total_removed_before_sd"] = (
        subject_session_exclusion_report["n_removed_non_correct"] +
        subject_session_exclusion_report["n_removed_rt_lt_150"]
    )
    subject_session_exclusion_report["total_removed_all_filters"] = (
        subject_session_exclusion_report["total_removed_before_sd"] +
        subject_session_exclusion_report["n_outliers_removed"]
    )
    subject_session_exclusion_report["percent_removed_non_correct"] = (
        subject_session_exclusion_report["n_removed_non_correct"] /
        subject_session_exclusion_report["n_total_trials"] * 100.0
    )
    subject_session_exclusion_report["percent_removed_rt_lt_150"] = (
        subject_session_exclusion_report["n_removed_rt_lt_150"] /
        subject_session_exclusion_report["n_total_trials"] * 100.0
    )
    subject_session_exclusion_report["percent_removed_before_sd"] = (
        subject_session_exclusion_report["total_removed_before_sd"] /
        subject_session_exclusion_report["n_total_trials"] * 100.0
    )
    subject_session_exclusion_report["percent_removed_all_filters"] = (
        subject_session_exclusion_report["total_removed_all_filters"] /
        subject_session_exclusion_report["n_total_trials"] * 100.0
    )
    exclusion_summary_by_session = _summarize_rt_exclusions_by_session(
        subject_session_exclusion_report
    )
    subject_session_summary = _aggregate_subject_reaction_times(trial_data_clean)
    _print_reaction_time_checks(
        trial_data,
        trial_data_clean,
        outlier_report,
        subject_session_summary,
        subject_session_exclusion_report,
        exclusion_summary_by_session,
    )

    cell_summary = (
        subject_session_summary
        .groupby(["group", "session", "trial_type"], observed=False)["mean_rt_ms"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    cell_summary["sem"] = cell_summary["std"] / np.sqrt(cell_summary["count"])
    cell_summary["ci95_low"] = cell_summary["mean"] - stats.t.ppf(0.975, cell_summary["count"] - 1) * cell_summary["sem"]
    cell_summary["ci95_high"] = cell_summary["mean"] + stats.t.ppf(0.975, cell_summary["count"] - 1) * cell_summary["sem"]

    print("\nCell means after outlier removal (ms):")
    print(cell_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSES")
    print("=" * 80)
    stats_anova = _run_reaction_time_anova(subject_session_summary)
    posthoc_tests = _run_reaction_time_posthocs(subject_session_summary)

    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    figures = _plot_reaction_time_figures(
        subject_session_summary,
        trial_data,
        trial_data_clean,
    )

    print("\nAnalysis 1 complete!")
    print("=" * 80)

    return {
        "trial_data_before_outlier_removal": trial_data,
        "trial_data_clean": trial_data_clean,
        "subject_session_exclusion_report": subject_session_exclusion_report,
        "exclusion_summary_by_session": exclusion_summary_by_session,
        "outlier_report": outlier_report,
        "subject_session_summary": subject_session_summary,
        "cell_summary": cell_summary,
        "stats_anova": stats_anova,
        "posthoc_tests": posthoc_tests,
        "figures": figures,
        "warnings": warnings_list,
    }


def _prepare_reaction_time_trial_data(df):
    """Validate and prepare trial-level RT data for Analysis 1."""
    required_columns = {"subject_id", "group", "session_id", "feedback", "rt_ms"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Reaction-time analysis is missing required columns: "
            f"{missing_columns}"
        )

    if "trial_type" in df.columns:
        trial_type_col = "trial_type"
    elif "task" in df.columns:
        trial_type_col = "task"
        _print_warning(
            "Input data uses `task` rather than `trial_type`; mapping "
            "`task == 0/1` to no_distractor/distractor for Analysis 1."
        )
    else:
        raise ValueError(
            "Reaction-time analysis requires either a `trial_type` column "
            "or the consolidated `task` column."
        )

    data = df.copy()
    warnings_list = []

    if "task_type" in data.columns:
        non_training_count = int((data["task_type"] != "training").sum())
        if non_training_count:
            warning = (
                f"Found {non_training_count} non-training rows; Analysis 1 uses "
                "training rows only."
            )
            warnings_list.append(warning)
            _print_warning(warning)
            data = data[data["task_type"] == "training"].copy()

    subject_count = int(data["subject_id"].nunique())
    if subject_count != 32:
        warning = (
            f"Consolidated training CSV contains {subject_count} subjects rather "
            "than the full 32-subject study roster. Proceeding with available data."
        )
        warnings_list.append(warning)
        _print_warning(warning)

    run_counts = (
        data.groupby(["subject_id", "session_id"], observed=False)["run_id"]
        .nunique()
        .reset_index(name="n_runs")
    )
    expected_runs = {1: 8, 5: 4}
    run_count_issues = run_counts[
        run_counts.apply(
            lambda row: expected_runs.get(int(row["session_id"]), row["n_runs"]) != row["n_runs"],
            axis=1,
        )
    ]
    if not run_count_issues.empty:
        warning = (
            "Some subject/session combinations have fewer runs than expected "
            "(8 in Session 1, 4 in Session 5)."
        )
        warnings_list.append(warning)
        _print_warning(warning)
        print(run_count_issues.to_string(index=False))

    invalid_feedback = sorted(set(data["feedback"].dropna()) - {1, 2, 3})
    if invalid_feedback:
        warning = f"Found unexpected feedback codes: {invalid_feedback}"
        warnings_list.append(warning)
        _print_warning(warning)

    session_map = {1: "pre", 5: "post"}
    invalid_sessions = sorted(set(data["session_id"].dropna()) - set(session_map))
    if invalid_sessions:
        warning = f"Found session IDs outside the expected pre/post mapping: {invalid_sessions}"
        warnings_list.append(warning)
        _print_warning(warning)

    data["session"] = data["session_id"].map(session_map)
    trial_map = {0: "no_distractor", 1: "distractor"}
    data["trial_type"] = data[trial_type_col].map(trial_map)

    invalid_trial_codes = sorted(
        set(data[trial_type_col].dropna()) - set(trial_map)
    )
    if invalid_trial_codes:
        warning = f"Found unexpected trial-type codes: {invalid_trial_codes}"
        warnings_list.append(warning)
        _print_warning(warning)

    missing_rt = int(data["rt_ms"].isna().sum())
    if missing_rt:
        warning = f"Found {missing_rt} trials with missing RT values; they will be excluded."
        warnings_list.append(warning)
        _print_warning(warning)

    included = data[
        (data["feedback"] == 1) &
        (data["rt_ms"].notna()) &
        (data["rt_ms"] >= 150) &
        (data["session"].notna()) &
        (data["trial_type"].notna())
    ].copy()

    print("\nInclusion summary for RT analysis:")
    print(f"Total training rows: {len(data)}")
    print(f"Correct-response rows retained before outlier removal: {len(included)}")
    print(f"Rows excluded for incorrect/timeout/rt<150/missing mapping: {len(data) - len(included)}")

    subject_session_exclusion_report = _build_rt_exclusion_report(data)

    return included, warnings_list, subject_session_exclusion_report


def _build_rt_exclusion_report(data):
    """Create a subject x session report for RT trial exclusions before SD filtering."""
    report = (
        data.groupby(["subject_id", "group", "session"], observed=False)
        .agg(
            n_total_trials=("rt_ms", "size"),
            n_removed_non_correct=("feedback", lambda x: int((x != 1).sum())),
            n_removed_rt_lt_150=(
                "rt_ms",
                lambda x: 0,
            ),
        )
        .reset_index()
    )

    rt_lt_150 = (
        data.assign(is_rt_lt_150=(data["feedback"] == 1) & data["rt_ms"].notna() & (data["rt_ms"] < 150))
        .groupby(["subject_id", "group", "session"], observed=False)["is_rt_lt_150"]
        .sum()
        .reset_index(name="n_removed_rt_lt_150")
    )
    report = report.drop(columns=["n_removed_rt_lt_150"]).merge(
        rt_lt_150,
        on=["subject_id", "group", "session"],
        how="left",
    )
    report["n_removed_rt_lt_150"] = report["n_removed_rt_lt_150"].fillna(0).astype(int)
    report["n_retained_before_sd"] = (
        report["n_total_trials"]
        - report["n_removed_non_correct"]
        - report["n_removed_rt_lt_150"]
    )
    return report


def _summarize_rt_exclusions_by_session(subject_session_exclusion_report):
    """Summarize RT exclusion percentages across subjects within each session and overall."""
    metrics = [
        "percent_removed_non_correct",
        "percent_removed_rt_lt_150",
        "percent_outliers_removed",
        "percent_removed_before_sd",
        "percent_removed_all_filters",
    ]
    available_metrics = [metric for metric in metrics if metric in subject_session_exclusion_report.columns]
    if not available_metrics:
        return pd.DataFrame({"session": []})

    by_session = (
        subject_session_exclusion_report
        .groupby("session", observed=False)[available_metrics]
        .agg(["mean", "std", "count"])
    )
    by_session.columns = [
        f"{metric}_{stat}" for metric, stat in by_session.columns.to_flat_index()
    ]
    by_session = by_session.reset_index()

    for metric in available_metrics:
        by_session[f"{metric}_sem"] = (
            by_session[f"{metric}_std"] / np.sqrt(by_session[f"{metric}_count"])
        )

    all_sessions_subject = (
        subject_session_exclusion_report
        .groupby(["subject_id", "group"], observed=False)
        .agg(
            n_total_trials=("n_total_trials", "sum"),
            n_removed_non_correct=("n_removed_non_correct", "sum"),
            n_removed_rt_lt_150=("n_removed_rt_lt_150", "sum"),
            n_outliers_removed=("n_outliers_removed", "sum"),
            total_removed_before_sd=("total_removed_before_sd", "sum"),
            total_removed_all_filters=("total_removed_all_filters", "sum"),
        )
        .reset_index()
    )
    all_sessions_subject["percent_removed_non_correct"] = (
        all_sessions_subject["n_removed_non_correct"] /
        all_sessions_subject["n_total_trials"] * 100.0
    )
    all_sessions_subject["percent_removed_rt_lt_150"] = (
        all_sessions_subject["n_removed_rt_lt_150"] /
        all_sessions_subject["n_total_trials"] * 100.0
    )
    all_sessions_subject["percent_outliers_removed"] = (
        all_sessions_subject["n_outliers_removed"] /
        all_sessions_subject["n_total_trials"] * 100.0
    )
    all_sessions_subject["percent_removed_before_sd"] = (
        all_sessions_subject["total_removed_before_sd"] /
        all_sessions_subject["n_total_trials"] * 100.0
    )
    all_sessions_subject["percent_removed_all_filters"] = (
        all_sessions_subject["total_removed_all_filters"] /
        all_sessions_subject["n_total_trials"] * 100.0
    )

    all_sessions_summary = (
        all_sessions_subject[available_metrics]
        .agg(["mean", "std", "count"])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    overall_row = {"session": "all_sessions"}
    for _, row in all_sessions_summary.iterrows():
        metric = row["metric"]
        overall_row[f"{metric}_mean"] = row["mean"]
        overall_row[f"{metric}_std"] = row["std"]
        overall_row[f"{metric}_count"] = row["count"]
        overall_row[f"{metric}_sem"] = row["std"] / np.sqrt(row["count"])

    return pd.concat([by_session, pd.DataFrame([overall_row])], ignore_index=True)


def _remove_rt_outliers(trial_data):
    """Remove RT outliers using mean +/- 3 SD within subject x session across all trial types."""
    grouped = (
        trial_data
        .groupby(["subject_id", "group", "session"], observed=False)["rt_ms"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "session_mean_rt_ms",
            "std": "session_sd_rt_ms",
            "count": "n_trials_before",
        })
    )

    merged = trial_data.merge(
        grouped,
        on=["subject_id", "group", "session"],
        how="left",
    )

    merged["is_outlier"] = False
    valid_sd_mask = merged["session_sd_rt_ms"].notna() & (merged["session_sd_rt_ms"] > 0)
    merged.loc[valid_sd_mask, "is_outlier"] = (
        np.abs(merged.loc[valid_sd_mask, "rt_ms"] - merged.loc[valid_sd_mask, "session_mean_rt_ms"]) >
        (3 * merged.loc[valid_sd_mask, "session_sd_rt_ms"])
    )

    outlier_report = (
        merged.groupby(["subject_id", "group", "session"], observed=False)
        .agg(
            n_trials_before=("rt_ms", "size"),
            n_outliers_removed=("is_outlier", "sum"),
            mean_before_ms=("rt_ms", "mean"),
            sd_before_ms=("rt_ms", "std"),
        )
        .reset_index()
    )
    outlier_report["percent_removed"] = (
        outlier_report["n_outliers_removed"] / outlier_report["n_trials_before"] * 100.0
    )
    outlier_report["n_trials_after"] = (
        outlier_report["n_trials_before"] - outlier_report["n_outliers_removed"]
    )

    cleaned = merged.loc[~merged["is_outlier"]].copy()
    return cleaned, outlier_report


def _aggregate_subject_reaction_times(cleaned_trial_data):
    """Aggregate cleaned RTs to subject x session x trial type mean RT."""
    summary = (
        cleaned_trial_data
        .groupby(["subject_id", "group", "session", "trial_type"], observed=False)
        .agg(
            mean_rt_ms=("rt_ms", "mean"),
            sd_rt_ms=("rt_ms", "std"),
            n_trials=("rt_ms", "size"),
        )
        .reset_index()
    )

    summary["session_numeric"] = summary["session"].map({"pre": 1, "post": 5})
    summary["mean_rt_s"] = summary["mean_rt_ms"] / 1000.0
    return summary


def _print_reaction_time_checks(
    trial_data,
    trial_data_clean,
    outlier_report,
    summary,
    subject_session_exclusion_report,
    exclusion_summary_by_session,
):
    """Print validation and sanity checks for Analysis 1."""
    print("\nRT analysis sanity checks:")
    print(f"Unique subjects: {summary['subject_id'].nunique()}")
    print(f"Groups present: {sorted(summary['group'].unique().tolist())}")
    print(f"Sessions present: {sorted(summary['session'].unique().tolist())}")
    print(f"Trial types present: {sorted(summary['trial_type'].unique().tolist())}")
    print(f"Trials before outlier removal: {len(trial_data)}")
    print(f"Trials after outlier removal: {len(trial_data_clean)}")
    print(f"Total RT outliers removed: {int(outlier_report['n_outliers_removed'].sum())}")

    coverage = (
        summary.groupby(["subject_id", "group"], observed=False)
        .agg(
            n_sessions=("session", "nunique"),
            n_trial_types=("trial_type", "nunique"),
        )
        .reset_index()
    )
    incomplete = coverage[(coverage["n_sessions"] < 2) | (coverage["n_trial_types"] < 2)]
    if incomplete.empty:
        print("All subjects have pre/post coverage for both distractor and no-distractor RT summaries.")
    else:
        _print_warning("Some subjects have incomplete pre/post or trial-type coverage in the RT summary.")
        print(incomplete.to_string(index=False))

    print("\nOutlier removal by subject x session across all trial types (% removed):")
    display_cols = [
        "subject_id", "group", "session",
        "n_trials_before", "n_outliers_removed", "n_trials_after", "percent_removed",
    ]
    print(outlier_report[display_cols].to_string(index=False))

    print("\nTrials removed by subject x session:")
    exclusion_cols = [
        "subject_id",
        "group",
        "session",
        "n_total_trials",
        "percent_removed_non_correct",
        "percent_removed_rt_lt_150",
        "percent_outliers_removed",
        "percent_removed_all_filters",
    ]
    print(subject_session_exclusion_report[exclusion_cols].to_string(index=False))

    print("\nAverage percent of trials removed per subject (mean ± SEM):")
    for _, row in exclusion_summary_by_session.iterrows():
        print(f"Session {row['session']}:")
        print(
            f"  non-correct = {row['percent_removed_non_correct_mean']:.2f}% ± "
            f"{row['percent_removed_non_correct_sem']:.2f}%"
        )
        print(
            f"  rt < 150 ms = {row['percent_removed_rt_lt_150_mean']:.2f}% ± "
            f"{row['percent_removed_rt_lt_150_sem']:.2f}%"
        )
        print(
            f"  SD outliers = {row['percent_outliers_removed_mean']:.2f}% ± "
            f"{row['percent_outliers_removed_sem']:.2f}%"
        )
        print(
            f"  total removed = {row['percent_removed_all_filters_mean']:.2f}% ± "
            f"{row['percent_removed_all_filters_sem']:.2f}%"
        )


def _run_reaction_time_anova(summary_df):
    """Run a subject-adjusted three-factor ANOVA on mean RT."""
    anova_data = summary_df.copy()
    formula = "mean_rt_ms ~ C(subject_id) + C(group) * C(session) * C(trial_type)"
    model = ols(formula, data=anova_data).fit()
    anova_table = anova_lm(model, typ=2)

    residual_ss = anova_table.loc["Residual", "sum_sq"]
    anova_report = anova_table.reset_index().rename(columns={"index": "effect"})
    anova_report["partial_eta_sq"] = np.where(
        anova_report["effect"] != "Residual",
        anova_report["sum_sq"] / (anova_report["sum_sq"] + residual_ss),
        np.nan,
    )

    print(
        "Mixed-design RT ANOVA is implemented with SciPy/Statsmodels preference "
        "using a subject-adjusted OLS model (subject fixed effects + Time x Trial Type x Group)."
    )
    print(anova_report.to_string(index=False))

    return {
        "model": model,
        "anova_table": anova_table,
        "anova_report": anova_report,
        "r_squared": model.rsquared,
    }


def _run_reaction_time_posthocs(summary_df):
    """Run planned post hoc RT contrasts with effect sizes and 95% CIs."""
    results = []

    # Paired pre/post within each group x trial type
    for group in ["control", "experimental"]:
        for trial_type in ["no_distractor", "distractor"]:
            subset = summary_df[
                (summary_df["group"] == group) &
                (summary_df["trial_type"] == trial_type)
            ].copy()
            wide = subset.pivot_table(index="subject_id", columns="session", values="mean_rt_ms")
            wide = wide.dropna(subset=["pre", "post"])
            if wide.empty:
                continue

            diff = wide["post"] - wide["pre"]
            t_stat, p_value = stats.ttest_rel(wide["post"], wide["pre"])
            effect_size = diff.mean() / diff.std(ddof=1) if len(diff) > 1 and diff.std(ddof=1) > 0 else np.nan
            ci_low, ci_high = _mean_difference_ci(diff)

            results.append({
                "contrast_type": "paired_pre_post",
                "group": group,
                "trial_type": trial_type,
                "n_subjects": len(wide),
                "estimate_ms": diff.mean(),
                "statistic": t_stat,
                "df": len(wide) - 1,
                "p_value": p_value,
                "effect_size": effect_size,
                "effect_size_label": "cohens_dz",
                "ci95_low_ms": ci_low,
                "ci95_high_ms": ci_high,
            })

    # Between-group comparison on change score within each trial type
    for trial_type in ["no_distractor", "distractor"]:
        subset = summary_df[summary_df["trial_type"] == trial_type].copy()
        wide = subset.pivot_table(index=["subject_id", "group"], columns="session", values="mean_rt_ms")
        wide = wide.dropna(subset=["pre", "post"]).reset_index()
        wide["change_ms"] = wide["post"] - wide["pre"]

        control_change = wide.loc[wide["group"] == "control", "change_ms"].to_numpy()
        experimental_change = wide.loc[wide["group"] == "experimental", "change_ms"].to_numpy()
        if len(control_change) == 0 or len(experimental_change) == 0:
            continue

        t_stat, p_value = stats.ttest_ind(experimental_change, control_change, equal_var=False)
        effect_size = _cohens_d_independent(experimental_change, control_change)
        ci_low, ci_high = _welch_mean_difference_ci(experimental_change, control_change)

        results.append({
            "contrast_type": "between_group_change",
            "group": "experimental-control",
            "trial_type": trial_type,
            "n_subjects": len(control_change) + len(experimental_change),
            "estimate_ms": experimental_change.mean() - control_change.mean(),
            "statistic": t_stat,
            "df": _welch_df(experimental_change, control_change),
            "p_value": p_value,
            "effect_size": effect_size,
            "effect_size_label": "cohens_d",
            "ci95_low_ms": ci_low,
            "ci95_high_ms": ci_high,
        })

    posthoc_df = pd.DataFrame(results)
    if not posthoc_df.empty:
        print("\nPlanned RT contrasts:")
        print(posthoc_df.to_string(index=False))
    else:
        _print_warning("No RT post hoc contrasts could be computed from the available summaries.")

    return posthoc_df


def _mean_difference_ci(diff_values, alpha=0.05):
    """Return a two-sided confidence interval for a paired mean difference."""
    diff_values = np.asarray(diff_values, dtype=float)
    diff_values = diff_values[np.isfinite(diff_values)]
    if diff_values.size < 2:
        return np.nan, np.nan

    mean_diff = float(np.mean(diff_values))
    sem = stats.sem(diff_values)
    margin = stats.t.ppf(1 - alpha / 2, diff_values.size - 1) * sem
    return mean_diff - margin, mean_diff + margin


def _paired_mean_difference_ci(diff_values, alpha=0.05):
    """Return a two-sided confidence interval for a paired mean difference."""
    return _mean_difference_ci(diff_values, alpha=alpha)


def _cohens_dz(diff_values):
    """Return Cohen's dz for paired-sample differences."""
    diff_values = np.asarray(diff_values, dtype=float)
    diff_values = diff_values[np.isfinite(diff_values)]
    if diff_values.size < 2:
        return np.nan

    sd = np.std(diff_values, ddof=1)
    if sd == 0:
        return np.nan
    return np.mean(diff_values) / sd


def _welch_mean_difference_ci(sample_a, sample_b, alpha=0.05):
    """Return a two-sided confidence interval for an independent mean difference."""
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    if sample_a.size < 2 or sample_b.size < 2:
        return np.nan, np.nan

    mean_diff = float(np.mean(sample_a) - np.mean(sample_b))
    se = np.sqrt(np.var(sample_a, ddof=1) / sample_a.size + np.var(sample_b, ddof=1) / sample_b.size)
    df = _welch_df(sample_a, sample_b)
    margin = stats.t.ppf(1 - alpha / 2, df) * se
    return mean_diff - margin, mean_diff + margin


def _welch_df(sample_a, sample_b):
    """Return Welch-Satterthwaite degrees of freedom."""
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    var_a = np.var(sample_a, ddof=1) / sample_a.size
    var_b = np.var(sample_b, ddof=1) / sample_b.size
    numerator = (var_a + var_b) ** 2
    denominator = (var_a ** 2) / (sample_a.size - 1) + (var_b ** 2) / (sample_b.size - 1)
    return numerator / denominator


def _cohens_d_independent(sample_a, sample_b):
    """Return Cohen's d for two independent samples."""
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    if sample_a.size < 2 or sample_b.size < 2:
        return np.nan

    pooled_sd = np.sqrt(
        ((sample_a.size - 1) * np.var(sample_a, ddof=1) + (sample_b.size - 1) * np.var(sample_b, ddof=1)) /
        (sample_a.size + sample_b.size - 2)
    )
    if pooled_sd == 0:
        return np.nan
    return (np.mean(sample_a) - np.mean(sample_b)) / pooled_sd


def analyze_accuracy_and_timeout(summary_df):
    """ANALYSIS 2: Compute accuracy and timeout metrics with plots and statistics.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Behavioral summary table from create_behavioral_summary_table()
        Columns: subject_id, group, session, trial_type, accuracy, timeout_rate, etc.

    Returns
    -------
    dict
        Keys: 'accuracy_by_condition', 'overall_accuracy', 'timeout_rate',
              'stats_accuracy', 'stats_overall_accuracy', 'stats_timeout',
              'figures'
    """
    _require_statistical_dependencies(require_scipy=False, require_statsmodels=True)

    print("=" * 80)
    print("ANALYSIS 2: Accuracy & Timeout")
    print("=" * 80)

    result = {}

    # === ACCURACY BY TRIAL TYPE ===
    # Use rows where trial_type != 'combined' (i.e., distractor and no_distractor)
    accuracy_by_condition = summary_df[summary_df['trial_type'] != 'combined'].copy()
    accuracy_by_condition['session_numeric'] = accuracy_by_condition['session'].map({'pre': 1, 'post': 5})
    result['accuracy_by_condition'] = accuracy_by_condition

    print("\nAccuracy by trial type (distractor/no_distractor):")
    print(accuracy_by_condition.groupby(['group', 'session', 'trial_type'])['accuracy'].agg(['mean', 'std', 'count']))

    # === OVERALL ACCURACY ===
    # Use rows where trial_type == 'combined'
    overall_accuracy = summary_df[summary_df['trial_type'] == 'combined'].copy()
    overall_accuracy['session_numeric'] = overall_accuracy['session'].map({'pre': 1, 'post': 5})
    result['overall_accuracy'] = overall_accuracy

    print("\nOverall accuracy (both trial types combined):")
    print(overall_accuracy.groupby(['group', 'session'])['accuracy'].agg(['mean', 'std', 'count']))

    # === TIMEOUT RATE ===
    # Use rows where trial_type == 'combined'
    timeout_rate = summary_df[summary_df['trial_type'] == 'combined'].copy()
    timeout_rate['session_numeric'] = timeout_rate['session'].map({'pre': 1, 'post': 5})
    result['timeout_rate'] = timeout_rate

    print("\nTimeout rate:")
    print(timeout_rate.groupby(['group', 'session'])['timeout_rate'].agg(['mean', 'std', 'count']))

    # === STATISTICS ===
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSES")
    print("=" * 80)

    # 1. Accuracy (trial type) → 3-way ANOVA: Time × Trial Type × Group
    print("\n--- Accuracy by Trial Type: 3-Way ANOVA (Time × Trial Type × Group) ---")
    acc_cond_wide = accuracy_by_condition.pivot_table(
        index=['subject_id', 'group'],
        columns=['session', 'trial_type'],
        values='accuracy'
    )
    
    # Prepare data for ANOVA: factors are session (pre/post), trial_type (distractor/no_distractor), group (control/experimental)
    anova_data_accuracy = accuracy_by_condition.copy()
    anova_data_accuracy['session_factor'] = anova_data_accuracy['session'].map({'pre': 0, 'post': 1})
    anova_data_accuracy['trial_type_factor'] = anova_data_accuracy['trial_type'].map({'no_distractor': 0, 'distractor': 1})
    anova_data_accuracy['group_factor'] = anova_data_accuracy['group'].map({'control': 0, 'experimental': 1})
    
    # Run 3-way ANOVA
    model_accuracy = ols('accuracy ~ C(session_factor) * C(trial_type_factor) * C(group_factor)', 
                         data=anova_data_accuracy).fit()
    anova_table_accuracy = anova_lm(model_accuracy, typ=2)
    
    print(anova_table_accuracy)
    result['stats_accuracy'] = {
        'model': model_accuracy,
        'anova_table': anova_table_accuracy,
        'r_squared': model_accuracy.rsquared
    }

    # 2. Overall Accuracy → 2-way ANOVA: Time × Group
    print("\n--- Overall Accuracy: 2-Way ANOVA (Time × Group) ---")
    anova_data_overall = overall_accuracy.copy()
    anova_data_overall['session_factor'] = anova_data_overall['session'].map({'pre': 0, 'post': 1})
    anova_data_overall['group_factor'] = anova_data_overall['group'].map({'control': 0, 'experimental': 1})
    
    model_overall = ols('accuracy ~ C(session_factor) * C(group_factor)', 
                        data=anova_data_overall).fit()
    anova_table_overall = anova_lm(model_overall, typ=2)
    
    print(anova_table_overall)
    result['stats_overall_accuracy'] = {
        'model': model_overall,
        'anova_table': anova_table_overall,
        'r_squared': model_overall.rsquared
    }

    # 3. Timeout Rate → 2-way ANOVA: Time × Group
    print("\n--- Timeout Rate: 2-Way ANOVA (Time × Group) ---")
    anova_data_timeout = timeout_rate.copy()
    anova_data_timeout['session_factor'] = anova_data_timeout['session'].map({'pre': 0, 'post': 1})
    anova_data_timeout['group_factor'] = anova_data_timeout['group'].map({'control': 0, 'experimental': 1})
    
    model_timeout = ols('timeout_rate ~ C(session_factor) * C(group_factor)', 
                        data=anova_data_timeout).fit()
    anova_table_timeout = anova_lm(model_timeout, typ=2)
    
    print(anova_table_timeout)
    result['stats_timeout'] = {
        'model': model_timeout,
        'anova_table': anova_table_timeout,
        'r_squared': model_timeout.rsquared
    }

    # === FIGURES ===
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    figures = _plot_accuracy_and_timeout(accuracy_by_condition, overall_accuracy, timeout_rate)
    result['figures'] = figures

    print("\nAnalysis 2 complete!")
    print("=" * 80)

    return result


def analyze_distractor_cost(df):
    """ANALYSIS 3: Compute distractor RT cost with plots and statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level consolidated training DataFrame.

    Returns
    -------
    dict
        Keys: 'trial_data_before_outlier_removal', 'trial_data_clean',
        'subject_session_exclusion_report', 'exclusion_summary_by_session',
        'outlier_report', 'subject_session_condition_summary',
        'subject_session_cost', 'cell_summary', 'stats_anova',
        'posthoc_tests', 'figures', 'warnings'
    """
    _require_statistical_dependencies(require_scipy=True, require_statsmodels=True)

    print("=" * 80)
    print("ANALYSIS 3: Distractor Cost")
    print("=" * 80)

    (
        trial_data,
        warnings_list,
        subject_session_exclusion_report,
    ) = _prepare_reaction_time_trial_data(df)
    trial_data_clean, outlier_report = _remove_rt_outliers(trial_data)
    subject_session_exclusion_report = subject_session_exclusion_report.merge(
        outlier_report[
            [
                "subject_id",
                "group",
                "session",
                "n_outliers_removed",
                "percent_removed",
            ]
        ],
        on=["subject_id", "group", "session"],
        how="left",
    )
    subject_session_exclusion_report["n_outliers_removed"] = (
        subject_session_exclusion_report["n_outliers_removed"].fillna(0).astype(int)
    )
    subject_session_exclusion_report["percent_outliers_removed"] = (
        subject_session_exclusion_report["n_outliers_removed"] /
        subject_session_exclusion_report["n_total_trials"] * 100.0
    )
    subject_session_exclusion_report["total_removed_before_sd"] = (
        subject_session_exclusion_report["n_removed_non_correct"] +
        subject_session_exclusion_report["n_removed_rt_lt_150"]
    )
    subject_session_exclusion_report["total_removed_all_filters"] = (
        subject_session_exclusion_report["total_removed_before_sd"] +
        subject_session_exclusion_report["n_outliers_removed"]
    )
    subject_session_exclusion_report["percent_removed_non_correct"] = (
        subject_session_exclusion_report["n_removed_non_correct"] /
        subject_session_exclusion_report["n_total_trials"] * 100.0
    )
    subject_session_exclusion_report["percent_removed_rt_lt_150"] = (
        subject_session_exclusion_report["n_removed_rt_lt_150"] /
        subject_session_exclusion_report["n_total_trials"] * 100.0
    )
    subject_session_exclusion_report["percent_removed_before_sd"] = (
        subject_session_exclusion_report["total_removed_before_sd"] /
        subject_session_exclusion_report["n_total_trials"] * 100.0
    )
    subject_session_exclusion_report["percent_removed_all_filters"] = (
        subject_session_exclusion_report["total_removed_all_filters"] /
        subject_session_exclusion_report["n_total_trials"] * 100.0
    )
    exclusion_summary_by_session = _summarize_rt_exclusions_by_session(
        subject_session_exclusion_report
    )

    subject_session_condition_summary = _aggregate_subject_reaction_times(
        trial_data_clean
    )
    subject_session_cost = _compute_subject_distractor_cost(
        subject_session_condition_summary
    )
    _print_distractor_cost_checks(
        trial_data,
        trial_data_clean,
        outlier_report,
        subject_session_cost,
        subject_session_exclusion_report,
        exclusion_summary_by_session,
    )

    cell_summary = (
        subject_session_cost
        .groupby(["group", "session"], observed=False)["distractor_cost_ms"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    cell_summary["sem"] = cell_summary["std"] / np.sqrt(cell_summary["count"])
    cell_summary["ci95_low"] = (
        cell_summary["mean"] -
        stats.t.ppf(0.975, cell_summary["count"] - 1) * cell_summary["sem"]
    )
    cell_summary["ci95_high"] = (
        cell_summary["mean"] +
        stats.t.ppf(0.975, cell_summary["count"] - 1) * cell_summary["sem"]
    )

    print("\nDistractor-cost cell means (ms):")
    print(cell_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSES")
    print("=" * 80)
    stats_anova = _run_distractor_cost_anova(subject_session_cost)
    posthoc_tests = _run_distractor_cost_posthocs(subject_session_cost)

    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    figures = _plot_distractor_cost_figures(subject_session_cost)

    print("\nAnalysis 3 complete!")
    print("=" * 80)

    return {
        "trial_data_before_outlier_removal": trial_data,
        "trial_data_clean": trial_data_clean,
        "subject_session_exclusion_report": subject_session_exclusion_report,
        "exclusion_summary_by_session": exclusion_summary_by_session,
        "outlier_report": outlier_report,
        "subject_session_condition_summary": subject_session_condition_summary,
        "subject_session_cost": subject_session_cost,
        "cell_summary": cell_summary,
        "stats_anova": stats_anova,
        "posthoc_tests": posthoc_tests,
        "figures": figures,
        "warnings": warnings_list,
    }


def _compute_subject_distractor_cost(subject_session_condition_summary):
    """Compute distractor cost per subject x session from cleaned RT summaries."""
    wide = (
        subject_session_condition_summary
        .pivot_table(
            index=["subject_id", "group", "session", "session_numeric"],
            columns="trial_type",
            values="mean_rt_ms",
        )
        .reset_index()
    )

    required_columns = ["distractor", "no_distractor"]
    missing_required = [col for col in required_columns if col not in wide.columns]
    if missing_required:
        raise ValueError(
            "Distractor-cost computation requires both distractor and "
            f"no_distractor RT summaries, but missing columns were found: {missing_required}"
        )

    incomplete = wide[wide[required_columns].isna().any(axis=1)]
    if not incomplete.empty:
        _print_warning(
            "Some subject x session rows are missing one RT condition and will "
            "be excluded from distractor-cost computation."
        )
        print(
            incomplete[
                ["subject_id", "group", "session", "distractor", "no_distractor"]
            ].to_string(index=False)
        )

    cost_df = wide.dropna(subset=required_columns).copy()
    cost_df["distractor_cost_ms"] = (
        cost_df["distractor"] - cost_df["no_distractor"]
    )
    cost_df["distractor_cost_s"] = cost_df["distractor_cost_ms"] / 1000.0

    print("\nComputed distractor cost for subject x session rows:")
    print(
        cost_df[
            [
                "subject_id",
                "group",
                "session",
                "distractor",
                "no_distractor",
                "distractor_cost_ms",
            ]
        ].to_string(index=False)
    )

    return cost_df


def _print_distractor_cost_checks(
    trial_data,
    trial_data_clean,
    outlier_report,
    subject_session_cost,
    subject_session_exclusion_report,
    exclusion_summary_by_session,
):
    """Print validation and sanity checks for Analysis 3."""
    print("\nDistractor-cost analysis sanity checks:")
    print(f"Unique subjects: {subject_session_cost['subject_id'].nunique()}")
    print(f"Groups present: {sorted(subject_session_cost['group'].unique().tolist())}")
    print(f"Sessions present: {sorted(subject_session_cost['session'].unique().tolist())}")
    print(f"Trials before outlier removal: {len(trial_data)}")
    print(f"Trials after outlier removal: {len(trial_data_clean)}")
    print(f"Total RT outliers removed: {int(outlier_report['n_outliers_removed'].sum())}")

    coverage = (
        subject_session_cost
        .groupby(["subject_id", "group"], observed=False)["session"]
        .nunique()
        .reset_index(name="n_sessions")
    )
    incomplete = coverage[coverage["n_sessions"] < 2]
    if incomplete.empty:
        print("All subjects contributing to distractor cost have both pre and post sessions.")
    else:
        _print_warning("Some subjects are missing pre or post distractor-cost summaries.")
        print(incomplete.to_string(index=False))

    print("\nOutlier removal by subject x session across all trial types (% removed):")
    display_cols = [
        "subject_id", "group", "session",
        "n_trials_before", "n_outliers_removed", "n_trials_after", "percent_removed",
    ]
    print(outlier_report[display_cols].to_string(index=False))

    print("\nTrials removed by subject x session:")
    exclusion_cols = [
        "subject_id",
        "group",
        "session",
        "n_total_trials",
        "percent_removed_non_correct",
        "percent_removed_rt_lt_150",
        "percent_outliers_removed",
        "percent_removed_all_filters",
    ]
    print(subject_session_exclusion_report[exclusion_cols].to_string(index=False))

    print("\nAverage percent of trials removed per subject (mean ± SEM):")
    for _, row in exclusion_summary_by_session.iterrows():
        print(f"Session {row['session']}:")
        print(
            f"  non-correct = {row['percent_removed_non_correct_mean']:.2f}% ± "
            f"{row['percent_removed_non_correct_sem']:.2f}%"
        )
        print(
            f"  rt < 150 ms = {row['percent_removed_rt_lt_150_mean']:.2f}% ± "
            f"{row['percent_removed_rt_lt_150_sem']:.2f}%"
        )
        print(
            f"  SD outliers = {row['percent_outliers_removed_mean']:.2f}% ± "
            f"{row['percent_outliers_removed_sem']:.2f}%"
        )
        print(
            f"  total removed = {row['percent_removed_all_filters_mean']:.2f}% ± "
            f"{row['percent_removed_all_filters_sem']:.2f}%"
        )


def _run_distractor_cost_anova(subject_session_cost):
    """Run a subject-adjusted two-factor ANOVA on distractor cost."""
    anova_data = subject_session_cost.copy()
    formula = "distractor_cost_ms ~ C(subject_id) + C(group) * C(session)"
    model = ols(formula, data=anova_data).fit()
    anova_table = anova_lm(model, typ=2)

    residual_ss = anova_table.loc["Residual", "sum_sq"]
    anova_report = anova_table.reset_index().rename(columns={"index": "effect"})
    anova_report["partial_eta_sq"] = np.where(
        anova_report["effect"] != "Residual",
        anova_report["sum_sq"] / (anova_report["sum_sq"] + residual_ss),
        np.nan,
    )

    print(
        "Distractor-cost ANOVA is implemented with a subject-adjusted OLS model "
        "(subject fixed effects + Time x Group)."
    )
    print(anova_report.to_string(index=False))

    return {
        "model": model,
        "anova_table": anova_table,
        "anova_report": anova_report,
        "r_squared": model.rsquared,
    }


def _run_distractor_cost_posthocs(subject_session_cost):
    """Run planned pre/post and change-score contrasts for distractor cost."""
    results = []

    for group in ["control", "experimental"]:
        subset = subject_session_cost[subject_session_cost["group"] == group].copy()
        wide = subset.pivot_table(
            index="subject_id", columns="session", values="distractor_cost_ms"
        )
        wide = wide.dropna(subset=["pre", "post"])
        if wide.empty:
            continue

        diff = wide["post"] - wide["pre"]
        t_stat, p_value = stats.ttest_rel(wide["post"], wide["pre"])
        effect_size = (
            diff.mean() / diff.std(ddof=1)
            if len(diff) > 1 and diff.std(ddof=1) > 0
            else np.nan
        )
        ci_low, ci_high = _mean_difference_ci(diff)

        results.append({
            "contrast_type": "paired_pre_post",
            "group": group,
            "n_subjects": len(wide),
            "estimate_ms": diff.mean(),
            "statistic": t_stat,
            "df": len(wide) - 1,
            "p_value": p_value,
            "effect_size": effect_size,
            "effect_size_label": "cohens_dz",
            "ci95_low_ms": ci_low,
            "ci95_high_ms": ci_high,
        })

    wide = (
        subject_session_cost
        .pivot_table(
            index=["subject_id", "group"],
            columns="session",
            values="distractor_cost_ms",
        )
        .dropna(subset=["pre", "post"])
        .reset_index()
    )
    wide["change_ms"] = wide["post"] - wide["pre"]
    control_change = wide.loc[wide["group"] == "control", "change_ms"].to_numpy()
    experimental_change = wide.loc[
        wide["group"] == "experimental", "change_ms"
    ].to_numpy()

    if len(control_change) and len(experimental_change):
        t_stat, p_value = stats.ttest_ind(
            experimental_change, control_change, equal_var=False
        )
        effect_size = _cohens_d_independent(
            experimental_change, control_change
        )
        ci_low, ci_high = _welch_mean_difference_ci(
            experimental_change, control_change
        )

        results.append({
            "contrast_type": "between_group_change",
            "group": "experimental-control",
            "n_subjects": len(control_change) + len(experimental_change),
            "estimate_ms": experimental_change.mean() - control_change.mean(),
            "statistic": t_stat,
            "df": _welch_df(experimental_change, control_change),
            "p_value": p_value,
            "effect_size": effect_size,
            "effect_size_label": "cohens_d",
            "ci95_low_ms": ci_low,
            "ci95_high_ms": ci_high,
        })

    posthoc_df = pd.DataFrame(results)
    if not posthoc_df.empty:
        print("\nPlanned distractor-cost contrasts:")
        print(posthoc_df.to_string(index=False))
    else:
        _print_warning(
            "No distractor-cost post hoc contrasts could be computed from the available summaries."
        )

    return posthoc_df


def _plot_distractor_cost_figures(subject_session_cost):
    """Generate publication-quality distractor-cost figures."""
    figures = {}

    with plt.rc_context(_publication_style_rcparams()):
        fig = _plot_distractor_cost_barplot(subject_session_cost)
        _save_figure_pdf(fig, "behavioral_distractor_cost")
        figures["distractor_cost"] = fig

    return figures


def _plot_distractor_cost_barplot(subject_session_cost):
    """Plot distractor cost with group mean lines and faint subject trajectories."""
    colors = {"control": "#4C72B0", "experimental": "#DD8452"}
    marker_color = "#333333"
    plotted_values = []

    fig, ax = plt.subplots(figsize=(4.8, 4.0))

    for group in ["control", "experimental"]:
        group_data = subject_session_cost[
            subject_session_cost["group"] == group
        ].copy()
        wide = group_data.pivot_table(
            index="subject_id", columns="session", values="distractor_cost_ms"
        )
        wide = wide.dropna(subset=["pre", "post"])
        if wide.empty:
            continue

        pre_values = wide["pre"].to_numpy()
        post_values = wide["post"].to_numpy()
        pre_mean = float(np.mean(pre_values))
        post_mean = float(np.mean(post_values))
        pre_sem = (
            float(np.std(pre_values, ddof=1) / np.sqrt(len(pre_values)))
            if len(pre_values) > 1 else 0.0
        )
        post_sem = (
            float(np.std(post_values, ddof=1) / np.sqrt(len(post_values)))
            if len(post_values) > 1 else 0.0
        )

        plotted_values.extend([
            pre_mean - pre_sem,
            pre_mean + pre_sem,
            post_mean - post_sem,
            post_mean + post_sem,
        ])

        ax.errorbar(
            [0, 1],
            [pre_mean, post_mean],
            yerr=[pre_sem, post_sem],
            marker="o",
            markersize=6,
            color=colors[group],
            capsize=3,
            capthick=1,
            linewidth=1.6,
            label=group.capitalize(),
            zorder=3,
        )

        for _, row in wide.iterrows():
            plotted_values.extend([row["pre"], row["post"]])
            ax.plot(
                [0, 1],
                [row["pre"], row["post"]],
                color=colors[group],
                alpha=0.22,
                linewidth=0.8,
                zorder=1,
            )
            ax.scatter(
                [0, 1],
                [row["pre"], row["post"]],
                color=marker_color,
                s=10,
                alpha=0.5,
                zorder=2,
            )

    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel("Distractor Cost (ms)")
    ax.set_title("Distractor Cost Across Sessions")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=3, width=0.8)
    _set_y_limits_with_padding(ax, plotted_values, pad_fraction=0.15, min_pad=15.0)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        handlelength=1.8,
    )
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    return fig


def _plot_accuracy_and_timeout(accuracy_by_condition, overall_accuracy, timeout_rate):
    """Generate publication-quality plots for accuracy and timeout analysis.

    Parameters
    ----------
    accuracy_by_condition : pd.DataFrame
        Accuracy data by trial type
    overall_accuracy : pd.DataFrame
        Overall accuracy (combined trial types)
    timeout_rate : pd.DataFrame
        Timeout rate data

    Returns
    -------
    dict
        Keys: figure handles for each plot
    """
    figures = {}

    # Shared styling
    with plt.rc_context(_publication_style_rcparams()):
        # Muted professional palette
        colors = {'control': '#4C72B0', 'experimental': '#DD8452'}
        marker_color = '#333333'

        # === FIGURE 1: Accuracy by Trial Type (Distractor vs No Distractor) ===
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0))
        fig.suptitle('Accuracy for BCI and Control Groups', fontsize=8, fontweight='bold')
        panel_accuracy_values = {'no_distractor': [], 'distractor': []}

        for ax_idx, trial_type in enumerate(['no_distractor', 'distractor']):
            ax = axes[ax_idx]
            subset = accuracy_by_condition[accuracy_by_condition['trial_type'] == trial_type]

            for group in ['control', 'experimental']:
                group_data = subset[subset['group'] == group]

                pre_data = group_data[group_data['session'] == 'pre']['accuracy'].values
                post_data = group_data[group_data['session'] == 'post']['accuracy'].values

                if len(pre_data) == 0 or len(post_data) == 0:
                    continue

                pre_mean = np.mean(pre_data)
                post_mean = np.mean(post_data)
                pre_sem = np.std(pre_data, ddof=1) / np.sqrt(len(pre_data))
                post_sem = np.std(post_data, ddof=1) / np.sqrt(len(post_data))
                panel_accuracy_values[trial_type].extend([
                    pre_mean - pre_sem,
                    pre_mean + pre_sem,
                    post_mean - post_sem,
                    post_mean + post_sem,
                ])

                x_pos = [0, 1]
                means = [pre_mean, post_mean]
                sems = [pre_sem, post_sem]
                ax.errorbar(
                    x_pos,
                    means,
                    yerr=sems,
                    marker='o',
                    markersize=4,
                    color=colors[group],
                    capsize=3,
                    capthick=1,
                    linewidth=1.0,
                    label=f"{group.capitalize()}"
                )

                for subj_id in group_data['subject_id'].unique():
                    subj_pre = group_data[(group_data['subject_id'] == subj_id) & (group_data['session'] == 'pre')]['accuracy'].values
                    subj_post = group_data[(group_data['subject_id'] == subj_id) & (group_data['session'] == 'post')]['accuracy'].values
                    if len(subj_pre) > 0 and len(subj_post) > 0:
                        panel_accuracy_values[trial_type].extend([subj_pre[0], subj_post[0]])
                        ax.plot([0, 1], [subj_pre[0], subj_post[0]], color=colors[group], alpha=0.25, linewidth=0.8)
                        ax.scatter([0, 1], [subj_pre[0], subj_post[0]], color=marker_color, s=8, alpha=0.5, zorder=2)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Pre', 'Post'])
            ax.set_ylabel('Accuracy')
            ax.set_title(f"{trial_type.replace('_', ' ').title()} Trials")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.8)
            ax.spines['left'].set_linewidth(0.8)
            ax.tick_params(axis='both', which='both', length=3, width=0.8)
            _set_y_limits_with_padding(
                ax,
                panel_accuracy_values[trial_type],
                pad_fraction=0.12,
                min_pad=0.003,
            )

        legend_handles = [
            plt.Line2D([0], [0], color=colors['control'], marker='o', markersize=4, linewidth=1.0, label='Control'),
            plt.Line2D([0], [0], color=colors['experimental'], marker='o', markersize=4, linewidth=1.0, label='Experimental'),
        ]
        fig.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(0.88, 0.5),
            ncol=1,
            handlelength=1.8,
        )

        fig.tight_layout(rect=[0, 0, 0.84, 0.95])
        _save_figure_pdf(fig, 'behavioral_accuracy_by_trial_type')
        figures['accuracy_by_trial_type'] = fig

        # === FIGURE 2: Overall Accuracy ===
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        overall_accuracy_values = []
        for group in ['control', 'experimental']:
            group_data = overall_accuracy[overall_accuracy['group'] == group]
            pre_data = group_data[group_data['session'] == 'pre']['accuracy'].values
            post_data = group_data[group_data['session'] == 'post']['accuracy'].values

            if len(pre_data) == 0 or len(post_data) == 0:
                continue

            pre_mean = np.mean(pre_data)
            post_mean = np.mean(post_data)
            pre_sem = np.std(pre_data, ddof=1) / np.sqrt(len(pre_data))
            post_sem = np.std(post_data, ddof=1) / np.sqrt(len(post_data))
            overall_accuracy_values.extend([
                pre_mean - pre_sem,
                pre_mean + pre_sem,
                post_mean - post_sem,
                post_mean + post_sem,
            ])

            x_pos = [0, 1]
            means = [pre_mean, post_mean]
            sems = [pre_sem, post_sem]
            ax.errorbar(
                x_pos,
                means,
                yerr=sems,
                marker='o',
                markersize=4,
                color=colors[group],
                capsize=3,
                capthick=1,
                linewidth=1.0,
                label=f"{group.capitalize()}"
            )

            for subj_id in group_data['subject_id'].unique():
                subj_pre = group_data[(group_data['subject_id'] == subj_id) & (group_data['session'] == 'pre')]['accuracy'].values
                subj_post = group_data[(group_data['subject_id'] == subj_id) & (group_data['session'] == 'post')]['accuracy'].values
                if len(subj_pre) > 0 and len(subj_post) > 0:
                    overall_accuracy_values.extend([subj_pre[0], subj_post[0]])
                    ax.plot([0, 1], [subj_pre[0], subj_post[0]], color=colors[group], alpha=0.25, linewidth=0.8)
                    ax.scatter([0, 1], [subj_pre[0], subj_post[0]], color=marker_color, s=8, alpha=0.5, zorder=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre', 'Post'])
        ax.set_ylabel('Overall Accuracy')
        ax.set_title('Overall Accuracy Across Sessions')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        ax.tick_params(axis='both', which='both', length=3, width=0.8)
        _set_y_limits_with_padding(ax, overall_accuracy_values, pad_fraction=0.12, min_pad=0.003)
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            handlelength=1.8,
        )
        fig.tight_layout(rect=[0, 0, 0.76, 1])
        _save_figure_pdf(fig, 'behavioral_overall_accuracy')
        figures['overall_accuracy'] = fig

        # === FIGURE 3: Timeout Rate ===
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        timeout_values = []
        for group in ['control', 'experimental']:
            group_data = timeout_rate[timeout_rate['group'] == group]
            pre_data = group_data[group_data['session'] == 'pre']['timeout_rate'].values
            post_data = group_data[group_data['session'] == 'post']['timeout_rate'].values

            if len(pre_data) == 0 or len(post_data) == 0:
                continue

            pre_mean = np.mean(pre_data)
            post_mean = np.mean(post_data)
            pre_sem = np.std(pre_data, ddof=1) / np.sqrt(len(pre_data))
            post_sem = np.std(post_data, ddof=1) / np.sqrt(len(post_data))
            timeout_values.extend([
                pre_mean - pre_sem,
                pre_mean + pre_sem,
                post_mean - post_sem,
                post_mean + post_sem,
            ])

            x_pos = [0, 1]
            means = [pre_mean, post_mean]
            sems = [pre_sem, post_sem]
            ax.errorbar(
                x_pos,
                means,
                yerr=sems,
                marker='o',
                markersize=4,
                color=colors[group],
                capsize=3,
                capthick=1,
                linewidth=1.0,
                label=f"{group.capitalize()}"
            )

            for subj_id in group_data['subject_id'].unique():
                subj_pre = group_data[(group_data['subject_id'] == subj_id) & (group_data['session'] == 'pre')]['timeout_rate'].values
                subj_post = group_data[(group_data['subject_id'] == subj_id) & (group_data['session'] == 'post')]['timeout_rate'].values
                if len(subj_pre) > 0 and len(subj_post) > 0:
                    timeout_values.extend([subj_pre[0], subj_post[0]])
                    ax.plot([0, 1], [subj_pre[0], subj_post[0]], color=colors[group], alpha=0.25, linewidth=0.8)
                    ax.scatter([0, 1], [subj_pre[0], subj_post[0]], color=marker_color, s=8, alpha=0.5, zorder=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre', 'Post'])
        ax.set_ylabel('Timeout Rate')
        ax.set_title('Timeout Rate Across Sessions')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        ax.tick_params(axis='both', which='both', length=3, width=0.8)
        _set_y_limits_with_padding(ax, timeout_values, pad_fraction=0.15, min_pad=0.002)
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            handlelength=1.8,
        )
        fig.tight_layout(rect=[0, 0, 0.76, 1])
        _save_figure_pdf(fig, 'behavioral_timeout_rate')
        figures['timeout_rate'] = fig

    return figures


def _plot_reaction_time_figures(summary_df, pre_outlier_trial_data, cleaned_trial_data):
    """Generate publication-quality RT figures for Analysis 1."""
    figures = {}

    with plt.rc_context(_publication_style_rcparams()):
        fig_distractor = _plot_rt_prepost_panel(summary_df, "distractor", "Reaction Time: Distractor Trials")
        _save_figure_pdf(fig_distractor, "behavioral_rt_distractor")
        figures["rt_distractor"] = fig_distractor

        fig_no_distractor = _plot_rt_prepost_panel(summary_df, "no_distractor", "Reaction Time: No-Distractor Trials")
        _save_figure_pdf(fig_no_distractor, "behavioral_rt_no_distractor")
        figures["rt_no_distractor"] = fig_no_distractor

        fig_combined = _plot_rt_combined_barplot(cleaned_trial_data)
        _save_figure_pdf(fig_combined, "behavioral_rt_combined_overall")
        figures["rt_combined_overall"] = fig_combined

        fig_hist = _plot_rt_distribution_histogram(pre_outlier_trial_data)
        _save_figure_pdf(fig_hist, "behavioral_rt_histogram_before_outlier_removal")
        figures["rt_histogram_before_outlier_removal"] = fig_hist

    return figures


def _plot_rt_prepost_panel(summary_df, trial_type, title):
    """Plot pre/post RT summary for one trial type."""
    colors = {"control": "#4C72B0", "experimental": "#DD8452"}
    marker_color = "#333333"

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    plotted_values = []

    subset = summary_df[summary_df["trial_type"] == trial_type].copy()

    for group in ["control", "experimental"]:
        group_data = subset[subset["group"] == group]
        wide = group_data.pivot_table(index="subject_id", columns="session", values="mean_rt_ms")
        wide = wide.dropna(subset=["pre", "post"])
        if wide.empty:
            continue

        pre_values = wide["pre"].to_numpy()
        post_values = wide["post"].to_numpy()
        pre_mean = float(np.mean(pre_values))
        post_mean = float(np.mean(post_values))
        pre_sem = float(np.std(pre_values, ddof=1) / np.sqrt(len(pre_values))) if len(pre_values) > 1 else 0.0
        post_sem = float(np.std(post_values, ddof=1) / np.sqrt(len(post_values))) if len(post_values) > 1 else 0.0

        plotted_values.extend([
            pre_mean - pre_sem,
            pre_mean + pre_sem,
            post_mean - post_sem,
            post_mean + post_sem,
        ])

        ax.errorbar(
            [0, 1],
            [pre_mean, post_mean],
            yerr=[pre_sem, post_sem],
            marker="o",
            markersize=4,
            color=colors[group],
            capsize=3,
            capthick=1,
            linewidth=1.0,
            label=group.capitalize(),
        )

        for subject_id, row in wide.iterrows():
            plotted_values.extend([row["pre"], row["post"]])
            ax.plot([0, 1], [row["pre"], row["post"]], color=colors[group], alpha=0.25, linewidth=0.8)
            ax.scatter([0, 1], [row["pre"], row["post"]], color=marker_color, s=8, alpha=0.5, zorder=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel("Mean RT (ms)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=3, width=0.8)
    _set_y_limits_with_padding(ax, plotted_values, pad_fraction=0.12, min_pad=20.0)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        handlelength=1.8,
    )
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    return fig


def _plot_rt_distribution_histogram(pre_outlier_trial_data):
    """Plot RT distribution before outlier removal, stratified by trial type."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)

    for ax, trial_type in zip(axes, ["no_distractor", "distractor"]):
        subset = pre_outlier_trial_data[pre_outlier_trial_data["trial_type"] == trial_type]["rt_ms"].dropna()
        mean_rt = float(subset.mean())
        sd_rt = float(subset.std(ddof=1))

        ax.hist(subset, bins=28, color="#BFBFBF", edgecolor="white")
        ax.axvline(mean_rt, color="#222222", linewidth=1.0, label="Mean")
        ax.axvline(mean_rt - 3 * sd_rt, color="#C44E52", linewidth=0.9, linestyle="--", label="Mean ± 3 SD")
        ax.axvline(mean_rt + 3 * sd_rt, color="#C44E52", linewidth=0.9, linestyle="--")

        ax.set_title(trial_type.replace("_", " ").title())
        ax.set_xlabel("RT (ms)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)
        ax.tick_params(axis="both", which="both", length=3, width=0.8)

    axes[0].set_ylabel("Trial count")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, handlelength=1.8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _plot_rt_combined_barplot(cleaned_trial_data):
    """Plot overall RT collapsed across trial types as grouped bars with SEM and subject scatter."""
    colors = {"control": "#4C72B0", "experimental": "#DD8452"}

    combined_summary = (
        cleaned_trial_data
        .groupby(["subject_id", "group", "session"], observed=False)
        .agg(mean_rt_ms=("rt_ms", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    session_positions = {"pre": 0.0, "post": 1.0}
    group_offsets = {"control": -0.18, "experimental": 0.18}
    bar_width = 0.32
    plotted_values = []

    rng = np.random.default_rng(42)

    for session in ["pre", "post"]:
        for group in ["control", "experimental"]:
            subset = combined_summary[
                (combined_summary["session"] == session) &
                (combined_summary["group"] == group)
            ].copy()
            if subset.empty:
                continue

            x_center = session_positions[session] + group_offsets[group]
            values = subset["mean_rt_ms"].to_numpy()
            mean_value = float(np.mean(values))
            sem_value = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0

            plotted_values.extend([mean_value - sem_value, mean_value + sem_value])

            ax.bar(
                x_center,
                mean_value,
                width=bar_width,
                color=colors[group],
                alpha=0.75,
                edgecolor="none",
                zorder=1,
            )
            ax.errorbar(
                x_center,
                mean_value,
                yerr=sem_value,
                color="#222222",
                capsize=3,
                linewidth=1.0,
                zorder=3,
            )

            jitter = rng.uniform(-0.045, 0.045, size=len(subset))
            ax.scatter(
                np.full(len(subset), x_center) + jitter,
                values,
                s=14,
                color="#333333",
                alpha=0.65,
                zorder=4,
            )
            plotted_values.extend(values.tolist())

    ax.set_xticks([session_positions["pre"], session_positions["post"]])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel("Mean RT (ms)")
    ax.set_title("Reaction Time Across All Trials")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=3, width=0.8)
    _set_y_limits_with_padding(ax, plotted_values, pad_fraction=0.12, min_pad=20.0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["control"], alpha=0.75, edgecolor="none", label="Control"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["experimental"], alpha=0.75, edgecolor="none", label="Experimental"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        handlelength=1.6,
    )
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    return fig


def _set_y_limits_with_padding(ax, values, pad_fraction=0.08, min_pad=0.002):
    """Set y-limits with explicit padding at both ends of the axis."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return

    data_min = float(np.min(finite_values))
    data_max = float(np.max(finite_values))
    data_range = data_max - data_min
    pad = max(data_range * pad_fraction, min_pad)

    ax.set_ylim(data_min - pad, data_max + pad)


def _publication_style_rcparams():
    """Return matplotlib rcParams for publication-style behavioral figures."""
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


def set_publication_style():
    """Apply publication-style matplotlib settings globally."""
    plt.rcParams.update(_publication_style_rcparams())


def _save_figure_pdf(fig, filename_stem):
    """Save a figure as PDF in the repository figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"{filename_stem}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {output_path}")
