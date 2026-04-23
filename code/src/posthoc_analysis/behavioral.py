"""Behavioral analysis for accuracy and timeout outcomes."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from .config import PROJECT_ROOT, BCI_GROUP_SUBJECTS


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
    with plt.rc_context({
        'font.family': 'Arial',
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.8,
        'figure.titlesize': 8,
    }):
        # Muted professional palette
        colors = {'control': '#4C72B0', 'experimental': '#DD8452'}
        marker_color = '#333333'

        # === FIGURE 1: Accuracy by Trial Type (Distractor vs No Distractor) ===
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        fig.suptitle('Accuracy for BCI and Control Groups', fontsize=8, fontweight='bold')

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
                        ax.plot([0, 1], [subj_pre[0], subj_post[0]], color=colors[group], alpha=0.25, linewidth=0.8)
                        ax.scatter([0, 1], [subj_pre[0], subj_post[0]], color=marker_color, s=8, alpha=0.5, zorder=2)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Pre', 'Post'])
            ax.set_ylabel('Accuracy')
            ax.set_ylim([0.90, 1.00])
            ax.set_title(f"{trial_type.replace('_', ' ').title()} Trials")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.8)
            ax.spines['left'].set_linewidth(0.8)
            ax.tick_params(axis='both', which='both', length=3, width=0.8)
            ax.legend(frameon=False, loc='upper left')

        figures['accuracy_by_trial_type'] = fig

        # === FIGURE 2: Overall Accuracy ===
        fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
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
                    ax.plot([0, 1], [subj_pre[0], subj_post[0]], color=colors[group], alpha=0.25, linewidth=0.8)
                    ax.scatter([0, 1], [subj_pre[0], subj_post[0]], color=marker_color, s=8, alpha=0.5, zorder=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre', 'Post'])
        ax.set_ylabel('Overall Accuracy')
        ax.set_ylim([0.90, 1.00])
        ax.set_title('Overall Accuracy Across Sessions')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        ax.tick_params(axis='both', which='both', length=3, width=0.8)
        ax.legend(frameon=False, loc='upper left')
        figures['overall_accuracy'] = fig

        # === FIGURE 3: Timeout Rate ===
        fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
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
                    ax.plot([0, 1], [subj_pre[0], subj_post[0]], color=colors[group], alpha=0.25, linewidth=0.8)
                    ax.scatter([0, 1], [subj_pre[0], subj_post[0]], color=marker_color, s=8, alpha=0.5, zorder=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre', 'Post'])
        ax.set_ylabel('Timeout Rate')
        ax.set_ylim([-0.0015, 0.03])
        ax.set_yticks([0.00, 0.01, 0.02, 0.03])
        ax.set_title('Timeout Rate Across Sessions')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        ax.tick_params(axis='both', which='both', length=3, width=0.8)
        ax.legend(frameon=False, loc='upper left')
        figures['timeout_rate'] = fig

    return figures