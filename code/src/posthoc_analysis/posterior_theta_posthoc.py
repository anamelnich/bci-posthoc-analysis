"""Posterior theta follow-ups matching the original midfrontal simple effects.

Adapted from theta_intervention_posthoc.py; retains tests, weighting, validation,
and effect-size conventions while returning tables in memory for the notebook.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import EXPECTED_SUBJECTS, get_subject_group


def run_posterior_theta_group_condition_followups(data, omnibus, correction=None):
    """Average five sessions per participant; run four two-sided simple-effect tests.

    Set correction=None for raw p values, or "holm" for the historical adjustment.

    Within-group condition contrasts use paired t tests and Cohen's dz.
    Between-group condition contrasts use Welch t tests and pooled-SD Cohen's d.
    The difference-in-differences reconstructs the original interaction (pooled
    variance t squared equals F); it is not a fifth new follow-up hypothesis.
    Confidence intervals are marginal 95%, not multiplicity-adjusted intervals.
    """
    if correction not in (None, 'holm'):
        raise ValueError('correction must be None or holm.')
    required = {'subject_id', 'group', 'condition', 'session_id', 'session_index', 'phase', 'roi', 'theta_db', 'n_trials', 'n_runs'}
    if not required.issubset(data):
        raise ValueError(f'Missing input columns: {sorted(required - set(data))}')
    if (len(data) != len(EXPECTED_SUBJECTS) * 2 * 5
            or data[list(required)].isna().any().any()
            or data.duplicated(['subject_id', 'condition', 'session_id']).any()
            or not np.isfinite(data.theta_db).all()
            or set(data.subject_id) != set(EXPECTED_SUBJECTS)
            or not data.group.eq(data.subject_id.map(get_subject_group)).all()
            or set(data.condition) != {'distractor_present', 'distractor_absent'}
            or set(data.session_id) != set(range(1, 6))
            or not data.session_index.eq(data.session_id).all()
            or not data.phase.eq('intervention').all()
            or not data.roi.eq('bilateral_posterior').all()
            or not data.n_runs.gt(0).all()
            or not data.n_runs.eq(np.floor(data.n_runs)).all()
            or not data.n_trials.eq(30 * data.n_runs).all()
            or not data.groupby(['subject_id', 'session_id']).n_runs.nunique().eq(1).all()
            or data.n_trials.sum() != 69000):
        raise ValueError('Expected complete finite configured cohort × two conditions × five intervention sessions.')
    counts = data.groupby(['subject_id', 'group', 'condition']).session_id.nunique()
    if not counts.eq(5).all():
        raise ValueError('Every participant/condition must contain five sessions; no imputation or exclusion allowed.')
    if not {'effect', 'F', 'p_uncorrected'}.issubset(omnibus):
        raise ValueError('Omnibus table requires effect, F and p_uncorrected columns.')
    original = omnibus[omnibus.effect.eq('Group × Condition')]
    if len(original) != 1 or not np.isfinite(original[['F', 'p_uncorrected']]).all().all():
        raise ValueError('Expected exactly one Group × Condition omnibus result.')
    averages = data.groupby(['subject_id', 'group', 'condition'], sort=True).agg(
        theta_db=('theta_db', 'mean'), n_sessions=('session_id', 'nunique')).reset_index()
    wide = averages.pivot(index=['subject_id', 'group'], columns='condition', values='theta_db').sort_index()
    if wide.isna().any().any() or wide.shape != (len(EXPECTED_SUBJECTS), 2):
        raise ValueError('Incomplete paired session-averaged values.')
    subject = wide.reset_index()
    subject['present_minus_absent_db'] = subject.distractor_present - subject.distractor_absent
    by_group = {g: subject[subject.group.eq(g)] for g in ['bci', 'control']}
    if [len(by_group[g]) for g in ['bci', 'control']] != [16, 16]:
        raise ValueError('Expected 16 participants in each group.')
    rows = []; max_t_error = 0.; max_ci_error = 0.
    for group, values in by_group.items():
        a, b = values.distractor_present.to_numpy(), values.distractor_absent.to_numpy()
        delta = a-b; n = len(delta); estimate = float(delta.mean())
        sd = float(delta.std(ddof=1)); se = sd/np.sqrt(n); df = n-1
        if not np.isfinite(se) or se <= 0:
            raise ValueError(f'{group}: degenerate paired-difference variance.')
        result = stats.ttest_rel(a, b, nan_policy='raise')
        ci = stats.t.ppf(.975, df)*se
        max_t_error = max(max_t_error, abs(result.statistic-estimate/se))
        scipy_ci = result.confidence_interval()
        max_ci_error = max(max_ci_error, abs(scipy_ci.low-(estimate-ci)), abs(scipy_ci.high-(estimate+ci)))
        rows.append(dict(contrast=f'Distractor − no distractor within {group}', test='paired t',
            n_a=n, n_b=n, mean_a_db=float(a.mean()), mean_b_db=float(b.mean()), difference_db=estimate,
            se_db=se, ci95_low_db=estimate-ci, ci95_high_db=estimate+ci,
            t=float(result.statistic), df=float(df), p_uncorrected=float(result.pvalue),
            effect_size=estimate/sd, effect_size_type='Cohen dz (SD of paired differences)'))
    for condition in ['distractor_present', 'distractor_absent']:
        a, b = [by_group[g][condition].to_numpy() for g in ['bci', 'control']]
        na, nb = len(a), len(b); va, vb = a.var(ddof=1), b.var(ddof=1)
        estimate = float(a.mean()-b.mean()); se = np.sqrt(va/na+vb/nb)
        if not np.isfinite(se) or se <= 0:
            raise ValueError(f'{condition}: degenerate between-group variance.')
        df = (va/na+vb/nb)**2 / ((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
        result = stats.ttest_ind(a, b, equal_var=False, nan_policy='raise')
        ci = stats.t.ppf(.975, df)*se
        max_t_error = max(max_t_error, abs(result.statistic-estimate/se), abs(result.df-df))
        scipy_ci = result.confidence_interval()
        max_ci_error = max(max_ci_error, abs(scipy_ci.low-(estimate-ci)), abs(scipy_ci.high-(estimate+ci)))
        sd_pooled = np.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2))
        rows.append(dict(contrast=f'BCI − mental rehearsal within {condition}', test='Welch t',
            n_a=na, n_b=nb, mean_a_db=float(a.mean()), mean_b_db=float(b.mean()), difference_db=estimate,
            se_db=float(se), ci95_low_db=float(estimate-ci), ci95_high_db=float(estimate+ci),
            t=float(result.statistic), df=float(df), p_uncorrected=float(result.pvalue),
            effect_size=float(estimate/sd_pooled), effect_size_type='Cohen d (pooled SD)'))
    tests = pd.DataFrame(rows)
    if max_t_error > 1e-10 or max_ci_error > 1e-10:
        raise ValueError('Independent t/df/CI validation failed.')
    if correction == 'holm':
        tests['p_holm'] = multipletests(tests.p_uncorrected, method='holm')[1]
        order = np.argsort(tests.p_uncorrected.to_numpy())
        manual_holm = np.empty(4)
        manual_holm[order] = np.minimum(1, np.maximum.accumulate(tests.p_uncorrected.to_numpy()[order]*np.arange(4, 0, -1)))
        if not np.allclose(manual_holm, tests.p_holm, atol=1e-14):
            raise ValueError('Independent Holm validation failed.')
    # Algebraically the existing interaction, expressed in interpretable dB units.
    a, b = [by_group[g].present_minus_absent_db.to_numpy() for g in ['bci', 'control']]
    interaction_test = stats.ttest_ind(a, b, equal_var=True, nan_policy='raise')
    difference = float(a.mean()-b.mean()); ci = interaction_test.confidence_interval()
    f_error = abs(interaction_test.statistic**2-float(original.F.iloc[0]))
    p_error = abs(interaction_test.pvalue-float(original.p_uncorrected.iloc[0]))
    if f_error > 1e-10 or p_error > 1e-12:
        raise ValueError('Session-averaged difference-in-differences does not reproduce the original interaction.')
    interaction = pd.DataFrame([dict(contrast='(Present − absent): BCI − mental rehearsal',
        difference_db=difference, ci95_low_db=float(ci.low), ci95_high_db=float(ci.high),
        t=float(interaction_test.statistic), df=float(interaction_test.df), F=float(interaction_test.statistic**2),
        p_uncorrected=float(interaction_test.pvalue),
        cohen_d=float(difference/np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)))])
    descriptives = averages.groupby(['group', 'condition']).theta_db.agg(mean_db='mean', sd_db='std', n='count').reset_index()
    descriptives['sem_db'] = descriptives.sd_db/np.sqrt(descriptives.n)
    descriptives['ci95_low_db'] = descriptives.mean_db-stats.t.ppf(.975, descriptives.n-1)*descriptives.sem_db
    descriptives['ci95_high_db'] = descriptives.mean_db+stats.t.ppf(.975, descriptives.n-1)*descriptives.sem_db
    outputs = {'participant_condition_means':averages, 'participant_condition_contrasts':subject,
               'simple_effects':tests, 'interaction_contrast':interaction, 'descriptives':descriptives}
    qc = dict(n_subjects=len(subject), n_per_group=16, n_sessions=5, n_input_cells=len(data),
        weighting='Arithmetic mean of the five session-level dB estimates per participant/condition; equal session and participant weights, matching the omnibus interaction.',
        condition_direction='distractor minus no distractor', group_direction='BCI minus mental rehearsal control',
        tests='Four two-sided simple effects: two within-group paired t tests; two within-condition Welch independent t tests.',
        correction=('None: raw two-sided p values for all four exploratory simple effects.' if correction is None else 'Holm across all four simple effects as one family; no new correction of the already tested omnibus interaction.'),
        intervals='Marginal 95% t confidence intervals, not adjusted for multiplicity.',
        effect_sizes='Paired Cohen dz = difference / SD of paired differences; between-group Cohen d = difference / pooled SD. Effect size intervals not estimated.',
        maximum_independent_t_or_df_error=float(max_t_error), maximum_independent_CI_error=float(max_ci_error),
        interaction_t_squared_F_error=float(f_error), interaction_p_error=float(p_error),
        additional_exclusions=0, pre_post_assessments_included=False, session_specific_tests_run=False,
        input_sha256=hashlib.sha256(data.sort_values(['subject_id','condition','session_id']).to_csv(index=False).encode()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        roi='bilateral_posterior', n_trials=int(data.n_trials.sum()),
        references=['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html',
                    'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html',
                    'https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html'])
    print(f'Exploratory Group × Condition follow-up: {len(subject)} participants, 16/group; five sessions equally weighted.')
    print('Four two-sided simple effects; no Holm correction; raw p values and marginal 95% CIs.' if correction is None else 'Four two-sided simple effects, Holm-adjusted together; marginal 95% CIs.')
    columns=['contrast','difference_db','ci95_low_db','ci95_high_db','t','df','p_uncorrected','effect_size']
    if correction == 'holm': columns.insert(-1,'p_holm')
    print(tests[columns].to_string(index=False))
    print('Original interaction reconstructed as between-group difference in condition contrasts:')
    print(interaction.to_string(index=False))
    return {**outputs, 'validation':qc}
