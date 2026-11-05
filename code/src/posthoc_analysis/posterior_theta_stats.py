"""Balanced posterior-theta mixed ANOVAs with explicit repeated factors.

Extends the validated midfrontal orthonormal-contrast method to three repeated
factors. Uses participant means, group-adjusted subject error strata, and
Greenhouse–Geisser corrections; independently audits original-space SS.
"""
from itertools import combinations
from functools import reduce
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import helmert

from .config import EXPECTED_SUBJECTS, get_subject_group

TRIAL_TYPES = ('left_distractor', 'right_distractor', 'no_distractor')
SIDE_MAP = {'left_posterior': 'left', 'right_posterior': 'right'}
MODELS = {'seven_point': tuple(range(7)), 'intervention': tuple(range(1, 6)),
          'pre_post': (0, 6)}


def prepare_posterior_theta_anova(participants):
    """Select independent trial categories and hemispheres from validated summaries."""
    required = {'subject_id', 'group', 'phase', 'session_id', 'session_index',
                'roi', 'condition', 'theta_db', 'n_trials', 'n_runs'}
    keys = ['subject_id', 'session_index', 'roi', 'condition']
    if (required - set(participants) or len(participants) != 3360
            or participants[keys].isna().any().any() or participants.duplicated(keys).any()
            or not np.isfinite(participants.theta_db).all()
            or set(participants.subject_id) != set(EXPECTED_SUBJECTS)
            or not participants.group.eq(participants.subject_id.map(get_subject_group)).all()):
        raise ValueError('Expected complete validated 32 × 7 × 3-ROI × 5-condition posterior summaries.')
    data = participants[participants.roi.isin(SIDE_MAP) & participants.condition.isin(TRIAL_TYPES)].copy()
    data['eeg_side'] = data.roi.map(SIDE_MAP)
    for subject, cell in data.groupby('subject_id'):
        if len(cell) != 42 or set(cell.session_index) != set(range(7)):
            raise ValueError(f'{subject}: expected all 42 Session × Trial type × EEG side cells.')
    phase_expected = np.where(data.session_index.isin([0, 6]), 'assessment', 'intervention')
    session_expected = data.session_index.replace({0: 1, 6: 5})
    if (len(data) != 1344 or not data.phase.eq(phase_expected).all()
            or not data.session_id.eq(session_expected).all()
            or not data.n_trials.gt(0).all() or not data.n_runs.gt(0).all()
            or not data.groupby(['subject_id', 'session_index', 'condition']).n_trials.nunique().eq(1).all()):
        raise ValueError('Invalid phase/session mapping, missing observations, or hemisphere trial-count mismatch.')
    # Count each trial once: left/right EEG measurements are repeated observations.
    phase_counts = data[data.eeg_side.eq('left')].groupby('phase').n_trials.sum().to_dict()
    if phase_counts != {'assessment': 21940, 'intervention': 69000}:
        raise ValueError(f'Trial totals disagree with existing theta extraction: {phase_counts}.')
    print('Prepared 1,344 cells: 32 participants × 7 time points × 3 trial types × 2 EEG sides.')
    print('Between factor: Group (16 BCI, 16 mental rehearsal); repeated factors: Session, Trial type, EEG side.')
    print('DV: 4–8 Hz, 100–500 ms normalized Laplacian theta (dB). No new trial or participant exclusions.')
    return data.sort_values(['subject_id', 'session_index', 'condition', 'eeg_side']).reset_index(drop=True)


def _balanced_mixed_anova(values, group, factor_labels):
    """Estimate complete balanced mixed ANOVA and audit every SS/F independently."""
    import statsmodels.api as sm
    y = np.asarray(values, dtype=float)
    group = np.asarray(group)
    sizes = y.shape[1:]
    n = len(y)
    groups = sorted(set(group))
    if (y.ndim != len(factor_labels) + 1 or not sizes or min(sizes) < 2
            or not np.isfinite(y).all() or group.shape != (n,) or len(groups) != 2
            or n <= 2 or len(set(factor_labels)) != len(factor_labels)
            or len(set(int(np.sum(group == g)) for g in groups)) != 1):
        raise ValueError('Require finite subjects × repeated-factor cube with two equally sized groups.')
    residual_df = n - 2
    subjects_per_group = n // 2
    flat = y.reshape(n, -1)
    terms = [()] + [term for order in range(1, len(sizes) + 1)
                   for term in combinations(range(len(sizes)), order)]
    rows, error_rows, audit_rows = [], [], []
    partition = 0.
    for term in terms:
        bases = [helmert(size, full=False).T if axis in term else np.ones((size, 1)) / np.sqrt(size)
                 for axis, size in enumerate(sizes)]
        basis = reduce(np.kron, bases)
        z = flat @ basis
        k = z.shape[1]
        mean = z.mean(axis=0)
        group_means = {g: z[group == g].mean(axis=0) for g in groups}
        residual = z - np.stack([group_means[g] for g in group])
        ss_error = float(np.sum(residual**2))
        if ss_error <= np.finfo(float).eps:
            raise ValueError(f'{term}: zero subject-error variance; F undefined.')
        df_error = residual_df * k
        covariance = residual.T @ residual / residual_df
        epsilon = (1. if k == 1 else float(np.clip(
            np.trace(covariance)**2 / (k * np.trace(covariance @ covariance)), 1 / k, 1.)))
        ss_group = float(sum(subjects_per_group * np.sum((group_means[g] - mean)**2) for g in groups))
        name = ' × '.join(factor_labels[i] for i in term)
        effects = [('Group', ss_group)] if not term else [
            (name, float(n * np.sum(mean**2))), ('Group × ' + name, ss_group)]
        partition += ss_error + sum(ss for _, ss in effects)
        error_name = 'Subject(Group)' if not term else name + ' × Subject(Group)'
        error_rows.append(dict(error_term=error_name, SS=ss_error, df=df_error, MS=ss_error/df_error,
                               contrast_dimension=k, epsilon_GG=epsilon))

        # Independent original-cell-space marginal decomposition: average absent
        # factors, center present factors; no Helmert/Kronecker basis is reused.
        marginal = y.copy()
        for axis in range(len(sizes)):
            average = marginal.mean(axis=axis+1, keepdims=True)
            marginal = marginal - average if axis in term else average
        repetition = int(np.prod([size for axis, size in enumerate(sizes) if axis not in term]))
        mmean = marginal.mean(axis=0)
        mg = {g: marginal[group == g].mean(axis=0) for g in groups}
        mr = marginal - np.stack([mg[g] for g in group])
        independent_error = float(repetition * np.sum(mr**2))
        independent_group = float(repetition * sum(subjects_per_group * np.sum((mg[g] - mmean)**2) for g in groups))
        independent_main = float(repetition * n * np.sum(mmean**2))
        # Eigenvalue form of epsilon, evaluated in the original cell space.
        residual_matrix = mr.reshape(n, -1)
        eigen = np.linalg.eigvalsh(residual_matrix.T @ residual_matrix / residual_df)
        epsilon_independent = float(np.clip(eigen.sum()**2 / (k * np.sum(eigen**2)), 1/k, 1))
        scalar_model = sm.OLS(z[:, 0], np.column_stack([np.ones(n), np.where(group == groups[0], 1., -1.)])).fit() if k == 1 else None
        for effect, ss in effects:
            fval = (ss/k) / (ss_error/df_error)
            independent_ss = independent_group if effect.startswith('Group') else independent_main
            independent_f = (independent_ss/k) / (independent_error/df_error)
            np.testing.assert_allclose([ss, ss_error, fval, epsilon],
                                       [independent_ss, independent_error, independent_f, epsilon_independent],
                                       rtol=1e-10, atol=1e-10,
                                       err_msg=f'{effect}: independent marginal/eigenvalue validation failed.')
            t_squared_error = np.nan
            if scalar_model is not None:
                t_squared = float(scalar_model.tvalues[1 if effect.startswith('Group') else 0]**2)
                np.testing.assert_allclose(fval, t_squared, rtol=1e-10, atol=1e-10)
                t_squared_error = abs(fval - t_squared)
            rows.append(dict(effect=effect, SS=ss, df_num=k, df_den=df_error, F=fval,
                p_uncorrected=float(stats.f.sf(fval, k, df_error)), epsilon_GG=epsilon,
                df_num_GG=k*epsilon, df_den_GG=df_error*epsilon,
                p_GG=float(stats.f.sf(fval, k*epsilon, df_error*epsilon)),
                partial_eta_squared=ss/(ss+ss_error), error_term=error_name, SS_error=ss_error,
                correction='Greenhouse–Geisser' if k > 1 else 'none (1-df effect)'))
            audit_rows.append(dict(effect=effect, absolute_SS_error=abs(ss-independent_ss),
                absolute_denominator_SS_error=abs(ss_error-independent_error),
                absolute_F_error=abs(fval-independent_f), absolute_epsilon_error=abs(epsilon-epsilon_independent),
                absolute_OLS_t_squared_error=t_squared_error))
    total = float(np.sum((y-y.mean())**2))
    np.testing.assert_allclose(partition, total, rtol=1e-11, atol=1e-10)
    anova = pd.DataFrame(rows)
    np.testing.assert_allclose(anova.partial_eta_squared,
                              anova.F*anova.df_num/(anova.F*anova.df_num+anova.df_den), atol=1e-12)
    qc = dict(n_subjects=n, n_per_group=subjects_per_group, n_cells=int(y.size),
              total_SS=total, partition_SS=partition, SS_partition_error=abs(total-partition),
              n_effects=len(anova), max_independent_F_error=max(r['absolute_F_error'] for r in audit_rows))
    print(f'Validated {len(anova)} fixed effects, {len(error_rows)} subject-error strata; SS partition error {abs(total-partition):.3g}.')
    return dict(anova=anova, error_strata=pd.DataFrame(error_rows), numerical_audit=pd.DataFrame(audit_rows), validation=qc)


def run_posterior_theta_mixed_anova(data, model='seven_point'):
    """Run one 2 × Session × 3 × 2 model with GG correction and explicit factors."""
    if model not in MODELS:
        raise ValueError(f'Unknown model {model!r}; choose {tuple(MODELS)}.')
    sessions = MODELS[model]
    keys = ['subject_id', 'session_index', 'condition', 'eeg_side']
    required = set(keys + ['group', 'theta_db', 'n_trials'])
    if required - set(data):
        raise ValueError(f'Missing ANOVA fields: {required-set(data)}')
    selected = data[data.session_index.isin(sessions)].copy()
    if (len(selected) != 32*len(sessions)*3*2 or selected.duplicated(keys).any()
            or selected[list(required)].isna().any().any()
            or not np.isfinite(selected.theta_db).all()
            or set(selected.condition) != set(TRIAL_TYPES) or set(selected.eeg_side) != {'left', 'right'}
            or set(selected.subject_id) != set(EXPECTED_SUBJECTS)
            or not selected.group.eq(selected.subject_id.map(get_subject_group)).all()):
        raise ValueError(f'{model}: incomplete/duplicate/nonfinite cells or invalid subject/group/factor assignments.')
    columns = pd.MultiIndex.from_product([sessions, TRIAL_TYPES, ['left', 'right']],
                                         names=['session_index', 'condition', 'eeg_side'])
    wide = selected.pivot(index=['subject_id', 'group'], columns=columns.names, values='theta_db').reindex(columns=columns).sort_index()
    if wide.shape != (32, len(columns)) or wide.isna().any().any():
        raise ValueError('All 32 participants must have all repeated cells; no silent exclusion or imputation.')
    y = wide.to_numpy().reshape(32, len(sessions), 3, 2)
    result = _balanced_mixed_anova(y, wide.index.get_level_values('group').to_numpy(),
                                   ['Session', 'Trial type', 'EEG side'])
    result['anova'].insert(0, 'model', model)
    result['input'] = selected
    result['descriptives'] = selected.groupby(['group', 'session_index', 'condition', 'eeg_side']).theta_db.agg(
        mean_db='mean', sd_db='std', n='count').reset_index()
    result['descriptives']['sem_db'] = result['descriptives'].sd_db / np.sqrt(result['descriptives'].n)
    result['validation'].update(model=model, session_levels=list(sessions), participant_exclusions=0,
        factors={'between': ['Group'], 'within': ['Session', 'Trial type', 'EEG side']},
        input_sha256=hashlib.sha256(selected.sort_values(keys).to_csv(index=False).encode()).hexdigest(),
        correction='GG for every multi-df repeated-effect stratum and its Group interaction; pooled within-group residual covariance.',
        multiplicity='No multiplicity adjustment across 15 omnibus effects or across the three overlapping models. Exploratory inference.',
        weighting='Equal participants and equal trial-type/hemisphere cells for marginal effects; each cell retains existing trial-weighted theta.',
        scope='Seven-point model mixes assessment/intervention; intervention and pre/post models analyze phases separately. Original trial-selection and target-position differences retained.')
    print(f'{model}: Group (2) × Session ({len(sessions)}) × Trial type (3) × EEG side (2), 16 participants/group.')
    print('All 15 effects available; use df_num_GG, df_den_GG and p_GG for reporting; raw results also retained.')
    return result


def run_posterior_theta_anova_models(participants):
    """Analyze the seven-point overview plus separate intervention and assessment phases."""
    data = prepare_posterior_theta_anova(participants)
    results = {model: run_posterior_theta_mixed_anova(data, model=model) for model in MODELS}
    if set(results) != set(MODELS) or any(len(r['anova']) != 15 for r in results.values()):
        raise ValueError('Expected three complete 15-effect ANOVA tables.')
    print('Completed three models; GG-adjusted p values address sphericity, not multiplicity. No post hoc tests run.')
    return results


def run_posterior_theta_condition_anova(participants):
    """Fit Group × Condition × Session for bilateral posterior intervention theta.

    Uses existing ``both_distractors`` values, pooled in linear power before dB,
    rather than averaging left- and right-distractor dB values. Both conditions
    retain all intervention outcomes and the original six-electrode ROI.
    """
    required = {'subject_id', 'group', 'phase', 'session_id', 'session_index',
                'roi', 'condition', 'theta_db', 'n_trials', 'n_runs'}
    if required - set(participants):
        raise ValueError(f'Missing posterior summary fields: {required-set(participants)}')
    sessions = tuple(range(1, 6))
    conditions = ('distractor_present', 'distractor_absent')
    mapping = {'both_distractors': conditions[0], 'no_distractor': conditions[1]}
    selected = participants[participants.roi.eq('bilateral_posterior')
                            & participants.session_index.isin(sessions)
                            & participants.condition.isin(mapping)].copy()
    keys = ['subject_id', 'condition', 'session_index']
    if (len(selected) != 320 or selected[list(required)].isna().any().any()
            or selected.duplicated(keys).any() or not np.isfinite(selected.theta_db).all()
            or set(selected.subject_id) != set(EXPECTED_SUBJECTS)
            or not selected.group.eq(selected.subject_id.map(get_subject_group)).all()
            or not selected.phase.eq('intervention').all()
            or not selected.session_id.eq(selected.session_index).all()
            or not selected.n_runs.gt(0).all()
            or not selected.n_runs.eq(np.floor(selected.n_runs)).all()
            or not selected.n_trials.eq(30 * selected.n_runs).all()
            or not selected.groupby(['subject_id', 'session_index']).n_runs.nunique().eq(1).all()
            or selected.n_trials.sum() != 69000):
        raise ValueError('Expected 320 complete bilateral intervention cells, 32 documented participants, '
                         '30 trials/condition/run, matching run counts, and 69,000 retained trials.')
    selected['condition'] = selected.condition.map(mapping)
    selected = selected.sort_values(keys).reset_index(drop=True)
    columns = pd.MultiIndex.from_product([conditions, sessions], names=['condition', 'session_index'])
    wide = selected.pivot(index=['subject_id', 'group'], columns=columns.names,
                          values='theta_db').reindex(columns=columns).sort_index()
    if wide.shape != (32, 10) or wide.isna().any().any():
        raise ValueError('Each participant must have both conditions in all five intervention sessions.')
    result = _balanced_mixed_anova(wide.to_numpy().reshape(32, 2, 5),
                                   wide.index.get_level_values('group').to_numpy(),
                                   ['Condition', 'Session'])
    result['input'] = selected
    result['descriptives'] = selected.groupby(['group', 'condition', 'session_index']).theta_db.agg(
        mean_db='mean', sd_db='std', n='count').reset_index()
    result['descriptives']['sem_db'] = result['descriptives'].sd_db / np.sqrt(result['descriptives'].n)
    subject_means = selected.groupby(['subject_id', 'group', 'condition']).theta_db.mean().unstack('condition')
    subject_means['present_minus_absent'] = subject_means.distractor_present - subject_means.distractor_absent
    result['condition_means'] = subject_means.groupby('group').mean().reset_index()
    result['validation'].update(session_levels=list(sessions), roi='bilateral_posterior',
        channels=['PO7', 'P7', 'P5', 'PO8', 'P8', 'P6'], n_trials=int(selected.n_trials.sum()),
        n_runs=int(selected.n_runs.sum() // 2), participant_exclusions=0,
        factors={'between': ['Group'], 'within': ['Condition', 'Session']},
        input_sha256=hashlib.sha256(selected.to_csv(index=False).encode()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        condition_pooling='Left/right distractor trials pooled as linear power before dB; existing both_distractors summary.',
        correction='Greenhouse–Geisser for Session effects and interactions; raw p values also retained.',
        multiplicity='No adjustment across the seven omnibus tests or earlier exploratory models.',
        scope='Intervention sessions only; all outcomes retained. Original target-position differences between conditions retained.')
    print('Group (2) × Condition (2) × Session (5): 32 participants, 16/group; 320 cells, 69,000 trials, 1,150 runs.')
    print('ROI: PO7/P7/P5/PO8/P8/P6; theta 4–8 Hz, 100–500 ms; original baseline and exclusions retained.')
    print('Distractor-present uses pooled linear power before dB. Report p_GG and corrected degrees of freedom.')
    return result


def save_posterior_theta_anova_audit(results, output_path):
    """Retain numerical/provenance QC; leave ordinary ANOVA tables in the notebook."""
    import scipy
    import statsmodels
    if set(results) != set(MODELS) or any(len(r['anova']) != 15 for r in results.values()):
        raise ValueError('Audit requires all three complete 15-effect results.')
    audit = {}
    for model, result in results.items():
        errors = result['numerical_audit'].drop(columns='effect')
        if not np.isfinite(errors.drop(columns='absolute_OLS_t_squared_error')).all().all() or errors.max().max() > 1e-8:
            raise ValueError(f'{model}: failed numerical audit.')
        audit[model] = result['validation']
        audit[model]['maximum_numerical_errors'] = errors.max().to_dict()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = dict(models=audit, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   versions={'numpy': np.__version__, 'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__},
                   methods_reference='https://search.r-project.org/CRAN/refmans/car/html/Anova.html',
                   posthoc_tests_run=False, target_position_sensitivity_run=False)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    print(f'Saved independent SS/F/epsilon/OLS validation and input/code provenance: {path}')
    return content
