"""Participant-level descriptions, repeated-measures tests and feedback RT figure."""
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import helmert
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

from .feedback_rt import FEEDBACK, TYPES

CONDITIONS = ['overall', 'distractor_present', 'distractor_absent']
FEEDBACK_ORDER = ['positive', 'negative', 'ambivalent']


def summarize_feedback_rt(retained, subject_exclusions):
    """Pool trials within participants first; preserve absent cells as NaN."""
    required = {'subject_id', 'group', 'current_rt_ms', 'current_trial_type', 'previous_bci_feedback'}
    if required - set(retained) or not np.isfinite(retained.current_rt_ms).all() or retained.current_rt_ms.lt(150).any():
        raise ValueError('Expected retained valid RT >=150 ms with pair metadata.')
    roster = subject_exclusions.loc[~subject_exclusions.subject_excluded, ['subject_id', 'group']]
    if not set(retained.subject_id) <= set(roster.subject_id) or roster.subject_id.duplicated().any():
        raise ValueError('Retained trials and included-subject roster disagree.')
    subject_rows = []
    for subject in roster.itertuples(index=False):
        block = retained.loc[retained.subject_id.eq(subject.subject_id)]
        categories = FEEDBACK_ORDER if subject.group == 'experimental' else ['unconditional']
        for condition in CONDITIONS:
            trials = block if condition == 'overall' else block.loc[block.current_trial_type.map(TYPES).eq(condition)]
            for feedback in categories:
                cell = trials if feedback == 'unconditional' else trials.loc[trials.previous_bci_feedback.map(FEEDBACK).eq(feedback)]
                subject_rows.append(dict(subject_id=subject.subject_id, group=subject.group,
                    condition=condition, previous_feedback=feedback, n_trials=len(cell), mean_rt_ms=cell.current_rt_ms.mean()))
    subjects = pd.DataFrame(subject_rows, columns=['subject_id', 'group', 'condition', 'previous_feedback', 'n_trials', 'mean_rt_ms'])
    group_rows = []
    for group, categories in [('experimental', FEEDBACK_ORDER), ('control', ['unconditional'])]:
        for condition, feedback in itertools.product(CONDITIONS, categories):
            block = subjects.loc[subjects.group.eq(group) & subjects.condition.eq(condition) & subjects.previous_feedback.eq(feedback)]
            values = block.mean_rt_ms.dropna()
            n = len(values)
            mean = values.mean()
            sd = values.std(ddof=1) if n > 1 else np.nan
            sem = sd / np.sqrt(n) if n > 1 else np.nan
            half = stats.t.ppf(.975, n - 1) * sem if n > 1 else np.nan
            group_rows.append(dict(group=group, condition=condition, previous_feedback=feedback,
                n_subjects=n, n_trials=int(block.n_trials.sum()), mean_trials_per_subject=block.loc[block.mean_rt_ms.notna(), 'n_trials'].mean(),
                mean_rt_ms=mean, sd_ms=sd, sem_ms=sem, ci95_low_ms=mean-half, ci95_high_ms=mean+half))
    groups = pd.DataFrame(group_rows)
    for group in ['experimental', 'control']:
        expected = int(retained.group.eq(group).sum())
        actual = int(subjects.loc[subjects.group.eq(group) & subjects.condition.eq('overall'), 'n_trials'].sum())
        if expected != actual:
            raise ValueError(f'{group}: descriptive trial counts do not conserve retained trials.')
    print('Participant-averaged RT summaries (t-distribution 95% CIs):\n' + groups.to_string(index=False))
    return dict(bci_subject_summary=subjects.loc[subjects.group.eq('experimental')].copy(),
                bci_group_summary=groups.loc[groups.group.eq('experimental')].copy(),
                control_subject_summary=subjects.loc[subjects.group.eq('control')].copy(),
                control_group_summary=groups.loc[groups.group.eq('control')].copy(),
                trial_count_summary=subjects[['subject_id', 'group', 'condition', 'previous_feedback', 'n_trials']].copy())


def feedback_sphericity(values):
    """Mauchly W and GG epsilon for three repeated levels (two contrasts).

    Applied to feedback means or to present-minus-absent differences for the
    3 x 2 interaction. In contrast dimension q=2, the second-order Mauchly
    correction vanishes: chi-square = -(n-2)*log(W), df=2. Contrast scaling
    does not affect W or epsilon. Degenerate covariance is reported explicitly.
    Reference: https://pingouin-stats.org/generated/pingouin.sphericity.html
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError('Sphericity requires finite subjects x 3 repeated levels.')
    n = len(values)
    result = dict(mauchly_W=np.nan, mauchly_chi2=np.nan, mauchly_df=2,
                  mauchly_p=np.nan, epsilon_gg=np.nan, sphericity_status='unavailable_n_lt_3')
    if n >= 3:
        covariance = np.cov(values @ helmert(3).T, rowvar=False, ddof=1)
        eigen = np.maximum(np.linalg.eigvalsh(covariance), 0)
        if eigen.sum() <= np.finfo(float).eps:
            result['sphericity_status'] = 'unavailable_zero_covariance'
        else:
            epsilon = float(np.clip(eigen.sum()**2 / (2 * (eigen**2).sum()), .5, 1))
            w = float(np.clip(np.prod(eigen) / (eigen.mean()**2), 0, 1))
            chi = -(n-2) * np.log(w) if w > 0 else np.inf
            p = float(stats.chi2.sf(chi, 2))
            result.update(mauchly_W=w, mauchly_chi2=chi, mauchly_p=p, epsilon_gg=epsilon,
                          sphericity_status='violated' if p < .05 else 'not_rejected')
    print(f'Sphericity n={n}: {result["sphericity_status"]}, W={result["mauchly_W"]:.6g}, '
          f'p={result["mauchly_p"]:.6g}, epsilon={result["epsilon_gg"]:.6g}.')
    return result


def run_feedback_rt_statistics(subject_summary):
    """Four complete-cell RM ANOVAs and pairwise-complete paired t-tests.

    Session/run/group are never model factors. Post hoc Holm families are the
    three requested contrasts separately within each current-trial condition.
    """
    required = {'subject_id', 'group', 'condition', 'previous_feedback', 'mean_rt_ms'}
    if required - set(subject_summary) or not subject_summary.group.eq('experimental').all():
        raise ValueError('Statistics accept BCI participant means only.')
    if subject_summary.duplicated(['subject_id', 'condition', 'previous_feedback']).any():
        raise ValueError('Duplicate participant condition means; individual trials are not ANOVA inputs.')
    models, membership, posthoc = [], [], []
    for model in ['overall', 'feedback_x_trial_type', 'distractor_present', 'distractor_absent']:
        two_way = model == 'feedback_x_trial_type'
        conditions = CONDITIONS[1:] if two_way else [model]
        sub = subject_summary.loc[subject_summary.condition.isin(conditions)]
        columns = pd.MultiIndex.from_product([FEEDBACK_ORDER, conditions], names=['previous_feedback', 'condition'])
        wide = sub.pivot(index='subject_id', columns=['previous_feedback', 'condition'], values='mean_rt_ms').reindex(columns=columns)
        complete = wide.notna().all(axis=1)
        for sid, row in wide.iterrows():
            membership.append(dict(model=model, subject_id=sid, included=bool(complete.loc[sid]),
                missing_cells=';'.join(f'{a}/{b}' for (a, b), val in row.items() if pd.isna(val))))
        omitted = wide.index[~complete].tolist()
        wide = wide.loc[complete]
        n = len(wide)
        print(f'{model}: n={n}; omitted {len(omitted)} incomplete participants: {omitted}.')
        if n < 2:
            models.append(dict(model=model, effect='unavailable', n_subjects=n, status='fewer_than_two_complete_subjects'))
            continue
        long = sub.loc[sub.subject_id.isin(wide.index)].copy()
        within = ['previous_feedback', 'condition'] if two_way else ['previous_feedback']
        table = AnovaRM(long, depvar='mean_rt_ms', subject='subject_id', within=within).fit().anova_table
        cube = wide.to_numpy().reshape(n, 3, len(conditions))
        for effect, row in table.iterrows():
            fval, df1, df2 = float(row['F Value']), float(row['Num DF']), float(row['Den DF'])
            if effect == 'condition':
                spher = dict(mauchly_W=np.nan, mauchly_chi2=np.nan, mauchly_df=0, mauchly_p=np.nan,
                             epsilon_gg=1., sphericity_status='automatic_two_levels')
            else:
                values = cube[:, :, 0] - cube[:, :, 1] if ':' in effect else cube.mean(axis=2)
                spher = feedback_sphericity(values)
            eps = spher['epsilon_gg']
            corrected = spher['sphericity_status'] == 'violated'
            p_unc = float(stats.f.sf(fval, df1, df2))
            p_gg = float(stats.f.sf(fval, df1*eps, df2*eps)) if np.isfinite(eps) else np.nan
            models.append(dict(model=model, effect=effect, n_subjects=n, F=fval, df1=df1, df2=df2,
                p_uncorrected=p_unc, p_gg=p_gg, df1_gg=df1*eps, df2_gg=df2*eps,
                partial_eta_squared=fval*df1/(fval*df1+df2), **spher,
                correction='Greenhouse-Geisser' if corrected else 'none',
                df1_reported=df1*eps if corrected else df1, df2_reported=df2*eps if corrected else df2,
                p_reported=p_gg if corrected else p_unc, status='ok'))
    for condition in CONDITIONS:
        wide = subject_summary.loc[subject_summary.condition.eq(condition)].pivot(index='subject_id', columns='previous_feedback', values='mean_rt_ms').reindex(columns=FEEDBACK_ORDER)
        family = []
        for first, second in [('negative', 'positive'), ('negative', 'ambivalent'), ('positive', 'ambivalent')]:
            paired = wide[[first, second]].dropna()
            diff = paired[first] - paired[second]
            n = len(diff)
            mean = diff.mean()
            sd = diff.std(ddof=1) if n > 1 else np.nan
            half = stats.t.ppf(.975, n-1)*sd/np.sqrt(n) if n > 1 else np.nan
            test = stats.ttest_rel(paired[first], paired[second]) if n > 1 else None
            family.append(dict(condition=condition, first=first, second=second, n_pairs=n,
                mean_difference_ms=mean, ci95_low_ms=mean-half, ci95_high_ms=mean+half,
                ci_label='unadjusted 95% paired-difference CI', t=float(test.statistic) if test else np.nan,
                df=n-1 if n else np.nan, p_raw=float(test.pvalue) if test else np.nan,
                cohens_dz=mean/sd if sd > 0 else np.nan, subject_ids=';'.join(paired.index)))
        p = np.array([row['p_raw'] for row in family])
        # Preserve a three-test family even when an unavailable test has no p.
        adjusted = multipletests(np.where(np.isfinite(p), p, 1), method='holm')[1]
        for row, adj, raw in zip(family, adjusted, p):
            row['p_holm'] = float(adj) if np.isfinite(raw) else np.nan
            row['significant_holm_05'] = bool(np.isfinite(raw) and adj < .05)
        posthoc.extend(family)
    anovas, comparisons = pd.DataFrame(models), pd.DataFrame(posthoc)
    print('ANOVAs (use p_reported and reported df):\n' + anovas.to_string(index=False))
    print('Paired post hoc tests, Holm within each three-comparison family:\n' + comparisons.drop(columns='subject_ids').to_string(index=False))
    return dict(anova_results=anovas, anova_membership=pd.DataFrame(membership), posthoc_comparisons=comparisons)


def plot_feedback_next_trial_rt(descriptions, output_path):
    """Three panels of participant means, mean +/- SEM, and control reference."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    subjects = descriptions['bci_subject_summary']
    groups = descriptions['bci_group_summary']
    control = descriptions['control_group_summary']
    if groups.duplicated(['condition', 'previous_feedback']).any() or not set(CONDITIONS) <= set(control.condition):
        raise ValueError('Figure needs unique feedback summaries and three control summaries.')
    path = Path(output_path)
    if path.suffix.lower() != '.pdf':
        raise ValueError('Main figure must be a vector PDF.')
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = ['#3B738E', '#B77549', '#82749C']
    rc = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
          'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
          'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
          'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(7.05, 3.05), sharey=True)
        extents = []
        for i, (ax, condition, title) in enumerate(zip(axes, CONDITIONS, ['Overall', 'Distractor present', 'Distractor absent'])):
            wide = subjects.loc[subjects.condition.eq(condition)].pivot(index='subject_id', columns='previous_feedback', values='mean_rt_ms').reindex(columns=FEEDBACK_ORDER)
            for sid, row in wide.iterrows():
                # NaNs break lines rather than connecting across an absent middle cell.
                ax.plot(range(3), row.to_numpy(), color='.65', alpha=.4, lw=.45, zorder=1)
                ax.scatter(range(3), row.to_numpy(), s=6, c=colors, alpha=.55, linewidths=0, zorder=2)
            extents.extend(wide.to_numpy().ravel())
            block = groups.loc[groups.condition.eq(condition)].set_index('previous_feedback').reindex(FEEDBACK_ORDER)
            for x, (feedback, row) in enumerate(block.iterrows()):
                ax.errorbar(x, row.mean_rt_ms, yerr=row.sem_ms, color=colors[x], marker='o', ms=4.2,
                            lw=1.1, capsize=2.3, markeredgecolor='white', markeredgewidth=.4, zorder=4)
                extents.extend([row.mean_rt_ms-row.sem_ms, row.mean_rt_ms+row.sem_ms])
            ref = control.loc[control.condition.eq(condition)].iloc[0]
            if np.isfinite(ref.mean_rt_ms):
                ax.axhline(ref.mean_rt_ms, color='.4', ls=(0, (4, 3)), lw=.8, zorder=0)
                extents.append(ref.mean_rt_ms)
            ns = '/'.join(str(int(n)) for n in block.n_subjects)
            ax.set_title(f'{title}\nBCI n = {ns}; control n = {int(ref.n_subjects)}', fontsize=7, pad=8)
            ax.set_xticks(range(3), [s.capitalize() for s in FEEDBACK_ORDER])
            ax.set_xlim(-.3, 2.3)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(direction='out', length=3)
            ax.text(-.13, 1.14, 'abc'[i], transform=ax.transAxes, fontsize=9, fontweight='bold')
        extents = np.asarray(extents, dtype=float)
        extents = extents[np.isfinite(extents)]
        if not len(extents):
            plt.close(fig)
            raise ValueError('No finite participant or group means to plot.')
        pad = max(15, .08*np.ptp(extents))
        axes[0].set_ylim(extents.min()-pad, extents.max()+pad)
        axes[0].set_ylabel('RT on trial n+1 (ms)')
        fig.supxlabel('Previous BCI feedback', y=.17, fontsize=7)
        handles = [Line2D([], [], marker='o', color='.25', lw=1, ms=4, label='BCI mean ± SEM; small points: participants'),
                   Line2D([], [], color='.4', lw=.8, ls='--', label='Mental rehearsal: unconditional mean')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.005), frameon=False, fontsize=6.5, ncol=1)
        fig.subplots_adjust(left=.09, right=.985, bottom=.31, top=.77, wspace=.22)
        fig.savefig(path, bbox_inches='tight')
        fig.savefig(path.with_suffix('.png'), dpi=350, bbox_inches='tight')
    print(f'Saved participant-averaged RT figure: {path}; shared y limits {axes[0].get_ylim()}.')
    return fig


def save_feedback_rt_tables(output_dir, *table_collections):
    """Save requested reproducible CSV tables after validating table types."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = set()
    for collection in table_collections:
        for name, table in collection.items():
            if not isinstance(table, pd.DataFrame) or name in names or Path(name).name != name:
                raise ValueError(f'Invalid or duplicate output table: {name}.')
            names.add(name)
            table.to_csv(output / f'{name}.csv', index=False)
    print(f'Saved {len(names)} validated analysis tables under {output}.')
    return output


def validate_feedback_rt_results(pairs, cleaning, descriptions, statistics):
    """Cross-check count conservation and ANOVA F via orthonormal contrasts."""
    retained = cleaning['retained_trials']
    report = cleaning['subject_exclusions']
    if report.candidate_pairs.sum() != len(pairs):
        raise ValueError('Subject candidate counts disagree with paired data.')
    if report.loc[~report.subject_excluded, 'retained_pairs'].sum() != len(retained):
        raise ValueError('Final trial count differs from included-subject report.')
    if retained.subject_excluded.any() or not retained.exclusion_reason.eq('retained').all():
        raise ValueError('Excluded observations entered descriptive results.')
    summary = descriptions['bci_subject_summary']
    checks = []
    for row in statistics['anova_results'].itertuples(index=False):
        if row.status != 'ok':
            continue
        conditions = CONDITIONS[1:] if row.model == 'feedback_x_trial_type' else [row.model]
        columns = pd.MultiIndex.from_product([FEEDBACK_ORDER, conditions])
        wide = summary.loc[summary.condition.isin(conditions)].pivot(
            index='subject_id', columns=['previous_feedback', 'condition'], values='mean_rt_ms').reindex(columns=columns).dropna()
        n = len(wide)
        cube = wide.to_numpy().reshape(n, 3, len(conditions))
        if row.effect == 'condition':
            values = cube.mean(axis=1)
        elif ':' in row.effect:
            values = cube[:, :, 0] - cube[:, :, 1]
        else:
            values = cube.mean(axis=2)
        contrasts = values @ helmert(values.shape[1]).T
        ss_effect = n * np.square(contrasts.mean(axis=0)).sum()
        ss_error = np.square(contrasts - contrasts.mean(axis=0)).sum()
        independent_f = ss_effect * (n-1) / ss_error
        if n != row.n_subjects or not np.isclose(independent_f, row.F, rtol=1e-9, atol=1e-10):
            raise ValueError(f'Independent ANOVA cross-check failed: {row.model}/{row.effect}.')
        checks.append(dict(model=row.model, effect=row.effect, n_subjects=n,
                           statsmodels_F=row.F, independent_contrast_F=independent_f, passed=True))
    for condition, family in statistics['posthoc_comparisons'].groupby('condition'):
        if len(family) != 3:
            raise ValueError(f'{condition}: expected a three-comparison Holm family.')
        p = family.p_raw.to_numpy()
        order = np.argsort(np.where(np.isfinite(p), p, 1))
        expected = np.minimum(1, np.maximum.accumulate(np.where(np.isfinite(p[order]), p[order], 1) * [3, 2, 1]))
        observed = family.p_holm.to_numpy()[order]
        if not np.allclose(expected[np.isfinite(observed)], observed[np.isfinite(observed)]):
            raise ValueError(f'{condition}: Holm correction cross-check failed.')
    print(f'Independent numerical QA passed: {len(checks)} ANOVA effects, three Holm families, '
          f'{len(pairs):,} candidate pairs and {len(retained):,} final trials conserved.')
    return pd.DataFrame(checks)
