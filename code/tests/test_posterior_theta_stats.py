"""Verify four-factor mixed ANOVA strata, GG invariance and missing-cell rejection."""
import contextlib
import io
from itertools import combinations, product
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posthoc_analysis.posterior_theta_stats import (
    _balanced_mixed_anova, run_posterior_theta_mixed_anova,
    run_posterior_theta_condition_anova, TRIAL_TYPES,
)
from posthoc_analysis.config import EXPECTED_SUBJECTS, get_subject_group
from posthoc_analysis.posterior_theta_posthoc import run_posterior_theta_group_condition_followups


class PosteriorThetaAnovaTests(unittest.TestCase):
    def test_condition_model_uses_existing_pooled_power_and_five_sessions(self):
        rng = np.random.default_rng(26)
        rows = []
        for i, subject in enumerate(EXPECTED_SUBJECTS):
            for session in range(1, 6):
                runs = (4 if session == 5 else 8) - int(i < 2 and session == 1)
                for condition in ('both_distractors', 'no_distractor', 'left_distractor', 'right_distractor'):
                    rows.append(dict(subject_id=subject, group=get_subject_group(subject),
                        phase='intervention', session_id=session, session_index=session,
                        roi='bilateral_posterior', condition=condition, theta_db=rng.normal(),
                        n_trials=30*runs, n_runs=runs))
        data = pd.DataFrame(rows)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_posterior_theta_condition_anova(data)
            # Changing separate-side summaries must not alter the pooled analysis.
            changed = data.copy()
            changed.loc[changed.condition.isin(['left_distractor', 'right_distractor']), 'theta_db'] += 1000
            other = run_posterior_theta_condition_anova(changed)
        pd.testing.assert_frame_equal(result['anova'], other['anova'])
        table = result['anova'].set_index('effect')
        self.assertEqual(len(table), 7)
        self.assertEqual(table.loc['Group × Condition × Session', 'df_num'], 4)
        self.assertEqual(table.loc['Group × Condition × Session', 'df_den'], 120)
        # Group × Condition must equal a between-group t test of each person's
        # present-minus-absent contrast averaged over the five sessions.
        contrast = result['input'].groupby(['subject_id', 'group', 'condition']).theta_db.mean().unstack()
        fit = sm.OLS(contrast.distractor_present - contrast.distractor_absent,
                     np.column_stack([np.ones(32), np.where(contrast.index.get_level_values('group') == 'bci', 1., -1.)])).fit()
        self.assertAlmostEqual(table.loc['Group × Condition', 'F'], fit.tvalues.iloc[1]**2, places=10)
        with contextlib.redirect_stdout(io.StringIO()):
            followups = run_posterior_theta_group_condition_followups(result['input'], result['anova'])
            corrected = run_posterior_theta_group_condition_followups(result['input'], result['anova'], correction='holm')
        self.assertEqual(len(followups['simple_effects']), 4)
        self.assertNotIn('p_holm', followups['simple_effects'])
        self.assertTrue((corrected['simple_effects'].p_holm >= corrected['simple_effects'].p_uncorrected).all())
        self.assertAlmostEqual(followups['interaction_contrast'].F.iloc[0], table.loc['Group × Condition', 'F'], places=10)
        wrong_omnibus = result['anova'].copy()
        wrong_omnibus.loc[wrong_omnibus.effect.eq('Group × Condition'), 'F'] += 1
        for bad_input, bad_omnibus in [(result['input'].iloc[1:], result['anova']),
                                       (result['input'], wrong_omnibus),
                                       (result['input'].assign(theta_db=np.nan), result['anova'])]:
            with self.assertRaises(ValueError):
                run_posterior_theta_group_condition_followups(bad_input, bad_omnibus)
        for bad in (data.iloc[1:], pd.concat([data, data.iloc[:1]]),
                    data.assign(phase='assessment'), data.assign(n_trials=1)):
            with self.assertRaises(ValueError):
                run_posterior_theta_condition_anova(bad)

    def fit(self, y, group, labels=('A', 'B', 'C')):
        with contextlib.redirect_stdout(io.StringIO()):
            return _balanced_mixed_anova(y, group, labels)['anova'].set_index('effect')

    def test_all_fifteen_binary_effects_against_direct_difference_regressions(self):
        rng = np.random.default_rng(512)
        group = np.repeat(['bci', 'control'], 16)
        y = rng.normal(size=(32, 2, 2, 2))
        y += rng.normal(0, 8, size=(32, 1, 1, 1))
        y[:16, 1, 1, 1] += 1.5
        result = self.fit(y, group)
        design = np.column_stack([np.ones(32), np.where(group == 'bci', 1., -1.)])
        for r in range(4):
            for term in combinations(range(3), r):
                # Direct signed condition differences, using no estimator basis.
                values = np.zeros(32)
                for indices in product(range(2), repeat=3):
                    sign = np.prod([1 if indices[a] == 1 else -1 for a in term])
                    values += sign * y[(slice(None), *indices)] / 8
                fit = sm.OLS(values, design).fit()
                name = ' × '.join(('A', 'B', 'C')[a] for a in term)
                effects = [('Group', 1)] if not term else [(name, 0), ('Group × ' + name, 1)]
                for effect, coef in effects:
                    self.assertAlmostEqual(result.loc[effect, 'F'], fit.tvalues[coef]**2, places=10)
                    self.assertAlmostEqual(result.loc[effect, 'p_GG'], fit.pvalues[coef], places=10)
                    self.assertEqual(result.loc[effect, 'epsilon_GG'], 1.)

    def test_multilevel_GG_and_F_invariant_to_order_scale_and_offset(self):
        rng = np.random.default_rng(105)
        y = rng.normal(size=(32, 7, 3, 2))
        y *= np.arange(1, 8)[None, :, None, None]
        group = np.repeat(['bci', 'control'], 16)
        reference = self.fit(y, group)
        order = rng.permutation(32)
        changed = (-2.7 * y[order][:, [6, 1, 4, 0, 3, 5, 2]][:, :, [2, 0, 1], ::-1]) + 51.
        observed = self.fit(changed, group[order])
        columns = ['F', 'p_uncorrected', 'p_GG', 'epsilon_GG', 'partial_eta_squared']
        np.testing.assert_allclose(observed[columns], reference[columns], rtol=1e-10, atol=1e-10)
        self.assertEqual(reference.loc['Group × A × B × C', 'df_num'], 12)
        self.assertEqual(reference.loc['Group × A × B × C', 'df_den'], 360)
        self.assertLess(reference.loc['A', 'epsilon_GG'], .9)

    def test_missing_duplicate_nonfinite_and_unequal_groups_are_rejected(self):
        rng = np.random.default_rng(8)
        rows = [dict(subject_id=s, group=get_subject_group(s), session_index=t, condition=c,
                     eeg_side=h, theta_db=rng.normal(), n_trials=10)
                for s, t, c, h in product(EXPECTED_SUBJECTS, range(7), TRIAL_TYPES, ['left', 'right'])]
        data = pd.DataFrame(rows)
        with self.assertRaises(ValueError):
            run_posterior_theta_mixed_anova(data.iloc[1:])
        duplicate = data.copy()
        duplicate.iloc[1] = duplicate.iloc[0]
        with self.assertRaises(ValueError):
            run_posterior_theta_mixed_anova(duplicate)
        nonfinite = data.copy()
        nonfinite.loc[0, 'theta_db'] = np.nan
        with self.assertRaises(ValueError):
            run_posterior_theta_mixed_anova(nonfinite)
        with self.assertRaises(ValueError):
            self.fit(rng.normal(size=(19, 2, 3, 2)), np.array(['bci']*10+['control']*9))


if __name__ == '__main__':
    unittest.main()
