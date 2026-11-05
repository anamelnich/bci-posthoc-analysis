"""Verify Stroop RT mixed-ANOVA contrasts, labels, and unchanged inputs."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posthoc_analysis.behavioral import run_stroop_rt_mixed_anova


class StroopRTMixedTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(319)
        rows = []
        for group in ['control', 'experimental']:
            for subject in range(8):
                baseline = rng.normal(650, 65)
                change = rng.normal(-35 if group == 'experimental' else 0, 22)
                cost = rng.normal(95, 15)
                cost_change = rng.normal(-15 if group == 'experimental' else 0, 8)
                for session in ['pre', 'post']:
                    for condition in ['congruent', 'incongruent']:
                        post, inc = session == 'post', condition == 'incongruent'
                        rows.append(dict(subject_id=f'{group}_{subject}', group=group,
                                         session=session, trial_type=condition,
                                         mean_rt_ms=baseline + post * change + inc * cost
                                         + post * inc * cost_change))
        self.data = pd.DataFrame(rows)

    def anova(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_stroop_rt_mixed_anova(data)

    def test_seven_effects_against_independent_contrast_regressions(self):
        before = self.data.copy(deep=True)
        result = self.anova(self.data)
        pd.testing.assert_frame_equal(before, self.data)
        table = result['anova_table']
        wide = before.pivot(index=['subject_id', 'group'], columns=['session', 'trial_type'], values='mean_rt_ms')
        pre_cost = wide['pre']['incongruent'] - wide['pre']['congruent']
        post_cost = wide['post']['incongruent'] - wide['post']['congruent']
        for effect, contrast in {'Session': wide['post'].mean(axis=1)-wide['pre'].mean(axis=1),
                                 'Congruency': (pre_cost+post_cost)/2,
                                 'Session x Congruency': post_cost-pre_cost}.items():
            fit = ols('value ~ C(group, Sum)', contrast.rename('value').reset_index()).fit()
            for name, coefficient in [(effect, 0), (f'Group x {effect}', 1)]:
                self.assertAlmostEqual(table.loc[name, 'F'], fit.tvalues.iloc[coefficient]**2, places=9)
                self.assertAlmostEqual(table.loc[name, 'PR(>F)'], fit.pvalues.iloc[coefficient], places=10)
        mean = wide.mean(axis=1)
        t, p = stats.ttest_ind(mean.xs('control', level='group'), mean.xs('experimental', level='group'), equal_var=True)
        self.assertAlmostEqual(table.loc['Group', 'F'], t**2)
        self.assertAlmostEqual(table.loc['Group', 'PR(>F)'], p)
        self.assertEqual(len(table), 7)
        self.assertTrue(table.error_df.eq(14).all())
        np.testing.assert_allclose(table.partial_eta_sq, table.F / (table.F + 14))
        self.assertEqual(result['error_terms'].error_term.nunique(), 4)
        self.assertEqual(len(result['cell_summary']), 8)
        self.assertTrue(result['cell_summary'].n_subjects.eq(8).all())

    def test_invalid_or_incomplete_cells_are_not_silently_dropped(self):
        nonpositive = self.data.copy()
        nonpositive.loc[0, 'mean_rt_ms'] = 0
        nonfinite = self.data.copy()
        nonfinite.loc[0, 'mean_rt_ms'] = np.inf
        overall = self.data.iloc[:1].assign(trial_type='overall')
        for data, message in [(self.data.iloc[1:], 'Complete four-cell'),
                              (pd.concat([self.data, self.data.iloc[:1]]), 'Duplicate'),
                              (nonpositive, 'positive milliseconds'), (nonfinite, 'finite'),
                              (pd.concat([self.data, overall]), 'trial_type levels')]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.anova(data)


if __name__ == '__main__':
    unittest.main()
