"""Check pre/post error strata independently of the production ANOVA algebra."""
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
from posthoc_analysis.behavioral import run_behavioral_prepost_mixed_anova

class PrepostMixedAnovaTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(802)
        rows = []
        for group in ['control', 'experimental']:
            for i in range(8):
                baseline = rng.normal(20 if group == 'experimental' else 0, 40)
                change = rng.normal(-8 if group == 'experimental' else 3, 4)
                for session, value in [('pre', baseline), ('post', baseline + change)]:
                    rows.append(dict(subject_id=f'{group}_{i}', group=group, session=session,
                                     distractor_cost_ms=value))
        self.data = pd.DataFrame(rows)

    def run_anova(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_behavioral_prepost_mixed_anova(data, 'distractor_cost_ms')

    def test_independent_participant_mean_and_change_tests(self):
        result = self.run_anova(self.data)
        table = result['anova_table']
        wide = self.data.pivot(index=['subject_id', 'group'], columns='session', values='distractor_cost_ms')
        averages = wide.mean(axis=1)
        t, p = stats.ttest_ind(averages.xs('control',level='group'), averages.xs('experimental',level='group'), equal_var=True)
        self.assertAlmostEqual(table.loc['Group','F'], t*t, places=10)
        self.assertAlmostEqual(table.loc['Group','PR(>F)'], p, places=10)
        changes = (wide.post - wide.pre).rename('change').reset_index()
        reference = ols('change ~ C(group, Sum)', changes).fit()
        for effect, i in [('Session',0), ('Group x Session',1)]:
            self.assertAlmostEqual(table.loc[effect,'F'],reference.tvalues.iloc[i]**2,places=9)
            self.assertAlmostEqual(table.loc[effect,'PR(>F)'],reference.pvalues.iloc[i],places=10)
        self.assertTrue(table.error_df.eq(14).all())
        self.assertEqual(table.error_term.nunique(),2)
        self.assertAlmostEqual(result['diagnostics']['total_sum_sq'],result['diagnostics']['partitioned_sum_sq'],places=8)

    def test_invariance_and_no_mutation(self):
        before = self.data.copy(deep=True)
        expected=self.run_anova(self.data)['anova_table']
        changed=self.data.sample(frac=1,random_state=3).copy()
        changed.distractor_cost_ms += 1000
        actual=self.run_anova(changed)['anova_table']
        np.testing.assert_allclose(actual[['F','PR(>F)','partial_eta_sq']],expected[['F','PR(>F)','partial_eta_sq']],rtol=1e-10,atol=1e-10)
        pd.testing.assert_frame_equal(self.data,before)

    def test_rejects_malformed_design(self):
        bad_cases=[self.data.iloc[1:],pd.concat([self.data,self.data.iloc[:1]]),
                   self.data[self.data.subject_id!='control_0']]
        nonfinite=self.data.copy();nonfinite.loc[0,'distractor_cost_ms']=np.inf;bad_cases.append(nonfinite)
        group_change=self.data.copy();group_change.loc[0,'group']='experimental';bad_cases.append(group_change)
        for data in bad_cases:
            with self.subTest(shape=data.shape), self.assertRaises(ValueError): self.run_anova(data)

if __name__=='__main__': unittest.main()
