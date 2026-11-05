"""Verify between-group changes, uncertainty, and unadjusted p values."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posthoc_analysis.behavioral import compare_stroop_rt_changes_by_congruency, compare_training_rt_changes_by_condition


class StroopChangeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.changes = {'congruent': {'experimental': [-80., -60., -40.], 'control': [-20., 0., 10.]},
                        'incongruent': {'experimental': [-50., -10., 20.], 'control': [-10., 0., 5.]}}
        rows = []
        for condition, groups in self.changes.items():
            for group, values in groups.items():
                for i, change in enumerate(values):
                    for session, value in [('pre', 1000. + 10*i), ('post', 1000. + 10*i + change)]:
                        rows.append(dict(subject_id=f'{group}_{i}', group=group, session=session,
                                         trial_type=condition, mean_rt_ms=value))
        self.data = pd.DataFrame(rows)

    def compare(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            return compare_stroop_rt_changes_by_congruency(data)

    def test_manual_change_welch_ci_effect_size_and_unadjusted_p(self):
        before = self.data.copy(deep=True)
        result = self.compare(self.data)
        pd.testing.assert_frame_equal(before, self.data)
        table = result['welch_tests'].set_index('trial_type')
        for condition, groups in self.changes.items():
            a,b = np.array(groups['experimental']),np.array(groups['control'])
            diff = a.mean()-b.mean()
            va,vb = a.var(ddof=1)/len(a),b.var(ddof=1)/len(b)
            se=np.sqrt(va+vb); df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
            t=diff/se; p=2*stats.t.sf(abs(t),df)
            row=table.loc[condition]
            self.assertAlmostEqual(row.estimate_ms,diff)
            self.assertAlmostEqual(row.statistic,t)
            self.assertAlmostEqual(row.df,df)
            self.assertAlmostEqual(row.p_value,p)
            self.assertAlmostEqual(row.ci95_low_ms,diff-stats.t.ppf(.975,df)*se)
            self.assertAlmostEqual(row.ci95_high_ms,diff+stats.t.ppf(.975,df)*se)
            self.assertAlmostEqual(row.effect_size,diff/np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2))
        self.assertNotIn('p_holm', table.columns)
        self.assertNotIn('reject_holm_05', table.columns)
        self.assertEqual(result['correction'], 'None; unadjusted p values')
        self.assertEqual(len(result['change_scores']),12)

    def test_training_wrapper_uses_identical_tests_with_distractor_labels(self):
        mapping = {'congruent': 'distractor', 'incongruent': 'no_distractor'}
        training = self.data.replace({'trial_type': mapping})
        before = training.copy(deep=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = compare_training_rt_changes_by_condition(training)
        expected = self.compare(self.data)['welch_tests'].replace({'trial_type': mapping})
        pd.testing.assert_frame_equal(result['welch_tests'].drop(columns='n_control'), expected)
        pd.testing.assert_frame_equal(training, before)
        self.assertNotIn('p_holm', result['welch_tests'])
        self.assertTrue(result['welch_tests'].n_control.eq(3).all())

    def test_training_wrapper_rejects_wrong_conditions(self):
        with self.assertRaisesRegex(ValueError, 'trial_type levels'):
            compare_training_rt_changes_by_condition(self.data)

    def test_row_order_does_not_change_results(self):
        pd.testing.assert_frame_equal(self.compare(self.data)['welch_tests'],
                                      self.compare(self.data.sample(frac=1,random_state=7))['welch_tests'])

    def test_invalid_incomplete_duplicate_or_degenerate_inputs_rejected(self):
        nonfinite=self.data.copy();nonfinite.loc[0,'mean_rt_ms']=np.inf
        degenerate=self.data.assign(mean_rt_ms=1000.)
        for data,message in [(self.data.iloc[1:],'four-cell'),
                             (pd.concat([self.data,self.data.iloc[:1]]),'Duplicate'),
                             (nonfinite,'finite positive'),(degenerate,'Zero change-score variance')]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError,message):
                self.compare(data)


if __name__ == '__main__':
    unittest.main()
