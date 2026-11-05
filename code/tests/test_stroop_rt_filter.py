"""Protect session-pooled Stroop trimming and disjoint overall exclusion counts."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posthoc_analysis.behavioral import summarize_stroop_reaction_time, _remove_stroop_rt_outliers


class StroopRTFilterTests(unittest.TestCase):
    def setUp(self):
        self.trials = pd.DataFrame({
            'subject_id': ['e21'] * 120, 'group': ['experimental'] * 120,
            'session_id': [1] * 120, 'run_id': [1] * 60 + [2] * 60,
            'trial_number': list(range(1, 61)) * 2,
            'trial_type': (['congruent'] * 30 + ['incongruent'] * 30) * 2,
            'stimulus': ['RED'] * 120,
            'ink_color': (['RED'] * 30 + ['BLUE'] * 30) * 2,
            'response': [1] * 120, 'reaction_time_ms': [500.] * 60 + [1000.] * 60,
        })
        self.trials.loc[0, 'reaction_time_ms'] = 650.  # per-run outlier, but retained when pooled
        self.trials.loc[1, 'reaction_time_ms'] = 149.  # no minimum-RT exclusion
        self.trials.loc[60, 'reaction_time_ms'] = 5000.  # pooled-session outlier
        self.trials.loc[61, ['response', 'reaction_time_ms']] = [2, 999999.]
        self.trials.loc[62, ['response', 'reaction_time_ms']] = [3, np.nan]

    def summarize(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            return summarize_stroop_reaction_time(data)

    def test_pools_runs_and_conditions_and_reconciles_all_exclusions(self):
        before = self.trials.copy(deep=True)
        r = self.summarize(self.trials)
        correct = self.trials.loc[self.trials.response.eq(1)]
        values = correct.reaction_time_ms.to_numpy()
        expected_keep = np.abs(values - np.mean(values)) <= 3 * np.std(values, ddof=1)
        keys = ['run_id', 'trial_number']
        self.assertEqual(set(map(tuple, r['trial_data_clean'][keys].to_numpy())),
                         set(map(tuple, correct.loc[expected_keep, keys].to_numpy())))
        report = r['outlier_report']
        self.assertEqual(len(report), 1)
        self.assertEqual(report.iloc[0].n_runs, 2)
        self.assertAlmostEqual(report.iloc[0].session_mean_rt_ms, values.mean())
        self.assertAlmostEqual(report.iloc[0].session_sd_rt_ms, values.std(ddof=1))
        self.assertEqual(r['n_removed_total'], 3)
        self.assertEqual(r['n_retained_trials'], 117)
        self.assertEqual(r['percent_removed_total'], 2.5)
        self.assertTrue({(1, 1), (1, 2)}.issubset(set(map(tuple, r['trial_data_clean'][keys].to_numpy()))))
        self.assertFalse(any(c.startswith('run_mean') or c.startswith('run_sd') for c in r['trial_data_clean']))
        pd.testing.assert_frame_equal(self.trials, before)

    def test_sessions_stay_separate_and_zero_sd_keeps_trials(self):
        post = self.trials.copy()
        post['session_id'] = 5
        post['response'] = 1
        post['reaction_time_ms'] = 20000.
        r = self.summarize(pd.concat([self.trials, post]))
        self.assertEqual(len(r['outlier_report']), 2)
        pre = r['trial_data_clean'].query('session_id == 1')
        self.assertEqual(len(pre), self.summarize(self.trials)['n_retained_trials'])
        self.assertEqual(len(r['trial_data_clean'].query('session_id == 5')), 120)
        self.assertEqual(r['subject_exclusion_summary'].iloc[0].n_runs, 4)
        self.assertEqual(r['subject_exclusion_summary'].iloc[0].n_sessions, 2)

    def test_row_order_invariance(self):
        a = self.summarize(self.trials)
        b = self.summarize(self.trials.sample(frac=1, random_state=3))
        keys = ['subject_id', 'session_id', 'run_id', 'trial_number']
        pd.testing.assert_frame_equal(a['trial_data_clean'].sort_values(keys).reset_index(drop=True),
                                      b['trial_data_clean'].sort_values(keys).reset_index(drop=True))

    def test_malformed_source_and_noncorrect_filter_input_rejected(self):
        bad = self.trials.copy()
        bad.loc[0, 'reaction_time_ms'] = np.nan
        with self.assertRaisesRegex(ValueError, 'must have reaction times'):
            self.summarize(bad)
        with self.assertRaisesRegex(ValueError, '60 rows'):
            self.summarize(self.trials.iloc[:-1])
        bad = self.trials.loc[self.trials.response.ne(3)].assign(session='pre')
        with self.assertRaisesRegex(ValueError, 'only correct trials'):
            _remove_stroop_rt_outliers(bad)


if __name__ == '__main__':
    unittest.main()
