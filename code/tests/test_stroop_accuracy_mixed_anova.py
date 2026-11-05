"""Verify unchanged Stroop accuracy counts and mixed-ANOVA error strata."""
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
from posthoc_analysis.behavioral import (
    create_stroop_accuracy_summary_table,
    run_stroop_accuracy_mixed_anova,
    summarize_stroop_accuracy,
)


class StroopAccuracyMixedTests(unittest.TestCase):
    def setUp(self):
        self.trials = pd.DataFrame({
            'subject_id': ['e21'] * 60, 'group': ['experimental'] * 60,
            'session_id': [1] * 60, 'run_id': [1] * 60,
            'trial_number': np.arange(1, 61),
            'trial_type': ['congruent'] * 30 + ['incongruent'] * 30,
            'stimulus': ['RED'] * 60, 'ink_color': ['RED'] * 30 + ['BLUE'] * 30,
            'response': [1] * 60, 'reaction_time_ms': [500.] * 60,
        })
        rng = np.random.default_rng(43)
        rows = []
        for group in ['control', 'experimental']:
            for subject in range(8):
                baseline = rng.uniform(.65, .8)
                change = rng.normal(.03 if group == 'experimental' else 0, .025)
                cost = rng.normal(-.08, .02)
                cost_change = rng.normal(.01, .01)
                for session in ['pre', 'post']:
                    for condition in ['congruent', 'incongruent']:
                        post, inc = session == 'post', condition == 'incongruent'
                        rows.append(dict(subject_id=f'{group}_{subject}', group=group,
                                         session=session, trial_type=condition,
                                         accuracy=baseline + post * change + inc * cost
                                         + post * inc * cost_change))
        self.summary = pd.DataFrame(rows)

    def summarize(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            return summarize_stroop_accuracy(data)

    def anova(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_stroop_accuracy_mixed_anova(data)

    def test_accuracy_preserves_fast_slow_and_incorrect_trials(self):
        self.trials.loc[:4, 'reaction_time_ms'] = [1., 149., 150., 99999., np.nan]
        self.trials.loc[[1, 3], 'response'] = 2
        self.trials.loc[4, 'response'] = 3
        before = self.trials.copy(deep=True)
        result = self.summarize(self.trials)
        row = result['subject_session_summary'].query("trial_type == 'congruent'").iloc[0]
        self.assertEqual(row.n_correct, 27)
        self.assertEqual(row.n_incorrect, 2)
        self.assertEqual(row.n_timeouts, 1)
        self.assertEqual(row.n_accuracy_trials, 29)
        self.assertAlmostEqual(row.accuracy, 27 / 29)
        pd.testing.assert_frame_equal(self.trials, before)
        pd.testing.assert_frame_equal(result['summary_table'], create_stroop_accuracy_summary_table(before))
        # Pool counts over runs, not an unweighted mean of run-specific ratios.
        second = before.copy()
        second['run_id'] = 2
        second['response'] = 1
        second['reaction_time_ms'] = 500.
        pooled = self.summarize(pd.concat([before, second]))['subject_session_summary']
        self.assertAlmostEqual(pooled.query("trial_type == 'congruent'").iloc[0].accuracy, 57 / 59)

    def test_practice_exclusion_and_invalid_trials(self):
        practice = self.trials.iloc[:24].copy()
        practice['task_type'] = 'stroop_practice'
        data = self.trials.assign(task_type='stroop')
        self.assertEqual(len(self.summarize(pd.concat([data, practice]))['trial_data']), 60)
        malformed = self.trials.iloc[:-1]
        duplicate = pd.concat([self.trials, self.trials.iloc[:1]])
        invalid_response = self.trials.copy()
        invalid_response.loc[0, 'response'] = 4
        wrong_group = self.trials.assign(group='control')
        nonfinite = self.trials.copy()
        nonfinite.loc[0, 'reaction_time_ms'] = np.inf
        zero = self.trials.copy()
        zero.loc[:29, 'response'] = 3
        zero.loc[:29, 'reaction_time_ms'] = np.nan
        for data, message in [(malformed, '60 rows'), (duplicate, 'Duplicate'),
                              (invalid_response, 'Invalid response'), (wrong_group, 'assignments'),
                              (nonfinite, 'finite'), (zero, 'missing accuracy')]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.summarize(data)

    def test_all_seven_effects_against_independent_subject_contrast_tests(self):
        before = self.summary.copy(deep=True)
        result = self.anova(self.summary)
        table = result['anova_table']
        pd.testing.assert_frame_equal(self.summary, before)
        wide = before.pivot(index=['subject_id', 'group'], columns=['session', 'trial_type'], values='accuracy')
        pre_cost = wide['pre']['incongruent'] - wide['pre']['congruent']
        post_cost = wide['post']['incongruent'] - wide['post']['congruent']
        contrasts = {'Session': wide['post'].mean(axis=1) - wide['pre'].mean(axis=1),
                     'Congruency': (pre_cost + post_cost) / 2,
                     'Session x Congruency': post_cost - pre_cost}
        for effect, values in contrasts.items():
            fit = ols('value ~ C(group, Sum)', values.rename('value').reset_index()).fit()
            for label, coefficient in [(effect, 0), (f'Group x {effect}', 1)]:
                self.assertAlmostEqual(table.loc[label, 'F'], fit.tvalues.iloc[coefficient] ** 2, places=9)
                self.assertAlmostEqual(table.loc[label, 'PR(>F)'], fit.pvalues.iloc[coefficient], places=10)
        participant = wide.mean(axis=1)
        t, p = stats.ttest_ind(participant.xs('control', level='group'),
                               participant.xs('experimental', level='group'), equal_var=True)
        self.assertAlmostEqual(table.loc['Group', 'F'], t ** 2)
        self.assertAlmostEqual(table.loc['Group', 'PR(>F)'], p)
        np.testing.assert_allclose(table.partial_eta_sq, table.F / (table.F + 14))
        self.assertTrue(table.error_df.eq(14).all())
        self.assertEqual(len(table), 7)
        self.assertEqual(result['error_terms'].error_term.nunique(), 4)
        self.assertFalse(table.index.str.contains('Distractor').any())

    def test_incomplete_duplicate_out_of_range_and_wrong_condition_rejected(self):
        out_of_range = self.summary.copy()
        out_of_range.loc[0, 'accuracy'] = 1.01
        wrong_condition = self.summary.replace({'trial_type': {'congruent': 'overall'}})
        for data, message in [(self.summary.iloc[1:], 'Complete four-cell'),
                              (pd.concat([self.summary, self.summary.iloc[:1]]), 'Duplicate'),
                              (out_of_range, r'\[0, 1\]'), (wrong_condition, 'trial_type levels')]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.anova(data)


if __name__ == '__main__':
    unittest.main()
