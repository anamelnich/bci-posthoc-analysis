"""Independent checks of constrained sampling, pooled ROC AUC, and validation."""
import contextlib
import io
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from posthoc_analysis.bci_auc_chance import (
    _sample_labels, _sequence_count, compute_bci_auc_shuffle_reference,
    validate_bci_auc_reference_input,
)


class AucChanceTests(unittest.TestCase):
    def setUp(self):
        y = np.tile([0, 1], 30)
        self.data = pd.DataFrame(dict(
            subject_id="e21", group="experimental", session_id=1, run_id=1,
            trial_id=np.arange(1, 61), decoding_trial_index=np.arange(1, 61),
            decoding_task=y, posterior_probability=np.linspace(.1, .9, 60)))

    def run_reference(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            return compute_bci_auc_shuffle_reference(data, n_permutations=500, random_seed=47)

    def test_sampler_matches_exhaustive_legal_sequences(self):
        legal = [x for x in itertools.product([0, 1], repeat=6)
                 if sum(x) == 3 and all(len(list(v)) <= 2 for _, v in itertools.groupby(x))]
        self.assertEqual(_sequence_count(3, 3, 2, 0, 2), len(legal))
        draws = _sample_labels(3, 3, 40000, np.random.default_rng(41), max_streak=2)
        unique, counts = np.unique(draws, axis=0, return_counts=True)
        self.assertEqual(set(map(tuple, unique)), set(legal))
        expected = len(draws) / len(legal)
        self.assertLess(np.max(np.abs(counts - expected)), 6 * np.sqrt(expected))

    def test_full_sampler_counts_streaks_and_symmetry(self):
        draws = _sample_labels(30, 30, 10000, np.random.default_rng(3))
        np.testing.assert_array_equal(draws.sum(axis=1), 30)
        repeats = draws[:, :-3] == draws[:, 1:-2]
        repeats &= draws[:, :-3] == draws[:, 2:-1]
        repeats &= draws[:, :-3] == draws[:, 3:]
        self.assertFalse(repeats.any())
        self.assertLess(np.max(np.abs(draws.mean(axis=0) - .5)), .025)

    def test_constant_scores_all_ties(self):
        self.data.posterior_probability = .5
        result = self.run_reference(self.data)
        np.testing.assert_array_equal(result['subject_null_auc'], .5)
        self.assertEqual(result['group_summary'].iloc[0].p_label_shuffle, 1)

    def test_pooled_auc_not_average_of_run_aucs_and_reproducible(self):
        # Each run individually perfectly discriminates, but their score ranges
        # do not overlap: pooled AUC is .75, whereas mean run AUC is 1.
        first = self.data.copy()
        first.posterior_probability = .1 + .1 * first.decoding_task
        second = first.assign(run_id=2, posterior_probability=.7 + .1 * first.decoding_task)
        data = pd.concat([first, second], ignore_index=True)
        before = data.copy(deep=True)
        result = self.run_reference(data)
        expected = roc_auc_score(data.decoding_task, data.posterior_probability)
        self.assertEqual(expected, .75)
        self.assertEqual(result['subject_session_auc'].iloc[0].auc, expected)
        rng = np.random.default_rng(47)
        labels1 = _sample_labels(30, 30, 500, rng)
        labels2 = _sample_labels(30, 30, 500, rng)
        for i in range(30):
            direct = roc_auc_score(np.r_[labels1[i], labels2[i]], data.posterior_probability)
            self.assertAlmostEqual(result['subject_null_auc'][i, 0], direct, places=14)
        reordered = self.run_reference(data.sample(frac=1, random_state=9))
        np.testing.assert_array_equal(result['subject_null_auc'], reordered['subject_null_auc'])
        pd.testing.assert_frame_equal(data, before)

    def test_equal_participant_weighting_and_holm(self):
        first = self.data.copy()
        first.posterior_probability = first.decoding_task
        second = self.data.assign(subject_id='e22', posterior_probability=.4)
        second2 = second.assign(run_id=2)
        result = self.run_reference(pd.concat([first, second, second2], ignore_index=True))
        self.assertEqual(result['group_summary'].iloc[0].mean_auc, .75)
        np.testing.assert_allclose(result['group_null_auc'][:, 0], result['subject_null_auc'].mean(axis=1))
        group = result['group_summary']
        self.assertTrue((group.p_label_shuffle_holm >= group.p_label_shuffle).all())
        self.assertTrue((group.p_label_shuffle > 0).all())

    def test_bad_data_rejected_before_integer_conversion(self):
        bad_cases = []
        fractional = self.data.astype({'decoding_task': float}); fractional.loc[0, 'decoding_task'] = .3
        bad_cases.append(fractional)
        bad_cases.append(pd.concat([self.data, self.data.iloc[:1]], ignore_index=True))
        nonfinite = self.data.copy(); nonfinite.loc[0, 'posterior_probability'] = np.inf
        bad_cases.append(nonfinite)
        partial = self.data.copy(); partial.loc[0, 'decoding_task'] = np.nan
        bad_cases.append(partial)
        misaligned = self.data.copy(); misaligned.loc[0, 'decoding_trial_index'] = 2
        bad_cases.append(misaligned)
        blocked = self.data.copy(); blocked.decoding_task = np.repeat([0, 1], 30)
        bad_cases.append(blocked)
        for data in bad_cases:
            with self.subTest(), self.assertRaises(ValueError):
                validate_bci_auc_reference_input(data)


if __name__ == '__main__':
    unittest.main()
