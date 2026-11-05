"""Missingness denominators, diagnostic boundaries, and ambiguity handling."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posthoc_analysis.rt_qc import summarize_missing_training_responses


class MissingResponseSummaryTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'subject_id': ['a'] * 3 + ['b'] * 3,
            'session_id': [1, 1, 5, 1, 5, 5],
            'group': ['control'] * 3 + ['experimental'] * 3,
            'trial': [1, 2, 1, 1, 1, 2],
            'feedback': [1, 2, 1, 1, 2, 3],
            'stimulus_trigger': [8, 32, 44, 8, 32, 44],
            'rt_ms': [np.nan, np.nan, 51., 149., 151., 2000.],
            'text_rt_ms': [1., 49., 50., 149., 150., 2000.],
            'response_count': [0, 0, 1, 1, 1, 1],
            'rt_source': ['gdf_status'] * 6,
        })

    def test_cutoff_boundaries_and_all_trial_denominators(self):
        before = self.data.copy(deep=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = summarize_missing_training_responses(self.data)
        bands = result['by_text_rt']
        self.assertEqual(bands.n_trials.tolist(), [2, 2, 2])
        self.assertEqual(bands.n_missing.tolist(), [2, 0, 0])
        self.assertEqual(bands.missing_percent.tolist(), [100., 0., 0.])
        for table in result.values():
            self.assertEqual(table.n_trials.sum(), 6)
            self.assertEqual(table.n_missing.sum(), 2)
        pd.testing.assert_frame_equal(self.data, before)

    def test_ambiguous_response_is_not_counted_as_absent(self):
        self.data.loc[0, 'response_count'] = 2
        with self.assertRaisesRegex(ValueError, 'ambiguous responses'):
            summarize_missing_training_responses(self.data)

    def test_pooled_text_does_not_hide_missing_hardware_responses(self):
        self.data['gdf_rt_ms'] = self.data.rt_ms
        self.data.loc[:1, 'rt_ms'] = self.data.loc[:1, 'text_rt_ms']
        self.data.loc[:1, 'rt_source'] = 'trigger_text_fallback'
        with contextlib.redirect_stdout(io.StringIO()):
            summary = summarize_missing_training_responses(self.data)
        self.assertEqual(summary['by_text_rt'].n_missing.tolist(), [2, 0, 0])

    def test_invalid_diagnostic_clock_rejected(self):
        self.data.loc[0, 'text_rt_ms'] = np.inf
        with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
            summarize_missing_training_responses(self.data)


if __name__ == '__main__':
    unittest.main()
