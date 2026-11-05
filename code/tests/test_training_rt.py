"""Clock, event-boundary, and alignment regressions for training RT."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posthoc_analysis.triggers import (
    compute_training_reaction_times_gdf,
    apply_training_text_rt_fallback,
    load_training_trigger_file,
)
from posthoc_analysis.consolidated import load_and_merge_training_run, validate_all_training_files


class TrainingRTTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.gdf = Path(self.temp.name) / 'e21_20251020125040_training.gdf'
        self.gdf.touch()
        self.codes = [8] * 30 + [32] * 15 + [44] * 15
        text, analysis, events = [], [], []
        for i, code in enumerate(self.codes):
            sample = 3813 + i * 2000
            time = 10000 + i * 4000
            text.extend([[i+1, 4, time-500], [i+1, code, time], [i+1, 64, time+522]])
            analysis.append([i+1, int(code != 8), 1, 1, {8: 0, 32: 2, 44: 4}[code], 0, 650, 99])
            events.extend([[sample-200, 0, 4], [sample, 0, code], [sample+268, 0, 64]])
        self.text_path = self.gdf.with_suffix('.triggers.txt')
        self.analysis_path = self.gdf.with_suffix('.analysis.txt')
        np.savetxt(self.text_path, text, fmt='%d')
        np.savetxt(self.analysis_path, analysis, fmt='%d')
        self.events = np.array(events)
        self.raw = Mock(info={'sfreq': 512.}, ch_names=['Status'], n_times=125000, first_samp=0)

    def compute(self):
        with patch('mne.io.read_raw_gdf', return_value=self.raw), patch('mne.find_events', return_value=self.events) as finder:
            with contextlib.redirect_stdout(io.StringIO()):
                result = compute_training_reaction_times_gdf(self.gdf)
            finder.assert_called_once_with(self.raw, stim_channel='Status', shortest_event=1, verbose='ERROR')
        return result

    def test_gdf_clock_and_last_trial(self):
        result = self.compute()
        self.assertEqual(result.iloc[0].stimulus_sample, 3813)
        self.assertEqual(result.iloc[0].response_sample, 4081)
        self.assertEqual(result.iloc[0].rt_samples, 268)
        self.assertEqual(result.iloc[0].rt_ms, 523.4375)
        self.assertEqual(result.iloc[0].text_rt_ms, 522)
        self.assertEqual(result.iloc[-1].rt_ms, 523.4375)
        self.raw.close.assert_called_once()

    def test_missing_response_does_not_borrow_next_trial_or_text(self):
        self.events = np.delete(self.events, 2, axis=0)
        with self.assertWarnsRegex(RuntimeWarning, 'missing responses'):
            result = self.compute()
        self.assertTrue(pd.isna(result.iloc[0].rt_ms))
        self.assertEqual(result.iloc[0].text_rt_ms, 522)
        self.assertEqual(result.iloc[1].rt_ms, 523.4375)
        self.assertEqual(result.iloc[0].response_count, 0)

    def test_multiple_responses_flag_and_nan(self):
        self.events = np.insert(self.events, 3, [4100, 0, 64], axis=0)
        with self.assertWarnsRegex(RuntimeWarning, 'multiple responses'):
            result = self.compute()
        self.assertTrue(result.iloc[0].alignment_error)
        self.assertTrue(pd.isna(result.iloc[0].rt_ms))
        self.assertEqual(result.iloc[0].response_count, 2)

    def test_text_fallback_only_for_absent_sub50_responses_and_is_idempotent(self):
        result = self.compute()
        original = result.copy(deep=True)
        # Missing responses at zero, 49, 50, and 522 text ms; ambiguous at 2 ms.
        for index, text_rt, count in [(0, 0, 0), (1, 49, 0), (2, 50, 0), (3, 522, 0), (4, 2, 2)]:
            result.loc[index, ['rt_ms', 'rt_samples', 'response_sample']] = np.nan
            result.loc[index, 'text_rt_ms'] = text_rt
            result.loc[index, 'response_count'] = count
            result.loc[index, 'alignment_error'] = count > 1
        pooled = apply_training_text_rt_fallback(result)
        self.assertEqual(pooled.rt_ms.iloc[:2].tolist(), [0, 49])
        self.assertTrue(pooled.rt_ms.iloc[2:5].isna().all())
        self.assertTrue(pooled.gdf_rt_ms.iloc[:5].isna().all())
        self.assertTrue(pooled.rt_samples.iloc[:5].isna().all())
        self.assertEqual(pooled.rt_source.eq('trigger_text_fallback').sum(), 2)
        pd.testing.assert_frame_equal(pooled, apply_training_text_rt_fallback(pooled))
        pd.testing.assert_series_equal(pooled.rt_ms.iloc[5:], original.rt_ms.iloc[5:])
        self.assertTrue(result.rt_ms.iloc[:5].isna().all())

    def test_text_fallback_rejects_tampered_sample_clock(self):
        result = self.compute()
        result.loc[0, 'rt_samples'] = 522
        with self.assertRaisesRegex(ValueError, 'GDF RT/sample fields'):
            apply_training_text_rt_fallback(result)

    def test_fallback_exclusion_accounting_includes_correct_fast_trials(self):
        from posthoc_analysis.behavioral import _build_rt_exclusion_report
        data = self.compute()
        for i in [0, 1]:
            data.loc[i, ['rt_ms', 'rt_samples', 'response_sample']] = np.nan
            data.loc[i, 'response_count'] = 0
            data.loc[i, 'text_rt_ms'] = 2
        data = apply_training_text_rt_fallback(data)
        data['subject_id'], data['group'], data['session'] = 'e21', 'experimental', 'pre'
        data['feedback'] = 1
        data.loc[1, 'feedback'] = 2
        row = _build_rt_exclusion_report(data).iloc[0]
        self.assertEqual(data.rt_ms.lt(150).sum(), 2)
        self.assertEqual(row.n_removed_missing_rt, 0)
        self.assertEqual(row.n_removed_rt_lt_150, 1)
        self.assertEqual(row.n_removed_non_correct, 1)
        self.assertEqual(row.n_retained_before_sd, 58)

    def test_response_before_first_stimulus_is_not_used(self):
        self.events = np.insert(self.events, 1, [3800, 0, 64], axis=0)
        with self.assertWarnsRegex(RuntimeWarning, 'outside trial intervals'):
            result = self.compute()
        self.assertEqual(result.iloc[0].response_count, 1)

    def test_stimulus_order_mismatch_rejected(self):
        self.events[1, 2], self.events[91, 2] = self.events[91, 2], self.events[1, 2]
        with self.assertRaisesRegex(ValueError, 'stimulus order mismatch'):
            self.compute()

    def test_wrong_sampling_rate_and_missing_status_rejected(self):
        self.raw.info = {'sfreq': 1000.}
        with self.assertRaisesRegex(ValueError, '512 Hz'):
            self.compute()
        self.raw.info = {'sfreq': 512.}
        self.raw.ch_names = ['EEG']
        with self.assertRaisesRegex(ValueError, 'missing Status'):
            self.compute()

    def test_bad_condition_counts_rejected(self):
        self.events[1, 2] = 32
        with self.assertRaisesRegex(ValueError, 'expected 60 stimuli'):
            self.compute()

    def test_analysis_side_mismatch_rejected(self):
        table = np.loadtxt(self.analysis_path, dtype=int)
        table[30, 4] = 4
        np.savetxt(self.analysis_path, table, fmt='%d')
        with self.assertRaisesRegex(ValueError, 'task/distractor side'):
            self.compute()

    def test_missing_gdf_and_practice_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, 'no text RT fallback'):
            compute_training_reaction_times_gdf(self.gdf.with_name('e22_20251020125040_training.gdf'))
        with self.assertRaisesRegex(ValueError, 'non-practice'):
            compute_training_reaction_times_gdf(self.gdf.with_name('e21_training_practice.gdf'))

    def test_bad_text_width_and_fractional_time_rejected(self):
        table = np.loadtxt(self.text_path)
        np.savetxt(self.text_path, np.column_stack([table, np.ones(180)]), fmt='%d')
        with self.assertRaisesRegex(ValueError, 'expected 3 columns'):
            load_training_trigger_file(self.text_path)
        table[0, 2] += .5
        np.savetxt(self.text_path, table)
        with self.assertRaisesRegex(ValueError, 'finite integers'):
            load_training_trigger_file(self.text_path)

    def test_training_counts_only_expected_in_sessions_one_and_five(self):
        root = Path(self.temp.name) / 'study'
        for day in range(1, 6):
            (root / 'e21' / f'e21_2025100{day}').mkdir(parents=True)
        result = validate_all_training_files(root)
        mismatches = result['issues']['file_count_mismatches']
        self.assertEqual([(x['session_number'], x['expected_runs']) for x in mismatches], [(1, 8), (5, 4)])

    def test_missing_rt_exclusions_do_not_double_count_incorrect_trials(self):
        from posthoc_analysis.behavioral import _build_rt_exclusion_report
        data = pd.DataFrame({
            'subject_id': ['e21'] * 5, 'group': ['experimental'] * 5,
            'session': ['pre'] * 5, 'feedback': [1, 2, 1, 1, 1],
            'rt_ms': [np.nan, np.nan, 149., 150., 500.],
        })
        row = _build_rt_exclusion_report(data).iloc[0]
        self.assertEqual(row.n_removed_missing_rt, 1)
        self.assertEqual(row.n_removed_non_correct, 1)
        self.assertEqual(row.n_removed_rt_lt_150, 1)
        self.assertEqual(row.n_retained_before_sd, 2)

    def test_consolidation_uses_gdf_not_text(self):
        run = {'subject_id': 'e21', 'session_number': 1, 'run_number': 1,
               'run_id': self.gdf.stem, 'task_type': 'training',
               'trigger_file': self.text_path, 'analysis_file': self.analysis_path}
        with patch('mne.io.read_raw_gdf', return_value=self.raw), patch('mne.find_events', return_value=self.events):
            result = load_and_merge_training_run(run)
        self.assertEqual(len(result), 60)
        self.assertEqual(result.iloc[0].rt_ms, 523.4375)
        self.assertEqual(result.iloc[0].rt_source, 'gdf_status')
        self.assertEqual(result.iloc[0].feedback, 1)

    def test_consolidation_labels_sub50_text_fallback(self):
        self.events = np.delete(self.events, 2, axis=0)
        text = np.loadtxt(self.text_path, dtype=int)
        text[2, 2] = text[1, 2] + 2
        np.savetxt(self.text_path, text, fmt='%d')
        run = {'subject_id': 'e21', 'session_number': 1, 'run_number': 1,
               'run_id': self.gdf.stem, 'task_type': 'training',
               'trigger_file': self.text_path, 'analysis_file': self.analysis_path}
        with patch('mne.io.read_raw_gdf', return_value=self.raw), patch('mne.find_events', return_value=self.events):
            with self.assertWarnsRegex(RuntimeWarning, 'missing responses'):
                result = load_and_merge_training_run(run)
        self.assertEqual(result.iloc[0].rt_ms, 2)
        self.assertEqual(result.iloc[0].rt_source, 'trigger_text_fallback')
        self.assertTrue(pd.isna(result.iloc[0].gdf_rt_ms))
        self.assertEqual(result.iloc[0].response_count, 0)
        self.assertEqual(result.iloc[1].rt_ms, 523.4375)


if __name__ == '__main__':
    unittest.main()
