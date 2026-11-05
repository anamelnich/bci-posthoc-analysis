"""Protect requested condition membership and subtraction signs."""
import numpy as np
import pandas as pd
import unittest
import tempfile
from pathlib import Path

from posthoc_analysis.pre_decoder_waveforms import (
    TIME, CONDITIONS, compute_pre_condition_waveforms, plot_pre_decoder_waveforms,
)


def epoch_fixture():
    codes = np.repeat([8, 32, 44], [30, 15, 15])
    samples = np.arange(60) * 2000 + 1000
    signal = np.zeros((2, 768, 60))
    signal[1, 256:, :] = np.repeat([1., 2., 3.], [30, 15, 15])
    return dict(baseline_corrected_epochs=signal, eeg_labels=['PO7', 'PO8'],
                stimulus_events=np.column_stack([samples, np.zeros(60), codes]),
                kept_event_samples=samples, time=TIME, baseline_indices=(154, 256), sample_rate=512)


class PreDecoderWaveformTests(unittest.TestCase):
    def test_all_conditions_and_trial_counts(self):
        sums, counts = compute_pre_condition_waveforms(epoch_fixture())
        np.testing.assert_array_equal(counts, [30, 15, 15])
        np.testing.assert_array_equal(sums[:, :256], 0)
        np.testing.assert_allclose(sums[:, 300] / counts, [1, 2, 3])


    def test_reject_shifted_trial_alignment(self):
        result = epoch_fixture()
        result['kept_event_samples'] = result['kept_event_samples'] + 1
        with self.assertRaisesRegex(ValueError, 'alignment'):
            compute_pre_condition_waveforms(result)


    def test_reject_incomplete_condition(self):
        result = epoch_fixture()
        result['stimulus_events'][0, 2] = 32
        with self.assertRaisesRegex(ValueError, 'counts'):
            compute_pre_condition_waveforms(result)


    def test_plot_uses_requested_sign_for_both_classes(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tmp_path = Path(directory.name)
        frames = []
        for subject, scale in [('e21', 1), ('e23', 2)]:
            for condition, amplitude, n_trials in zip(CONDITIONS, [1, 2, 3], [240, 120, 120]):
                frames.append(pd.DataFrame(dict(subject_id=subject, group='bci' if subject == 'e21' else 'control',
                    condition=condition, sample_index=np.arange(768), time_sec=TIME,
                    po8_minus_po7_uv=np.where(TIME >= 0, amplitude*scale, 0), n_trials=n_trials)))
        result = plot_pre_decoder_waveforms({'subject_waveforms': pd.concat(frames)}, output_dir=tmp_path)
        summary = result['summary']
        at = summary[np.isclose(summary.time_sec, .25)].set_index(['side', 'condition'])
        assert at.loc[('right', 'distractor_right'), 'mean_uv'] == -3
        assert at.loc[('left', 'distractor_left'), 'mean_uv'] == 4.5
        assert at.loc[('right', 'no_distractor'), 'mean_uv'] == -1.5
        assert at.loc[('left', 'no_distractor'), 'mean_uv'] == 1.5
        assert at.loc[('right', 'no_distractor'), 'sem_uv'] == .5
        assert all(path.exists() for path in result['saved_paths'])


if __name__ == "__main__":
    unittest.main()
