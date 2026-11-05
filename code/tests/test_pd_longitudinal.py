"""Scientific checks for seven-point Pd integration and coverage."""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from posthoc_analysis import eeg
from posthoc_analysis.pd_longitudinal import TIME, compute_seven_point_pd_auc


def waveform_table():
    rows = []
    # Two opposite single-trial difference waves cancel before rectification.
    first = np.where(TIME >= 0, 2., 0.)
    second = -first
    mean_wave = np.stack([first, second]).mean(axis=0)
    for point in range(7):
        rows.append(pd.DataFrame(dict(subject_id='e21', group='bci', point_index=point,
            sample_index=np.arange(768), time_sec=TIME, pd_amplitude_uv=mean_wave,
            session_id=1 if point == 0 else 5 if point == 6 else point,
            n_retained_distractor_trials=60, n_runs=2, trial_policy='all distractors')))
    return pd.concat(rows, ignore_index=True)


class LongitudinalPdTests(unittest.TestCase):
    def test_rectifies_mean_waveform_not_trials(self):
        data = waveform_table()
        with contextlib.redirect_stdout(io.StringIO()):
            result = compute_seven_point_pd_auc({'subject_point_pd': data})
        self.assertTrue(result['participant_auc'].positive_auc_uv_sec.eq(0).all())

    def test_matches_existing_prepost_integral_and_units(self):
        data = waveform_table()
        data['pd_amplitude_uv'] = np.where(data.time_sec >= 0, 2., 0.)
        endpoints = data[data.point_index.isin([0, 6])].copy()
        with contextlib.redirect_stdout(io.StringIO()):
            old = eeg.compute_pd_positive_auc(endpoints)
            new = compute_seven_point_pd_auc({'subject_point_pd': data})['participant_auc']
        values = new[new.point_index.isin([0, 6])].positive_auc_uv_sec
        np.testing.assert_array_equal(values.to_numpy(), old.positive_auc_uv_sec.to_numpy())
        self.assertTrue(new.n_samples.eq(154).all())
        np.testing.assert_allclose(new.positive_auc_uv_sec, 2*(.5-103/512))

    def test_rejects_missing_point_and_corrupt_time_grid(self):
        data = waveform_table()
        with self.assertRaisesRegex(ValueError, 'all seven'):
            compute_seven_point_pd_auc({'subject_point_pd': data[data.point_index != 3]})
        data.loc[0, 'time_sec'] += .001
        with self.assertRaisesRegex(ValueError, 'time grid'):
            compute_seven_point_pd_auc({'subject_point_pd': data})


if __name__ == '__main__':
    unittest.main()
