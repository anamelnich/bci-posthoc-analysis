"""Reject corrupt frozen transforms before scientific inference or cache reuse."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from posthoc_analysis.decoder_training import (
    CONVENTIONAL_PPO_PAIRS, load_session5_frozen_feature_reference,
)


class FrozenReferenceIntegrityTests(unittest.TestCase):
    def test_corrupt_numeric_transforms_and_coordinates_are_rejected(self):
        time = np.arange(768) / 512 - .5
        window = np.flatnonzero(time >= .2)
        valid = dict(
            artifact_version=np.array([1]), subject_id=np.array(['e21']),
            decoder_side=np.array(['right']), epoch_time_s=time,
            difference_channel_labels=np.array([f'{r}-{l}' for r, l in CONVENTIONAL_PPO_PAIRS]),
            xdawn_filters_components_by_channels=np.ones((2, 7)),
            feature_window_indices=window, resampled_indices=window[::8],
            normalization_means=np.zeros((104, 1)), normalization_stds=np.ones((104, 1)),
            selected_indices_zero_based=np.arange(30),
        )
        table = pd.DataFrame(dict(subject_id='e21', decoder_side='right',
                                  rank=np.arange(1,31), feature_index_zero_based=np.arange(30)))
        corruptions = [
            ('normalization_stds', np.full((104,1), np.nan), 'nonfinite'),
            ('normalization_means', np.full((104,1), np.inf), 'nonfinite'),
            ('xdawn_filters_components_by_channels', np.full((2,7), np.nan), 'nonfinite'),
            ('selected_indices_zero_based', np.arange(30)+.5, 'integer'),
            ('selected_indices_zero_based', np.zeros(30,dtype=int), 'unique'),
            ('selected_indices_zero_based', np.arange(100,130), 'in-range'),
            ('resampled_indices', window[::8]+1, 'stride'),
            ('epoch_time_s', time+.001, '512 Hz'),
            ('difference_channel_labels', valid['difference_channel_labels'][::-1], 'order'),
            ('artifact_version', np.array([99]), 'version'),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'transforms').mkdir();(root/'session5_reference_top30.parquet').touch()
            path=root/'transforms/e21_right_frozen_reference.npz'
            with patch('posthoc_analysis.decoder_training.pd.read_parquet',return_value=table):
                np.savez(path,**valid)
                loaded=load_session5_frozen_feature_reference('e21','right',root)
                np.testing.assert_array_equal(loaded['selected_indices_zero_based'],np.arange(30))
                for key,value,message in corruptions:
                    with self.subTest(field=key,error=message):
                        np.savez(path,**{**valid,key:value})
                        with self.assertRaisesRegex(ValueError,message):
                            load_session5_frozen_feature_reference('e21','right',root)
                np.savez(path,**valid)
                for key in ['rank','feature_index_zero_based']:
                    malformed=table.copy();malformed[key]=malformed[key].astype(float);malformed.loc[0,key]+=.5
                    with patch('posthoc_analysis.decoder_training.pd.read_parquet',return_value=malformed):
                        with self.assertRaisesRegex(ValueError,'ranks|coordinates'):
                            load_session5_frozen_feature_reference('e21','right',root)
