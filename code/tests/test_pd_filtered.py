"""Scientific regression checks for Pd exclusions and trial weighting."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from posthoc_analysis.pd_filtered import prepare_pd_trial_exclusions, rebuild_behavior_filtered_pd


def trial_table(session=1, run=1):
    codes = np.repeat([8, 32, 44], [30, 15, 15])
    samples = np.arange(60) * 2000 + 1000
    return pd.DataFrame({
        "subject_id": "e21", "group": "experimental", "session_id": session,
        "run_id": run, "trial": np.arange(1, 61), "task_type": "training",
        "task": (codes != 8).astype(int), "feedback": 1, "target_position": 1,
        "distractor_position": np.repeat([0, 2, 4], [30, 15, 15]), "dot_side": 0,
        "intertrial_interval_ms": 650, "bci_output": 99, "stimulus_trigger": codes,
        "stimulus_sample": samples, "condition": np.repeat(
            ["no_distractor", "distractor_right", "distractor_left"], [30, 15, 15]),
        "rt_ms": 500., "gdf_rt_ms": 500., "rt_samples": 256.,
        "response_sample": samples + 256, "response_count": 1, "alignment_error": False,
        "text_rt_ms": 500., "sfreq_hz": 512., "rt_source": "gdf_status",
    })


class FilteredPdTests(unittest.TestCase):
    def test_sequential_filter_and_sample_sd_pooled_across_conditions(self):
        data = trial_table()
        # Early exclusions must not inflate the SD and hide the slow correct trial.
        data.loc[[0, 1], "feedback"] = [2, 3]
        rt = np.full(60, 500.)
        rt[:6] = [100000, 100000, 149, 150, 5000, 500]
        data["rt_ms"] = data["gdf_rt_ms"] = data["text_rt_ms"] = rt
        data["rt_samples"] = rt * .512
        data["response_sample"] = data.stimulus_sample + data.rt_samples
        data.loc[5, ["rt_ms", "gdf_rt_ms", "rt_samples", "response_sample"]] = np.nan
        data.loc[5, "response_count"] = 0
        with contextlib.redirect_stdout(io.StringIO()):
            result = prepare_pd_trial_exclusions(data.sample(frac=1, random_state=7))
        audit = result["trials"].set_index("trial")
        self.assertEqual(audit.exclusion_reason.iloc[:6].tolist(),
                         ["incorrect", "timeout", "rt_below_150", "retained", "rt_outlier_3sd", "missing_rt"])
        expected = np.r_[150., 5000., np.full(54, 500.)]
        self.assertAlmostEqual(result["rt_bounds"].sd_before_ms.iloc[0], expected.std(ddof=1))
        self.assertEqual(result["rt_bounds"].shape[0], 1)
        self.assertEqual(int(audit.retained_for_pd.sum()), 30)

    def test_duplicate_trials_rejected(self):
        data = trial_table()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_pd_trial_exclusions(pd.concat([data, data.iloc[[0]]]))

    def test_pools_trials_not_runs_or_sides_and_checks_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tables, infos, paths, epoch_results = [], [], {}, {}
            time = np.arange(768) / 512 - .5
            shape = (time >= .2).astype(float)
            for session in [1, 5]:
                paths[session] = []
                for run in [1, 2]:
                    table = trial_table(session, run)
                    if run == 2:
                        table.loc[45:, "feedback"] = 2
                    tables.append(table)
                    gdf = root / f"session{session}_run{run}_training.gdf"
                    for suffix in [".gdf", ".triggers.txt", ".analysis.txt"]:
                        gdf.with_suffix(suffix).touch()
                    paths[session].append(gdf)
                    infos.append({"subject_id": "e21", "session_number": session,
                                  "run_number": run, "trigger_file": gdf.with_suffix(".triggers.txt"),
                                  "synthetic": table})
                    epochs = np.zeros((2, 768, 60))
                    value = 1 if run == 1 else 5
                    epochs[0, :, 30:45] = shape[:, None] * value
                    epochs[1, :, 45:] = shape[:, None] * value
                    epoch_results[gdf] = {"baseline_corrected_epochs": epochs,
                        "eeg_labels": ["PO7", "PO8"], "time": time,
                        "stimulus_events": np.column_stack([table.stimulus_sample, np.zeros(60), table.stimulus_trigger]),
                        "kept_event_samples": table.stimulus_sample.to_numpy(),
                        "baseline_indices": (154, 256)}
            data = pd.concat(tables, ignore_index=True)
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch("posthoc_analysis.pd_filtered.collect_all_training_runs", return_value=(infos, {}, ["e21"])), \
                    patch("posthoc_analysis.pd_filtered.eeg._get_training_run_gdf_files_for_session",
                          side_effect=lambda subject, session, *args, **kwargs: paths[session]), \
                    patch("posthoc_analysis.pd_filtered.load_and_merge_training_run", side_effect=lambda info: info["synthetic"]), \
                    patch("posthoc_analysis.pd_filtered.eeg.load_filter_epoch_baseline_correct_training_run",
                          side_effect=lambda path: epoch_results[path]) as loader:
                result = rebuild_behavior_filtered_pd(data, root, root / "cache")
                wave = result["subject_session_pd"].query("session_id == 1")
                np.testing.assert_allclose(wave.pd_amplitude_uv, shape * (30 * 1 + 15 * 5) / 45)
                self.assertEqual(loader.call_count, 4)
                cached = rebuild_behavior_filtered_pd(data, root, root / "cache")
                self.assertEqual(loader.call_count, 4)
                np.testing.assert_allclose(cached["subject_session_pd"].pd_amplitude_uv,
                                           result["subject_session_pd"].pd_amplitude_uv)
                with (root / "cache" / "subject_session_pd.csv").open("a") as file:
                    file.write("tampered")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    rebuild_behavior_filtered_pd(data, root, root / "cache")


if __name__ == "__main__":
    unittest.main()
