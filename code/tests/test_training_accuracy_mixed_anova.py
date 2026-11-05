"""Accuracy filtering must precede counting and must retain incorrect trials."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from posthoc_analysis.behavioral import (
    run_training_accuracy_mixed_anova,
    run_training_rt_mixed_anova,
    summarize_training_accuracy_after_rt_filter,
)


class TrainingAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.trials = pd.DataFrame({
            "subject_id": ["e21"] * 60, "group": ["experimental"] * 60,
            "session_id": [1] * 60, "run_id": [1] * 60, "trial": np.arange(1, 61),
            "task": [0] * 30 + [1] * 30, "task_type": ["training"] * 60,
            "feedback": [1] * 60, "rt_ms": [500.] * 60, "rt_source": ["gdf_status"] * 60,
        })

    def summarize(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            return summarize_training_accuracy_after_rt_filter(data)

    def test_cutoff_denominator_timeouts_missing_and_slow_trials(self):
        self.trials.loc[:7, "rt_ms"] = [149., 149., 150., 150., 99999., np.nan, 2., np.nan]
        self.trials.loc[[1, 3], "feedback"] = 2
        self.trials.loc[[6, 7], "feedback"] = 3
        before = self.trials.copy(deep=True)
        result = self.summarize(self.trials)
        absent = result["subject_session_summary"].query("trial_type == 'no_distractor'").iloc[0]
        self.assertEqual(absent.n_removed_rt_lt_cutoff, 3)
        self.assertEqual(absent.n_timeouts_after_rt_filter, 1)
        self.assertEqual(absent.n_correct, 25)
        self.assertEqual(absent.n_incorrect, 1)
        self.assertEqual(absent.n_accuracy_trials, 26)
        self.assertEqual(absent.n_missing_rt_retained, 1)
        self.assertAlmostEqual(absent.accuracy, 25 / 26)
        retained = result["accuracy_trial_data"]
        self.assertTrue({3, 4, 5, 6}.issubset(set(retained.trial)))
        self.assertEqual(len(retained), 56)
        pd.testing.assert_frame_equal(self.trials, before)

    def test_sub50_fallback_and_practice_exclusion(self):
        self.trials.loc[0, ["rt_ms", "rt_source"]] = [2., "trigger_text_fallback"]
        practice = self.trials.copy()
        practice["task_type"] = "training_practice"
        result = self.summarize(pd.concat([self.trials, practice]))
        self.assertEqual(result["exclusion_report"].n_total_trials.sum(), 60)
        self.assertEqual(result["exclusion_report"].n_removed_rt_lt_cutoff.sum(), 1)
        self.assertEqual(len(result["accuracy_trial_data"]), 59)

    def test_zero_denominator_and_invalid_data_rejected(self):
        zero = self.trials.copy()
        zero.loc[:29, "feedback"] = 3
        duplicate = pd.concat([self.trials, self.trials.iloc[:1]])
        legacy = self.trials.copy()
        legacy["rt_source"] = "legacy_text_samples"
        wrong_group = self.trials.copy()
        wrong_group["group"] = "control"
        malformed = self.trials.iloc[:-1]
        for data, message in [(zero, "Zero accuracy denominator"), (duplicate, "Duplicate"),
                              (legacy, "provenance"), (wrong_group, "assignments"),
                              (malformed, "60 trials")]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.summarize(data)

    def test_same_anova_engine_preserves_affine_equivalence_and_accepts_zero_accuracy(self):
        rng = np.random.default_rng(123)
        rows = []
        for group in ["control", "experimental"]:
            for subject in range(5):
                for session in ["pre", "post"]:
                    for condition in ["no_distractor", "distractor"]:
                        rows.append({"subject_id": f"{group}_{subject}", "group": group,
                                     "session": session, "trial_type": condition,
                                     "mean_rt_ms": rng.uniform(400, 600)})
        rt = pd.DataFrame(rows)
        accuracy = rt.drop(columns="mean_rt_ms").assign(accuracy=rt.mean_rt_ms / 1000 + 0.2)
        with contextlib.redirect_stdout(io.StringIO()):
            expected = run_training_rt_mixed_anova(rt)["anova_table"]
            actual = run_training_accuracy_mixed_anova(accuracy)["anova_table"]
            accuracy.loc[0, "accuracy"] = 0
            zero_result = run_training_accuracy_mixed_anova(accuracy)
        np.testing.assert_allclose(expected[["F", "PR(>F)", "partial_eta_sq"]],
                                   actual[["F", "PR(>F)", "partial_eta_sq"]], atol=1e-10)
        self.assertEqual(zero_result["input_data"].accuracy.min(), 0)
        accuracy.loc[0, "accuracy"] = 1.01
        with self.assertRaisesRegex(ValueError, "Accuracy must lie"):
            run_training_accuracy_mixed_anova(accuracy)


if __name__ == "__main__":
    unittest.main()
