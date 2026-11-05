"""Validate mixed-ANOVA error strata against independent contrast tests."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from posthoc_analysis.behavioral import run_training_rt_mixed_anova


class TrainingRTMixedAnovaTests(unittest.TestCase):
    def setUp(self):
        # Unequal component variances expose accidental pooling of error strata.
        rng = np.random.default_rng(420)
        rows = []
        for group in ["control", "experimental"]:
            for subject in range(8):
                baseline = 500 + rng.normal(0, 70)
                session_change = rng.normal(-35 if group == "experimental" else 0, 20)
                condition_cost = rng.normal(25, 10)
                cost_change = rng.normal(12 if group == "experimental" else 0, 4)
                for session in ["pre", "post"]:
                    for condition in ["no_distractor", "distractor"]:
                        post, present = session == "post", condition == "distractor"
                        rows.append({"subject_id": f"{group}_{subject}", "group": group,
                                     "session": session, "trial_type": condition,
                                     "mean_rt_ms": baseline + post * session_change
                                     + present * condition_cost + post * present * cost_change})
        self.data = pd.DataFrame(rows)

    def run_anova(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_training_rt_mixed_anova(data)

    def test_seven_effects_against_independent_contrast_regressions(self):
        before = self.data.copy(deep=True)
        result = self.run_anova(self.data)
        table = result["anova_table"]
        pd.testing.assert_frame_equal(self.data, before)
        wide = self.data.pivot(index=["subject_id", "group"],
                               columns=["session", "trial_type"], values="mean_rt_ms")
        pre = wide["pre"].mean(axis=1)
        post = wide["post"].mean(axis=1)
        pre_cost = wide["pre"]["distractor"] - wide["pre"]["no_distractor"]
        post_cost = wide["post"]["distractor"] - wide["post"]["no_distractor"]
        contrasts = {"Session": post - pre, "Distractor Condition": (pre_cost + post_cost) / 2,
                     "Session x Distractor Condition": post_cost - pre_cost}
        for label, values in contrasts.items():
            reference = values.rename("value").reset_index()
            fit = ols("value ~ C(group, Sum)", reference).fit()
            # In a balanced design, intercept tests the marginal within effect.
            for name, coefficient in [(label, 0), (f"Group x {label}", 1)]:
                self.assertAlmostEqual(table.loc[name, "F"], fit.tvalues.iloc[coefficient] ** 2, places=9)
                self.assertAlmostEqual(table.loc[name, "PR(>F)"], fit.pvalues.iloc[coefficient], places=10)
                f = table.loc[name, "F"]
                self.assertAlmostEqual(table.loc[name, "partial_eta_sq"], f / (f + 14))
        subject_mean = wide.mean(axis=1)
        t, p = stats.ttest_ind(subject_mean.xs("control", level="group"),
                               subject_mean.xs("experimental", level="group"), equal_var=True)
        self.assertAlmostEqual(table.loc["Group", "F"], t ** 2)
        self.assertAlmostEqual(table.loc["Group", "PR(>F)"], p)
        self.assertEqual(len(table), 7)
        self.assertTrue(table.error_df.eq(14).all())
        self.assertEqual(result["error_terms"].error_term.nunique(), 4)
        self.assertAlmostEqual(result["diagnostics"]["total_sum_sq"],
                               result["diagnostics"]["partitioned_sum_sq"], places=8)

    def test_row_order_and_common_offset_do_not_change_tests(self):
        expected = self.run_anova(self.data)["anova_table"]
        changed = self.data.sample(frac=1, random_state=4).copy()
        changed["mean_rt_ms"] += 150
        actual = self.run_anova(changed)["anova_table"]
        np.testing.assert_allclose(actual[["F", "PR(>F)", "partial_eta_sq"]],
                                   expected[["F", "PR(>F)", "partial_eta_sq"]], atol=1e-10)

    def test_incomplete_duplicate_nonfinite_and_unbalanced_inputs_rejected(self):
        cases = [(self.data.iloc[1:], "Complete four-cell"),
                 (pd.concat([self.data, self.data.iloc[:1]]), "Duplicate"),
                 (self.data[self.data.subject_id != "control_0"], "Balanced")]
        nonfinite = self.data.copy()
        nonfinite.loc[0, "mean_rt_ms"] = np.inf
        cases.append((nonfinite, "finite"))
        missing = self.data.copy()
        missing.loc[0, "mean_rt_ms"] = np.nan
        cases.append((missing, "nonmissing"))
        membership = self.data.copy()
        membership.loc[0, "group"] = "experimental"
        cases.append((membership, "exactly one group"))
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.run_anova(data)


if __name__ == "__main__":
    unittest.main()
