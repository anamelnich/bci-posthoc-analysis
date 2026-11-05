"""Guard the manuscript's RT-improvement sign without changing source means."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posthoc_analysis.config import EXPECTED_SUBJECTS, get_subject_group
from posthoc_analysis.theta_rt_association import analyze_theta_rt_changes, plot_theta_rt_changes


class ThetaRTImprovementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ['theta', 'rt', 'cache']:
            (self.root / name).mkdir()
        rng = np.random.default_rng(391)
        theta_change = rng.normal(size=32)
        improvement = 15 * theta_change + rng.normal(0, 12, 32)
        rows, powers = [], []
        for i, subject in enumerate(EXPECTED_SUBJECTS):
            group = get_subject_group(subject)
            for session in (1, 5):
                runs = 7 if subject == 'e30' and session == 1 else 8 if session == 1 else 4
                count = 60 * runs
                excluded = 17 if 2 * i + (session == 5) < 16 else 16
                mean = 600 + i - (improvement[i] if session == 5 else 0)
                for trial in range(count):
                    rows.append(dict(subject_id=subject, group=group, session_id=session,
                        run_id=trial // 60 + 1, trial=trial % 60 + 1, task=trial % 2,
                        feedback=1, gdf_rt_ms=mean, rt_ms=mean, retained=trial >= excluded,
                        missing_status_response=False))
                powers.append(dict(subject_id=subject, group=group, session_id=session,
                    n_trials=count-excluded, normalized_theta_db=1 + (theta_change[i] if session == 5 else 0)))
        mask = pd.DataFrame(rows)
        missing = mask.index[~mask.retained][:47]
        mask.loc[missing, ['gdf_rt_ms', 'rt_ms']] = np.nan
        mask.loc[missing, 'missing_status_response'] = True
        self.assertEqual((len(mask), int(mask.retained.sum())), (22980, 21940))
        mask.to_csv(self.root / 'cache/trial_mask.csv', index=False)
        mask.loc[mask.retained].to_csv(self.root / 'rt/trial_data_clean.csv', index=False)
        power_path = self.root / 'theta/participant_window_power.csv'
        pd.DataFrame(powers).to_csv(power_path, index=False)
        (self.root / 'theta/final_manifest.json').write_text(json.dumps({'files_sha256': {
            str(power_path): hashlib.sha256(power_path.read_bytes()).hexdigest()}}))
        (self.root / 'theta/validation.json').write_text(json.dumps({'roi': ['FZ', 'FCZ', 'CZ'], 'n_retained_trials': 21940}))
        (self.root / 'cache/mask_validation.json').write_text(json.dumps({'mask_sha256':
            hashlib.sha256((self.root / 'cache/trial_mask.csv').read_bytes()).hexdigest()}))

    def analyze(self, paper_only):
        with contextlib.redirect_stdout(io.StringIO()):
            return analyze_theta_rt_changes(self.root / 'theta', self.root / 'rt',
                self.root / 'cache', self.root / 'results', paper_only=paper_only)

    def test_improvement_correlations_and_compatibility_change(self):
        changes, result, validation = self.analyze(paper_only=False)
        np.testing.assert_allclose(changes.rt_improvement_ms, changes.rt_pre_ms-changes.rt_post_ms)
        np.testing.assert_allclose(changes.rt_change_ms, changes.rt_post_ms-changes.rt_pre_ms)
        self.assertTrue(result.rt_difference.eq('pre_minus_post').all())
        self.assertTrue(result.rt_measure.eq('rt_improvement_ms').all())
        self.assertEqual(validation['rt_measure'], 'rt_improvement_ms')
        for row in result.itertuples():
            sub = changes if row.scope in ('pooled_unadjusted', 'group_adjusted') else changes[changes.group.eq(row.scope)]
            x, y = sub.theta_change_db, sub.rt_improvement_ms
            if row.scope == 'group_adjusted':
                x = x-sub.groupby('group').theta_change_db.transform('mean')
                y = y-sub.groupby('group').rt_improvement_ms.transform('mean')
            expected = stats.pearsonr(x, y).statistic
            self.assertAlmostEqual(row.pearson_r, expected)
            self.assertAlmostEqual(row.pearson_r, -stats.pearsonr(x, -y).statistic)
            self.assertLess(row.ci95_low, row.ci95_high)
            k = int(row.scope == 'group_adjusted')
            old_t = -expected*np.sqrt((len(sub)-k-2)/(1-expected**2))
            self.assertAlmostEqual(row.p_value, 2*stats.t.sf(abs(old_t), len(sub)-k-2))
            design = sm.add_constant(np.c_[sub.theta_change_db, sub.group.eq('bci').astype(float)]
                                     if k else sub.theta_change_db.to_numpy())
            old_model = sm.OLS(sub.rt_change_ms.to_numpy(), design).fit()
            self.assertAlmostEqual(row.slope_ms_per_db, -old_model.params[1])
            np.testing.assert_allclose([row.slope_ci95_low, row.slope_ci95_high],
                                       -old_model.conf_int()[1][::-1])
            old_loo = []
            for index in sub.index:
                left = sub.drop(index)
                lx, ly = left.theta_change_db, left.rt_change_ms
                if k:
                    lx = lx-left.groupby('group').theta_change_db.transform('mean')
                    ly = ly-left.groupby('group').rt_change_ms.transform('mean')
                old_loo.append(stats.pearsonr(lx, ly).statistic)
            np.testing.assert_allclose([row.leave_one_out_r_min, row.leave_one_out_r_max],
                                       [-max(old_loo), -min(old_loo)])
            if row.scope != 'group_adjusted':
                self.assertAlmostEqual(row.p_value, stats.pearsonr(x, -y).pvalue)
                self.assertAlmostEqual(row.spearman_rho, stats.spearmanr(x, y).statistic)
                old_ci = np.tanh(np.arctanh(-expected) + np.array([-1, 1]) * stats.norm.ppf(.975)/np.sqrt(len(sub)-3))
                np.testing.assert_allclose([row.ci95_low, row.ci95_high], [-old_ci[1], -old_ci[0]])

    def test_plot_rejects_legacy_results_and_mislabeled_improvement(self):
        changes, result, _ = self.analyze(paper_only=True)
        with self.assertRaises(ValueError):
            plot_theta_rt_changes(changes, result.assign(rt_difference='post_minus_pre'), self.root / 'plot')
        wrong = changes.copy()
        wrong['rt_improvement_ms'] = wrong.rt_change_ms
        with self.assertRaises(ValueError):
            plot_theta_rt_changes(wrong, result, self.root / 'plot')


if __name__ == '__main__':
    unittest.main()
