"""Guard the explicit Stroop session-pooled RT mask used for EEG topographies."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posthoc_analysis.stroop_theta import select_stroop_theta_trials


class StroopThetaMaskTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(dict(subject_id=['e21']*120, group=['bci']*120,
            session_id=[1]*120, run_id=[1]*60+[2]*60,
            trial_number=list(range(1,61))*2, trial_type=['congruent','incongruent']*60,
            response=[1]*120, reaction_time_ms=[500.]*60+[1000.]*60))
        self.data.loc[0,'reaction_time_ms']=149.
        self.data.loc[60,'reaction_time_ms']=5000.
        self.data.loc[61,['response','reaction_time_ms']]=[2,999999.]
        self.data.loc[62,['response','reaction_time_ms']]=[3,np.nan]

    def test_pooled_runs_no_minimum_and_disjoint_exclusions(self):
        original=self.data.copy(deep=True)
        ledger,thresholds=select_stroop_theta_trials(self.data)
        self.assertEqual(len(thresholds),1)
        self.assertTrue(ledger.query('run_id == 1 and trial_number == 1').retained.item())
        self.assertEqual(ledger.exclusion_reason.value_counts().to_dict(),
                         {'retained':117,'outside_session_3sd':1,'incorrect':1,'timeout':1})
        correct=self.data.loc[self.data.response.eq(1),'reaction_time_ms']
        self.assertAlmostEqual(thresholds.session_sd_rt_ms.item(),correct.std(ddof=1))
        pd.testing.assert_frame_equal(self.data,original)

    def test_session_separation_and_zero_sd(self):
        post=self.data.assign(session_id=5,response=1,reaction_time_ms=20000.)
        ledger,thresholds=select_stroop_theta_trials(pd.concat([self.data,post],ignore_index=True))
        self.assertEqual(len(thresholds),2)
        self.assertEqual(ledger.query('session_id == 1').retained.sum(),117)
        self.assertEqual(ledger.query('session_id == 5').retained.sum(),120)
        shuffled,_=select_stroop_theta_trials(pd.concat([self.data,post],ignore_index=True).sample(frac=1,random_state=3))
        pd.testing.assert_frame_equal(ledger,shuffled)

    def test_malformed_behavior_is_rejected(self):
        for column,value in [('response',4),('trial_type','neutral'),('reaction_time_ms',np.nan)]:
            bad=self.data.copy();bad.loc[0,column]=value
            with self.assertRaises(ValueError):select_stroop_theta_trials(bad)
        with self.assertRaises(ValueError):select_stroop_theta_trials(pd.concat([self.data,self.data.iloc[:1]]))


if __name__ == '__main__':
    unittest.main()
