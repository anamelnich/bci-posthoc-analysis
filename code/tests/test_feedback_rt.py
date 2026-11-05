"""Next-trial timing, pairing, exclusion, aggregation and inference regressions."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from posthoc_analysis.feedback_rt import reconstruct_decoding_events, load_decoding_gdf_rt, build_feedback_rt_pairs, clean_feedback_rt_pairs
from posthoc_analysis.feedback_rt_stats import summarize_feedback_rt, run_feedback_rt_statistics, feedback_sphericity


class FeedbackRTTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.events = np.array([[s, 0, c] for i in range(60)
                                for s, c in [(100+i*1000, 4), (200+i*1000, [8,32,44][i%3]), (456+i*1000,64)]])

    def reconstruct(self, events=None):
        return reconstruct_decoding_events(self.events if events is None else events, 512, 61000)[0]

    def trials(self, group='experimental'):
        return pd.DataFrame(dict(subject_id=['s1']*8, group=[group]*8, session_id=[1]*4+[2]*4,
            run_id=[1]*8, trial_id=[1,2,3,4]*2, decoding_bci_output=[1,0,3,1]*2,
            decoding_feedback=[2,1,3,1]*2, decoding_task=[0,1,0,1]*2, rt_ms=[500.]*8))

    def test_sample_clock_and_last_trial(self):
        result = self.reconstruct()
        self.assertTrue(result.rt_ms.eq(500).all())
        self.assertEqual(result.response_stop_sample.iloc[-1], 61000)

    def test_missing_response_and_response_after_next_fixation(self):
        events = np.delete(self.events, 2, axis=0)
        # Response after trial 2 fixation but before its stimulus cannot be trial 1 RT.
        events = np.vstack([events, [1150,0,64]])
        events = events[np.argsort(events[:,0])]
        result = self.reconstruct(events)
        self.assertEqual(result.rt_reason.iloc[0], 'missing_response')
        self.assertTrue(np.isnan(result.rt_ms.iloc[0]))
        self.assertEqual(result.rt_ms.iloc[1],500)

    def test_ambiguous_responses_and_fixations(self):
        events = np.vstack([self.events, [500,0,64], [2500,0,4]])
        result = self.reconstruct(events[np.argsort(events[:,0])])
        self.assertEqual(result.rt_reason.iloc[0], 'multiple_responses')
        self.assertTrue(np.isnan(result.rt_ms.iloc[0]))
        self.assertEqual(result.rt_reason.iloc[2], 'ambiguous_fixation_boundary')

    def test_missing_stimulus_rejects_run(self):
        with self.assertRaisesRegex(ValueError,'60 stimulus'):
            self.reconstruct(np.delete(self.events,1,axis=0))

    def test_status_plateaus_generate_single_events(self):
        import mne
        values = np.zeros((1, 61000))
        for sample, _, code in self.events:
            values[0, sample:sample+20] = code
        raw = mne.io.RawArray(values, mne.create_info(['Status'], 512, ['stim']), verbose='ERROR')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'synthetic_decoding.gdf'
            path.write_bytes(b'synthetic reader fixture')
            with patch('mne.io.read_raw_gdf', return_value=raw), self.assertWarnsRegex(UserWarning, 'nonstandard channel'):
                trials, events, qc = load_decoding_gdf_rt(path)
        self.assertEqual(len(events), 180)
        self.assertEqual(qc['n_stimuli'], 60)
        self.assertTrue(trials.rt_ms.eq(500).all())

    def test_pairs_keep_incorrect_previous_trial_and_never_cross_session(self):
        pairs, losses = build_feedback_rt_pairs(self.trials())
        self.assertEqual(len(pairs),6)
        self.assertEqual(len(losses),0)
        self.assertTrue(pairs.current_trial_id.sub(pairs.previous_trial_id).eq(1).all())
        self.assertEqual(pairs.previous_bci_feedback.iloc[0],1)
        self.assertEqual(pairs.current_trial_type.iloc[0],1)
        self.assertEqual(pairs.current_behavioral_outcome.iloc[0],1)

    def test_invalid_feedback_is_technical_control_ignores_it(self):
        data = self.trials()
        data.loc[0,'decoding_bci_output']=99
        pairs, losses=build_feedback_rt_pairs(data)
        self.assertEqual(len(pairs),5)
        self.assertEqual(losses.technical_reason.iloc[0],'missing_invalid_previous_feedback')
        data['group']='control'
        pairs, losses=build_feedback_rt_pairs(data)
        self.assertEqual(len(pairs),6)
        self.assertTrue(pairs.previous_bci_feedback.isna().all())

    def test_nonconsecutive_and_missing_behavior(self):
        data=self.trials().drop(index=1)
        data.loc[7,'decoding_feedback']=np.nan
        pairs, losses=build_feedback_rt_pairs(data)
        self.assertEqual(set(losses.technical_reason),{'nonconsecutive_pair','missing_invalid_behavior'})

    def test_exclusion_order_exactly_half_allowed_and_zero_candidates(self):
        pairs,_=build_feedback_rt_pairs(self.trials())
        pairs['current_behavioral_outcome']=[2,3,1,1,1,1]
        pairs['current_rt_ms']=[np.nan,100,np.nan,500,500,500]
        roster=pd.DataFrame({'subject_id':['s1','s0'],'group':['experimental']*2})
        result=clean_feedback_rt_pairs(pairs,roster)
        report=result['subject_exclusions'].set_index('subject_id')
        self.assertEqual(report.loc['s1','exclusion_percentage'],50)
        self.assertFalse(report.loc['s1','subject_excluded'])
        self.assertTrue(report.loc['s0','subject_excluded'])
        self.assertEqual(report.loc['s1','missing_invalid_rt'],1)
        self.assertEqual(report.loc['s1','below_150_ms'],0)
        self.assertEqual(set(result['outlier_strata'].trim_status),{'skipped_zero_sd','skipped_n_lt_2'})
        pairs.loc[pairs.index[-1],'current_rt_ms']=149
        result=clean_feedback_rt_pairs(pairs,roster)
        self.assertTrue(result['subject_exclusions'].set_index('subject_id').loc['s1','subject_excluded'])

    def test_single_pass_outliers_pool_runs_and_feedback(self):
        pairs,_=build_feedback_rt_pairs(self.trials())
        data=pd.concat([pairs.iloc[[0]]]*21,ignore_index=True)
        data['current_rt_ms']=[500.]*20+[1000.]
        data['run_id']=[1]*10+[2]*11
        data['previous_bci_feedback']=[0]*10+[1]*11
        result=clean_feedback_rt_pairs(data)
        self.assertEqual(result['subject_exclusions'].outlier_3sd.iloc[0],1)
        self.assertEqual(len(result['outlier_strata']),1)
        self.assertAlmostEqual(result['outlier_strata']['mean'].iloc[0],11000/21)

    def test_subject_weighting_and_missing_cells(self):
        data=pd.DataFrame(dict(subject_id=['s1']*10+['s2'],group=['experimental']*11,
            current_trial_type=[1]*9+[0,0],previous_bci_feedback=[1]*11,current_rt_ms=[500.]*9+[1000.,1000.]))
        roster=pd.DataFrame(dict(subject_id=['s1','s2'],group=['experimental']*2,subject_excluded=[False]*2))
        result=summarize_feedback_rt(data,roster)
        overall=result['bci_group_summary'].query("condition=='overall' and previous_feedback=='positive'").iloc[0]
        self.assertEqual(overall.mean_rt_ms,775) # mean(550,1000), not pooled trials or equal condition means
        self.assertEqual(overall.n_trials,11)
        missing=result['bci_subject_summary'].query("previous_feedback=='negative'")
        self.assertTrue(missing.mean_rt_ms.isna().all())

    def test_sphericity_matches_published_reference(self):
        # Reference: https://pingouin-stats.org/generated/pingouin.sphericity.html
        values = np.array([[2.2,1.1,8.2],[3.1,2.5,4.5],[4.3,4.1,3.4],[4.1,5.2,6.2],[7.2,6.4,7.2]])
        result = feedback_sphericity(values)
        self.assertEqual(round(result['mauchly_W'],3),.210)
        self.assertEqual(round(result['mauchly_chi2'],3),4.677)
        self.assertEqual(round(result['mauchly_p'],3),.096)

    def test_sphericity_scale_translation_invariance(self):
        rng=np.random.default_rng(50)
        values=rng.normal(size=(24,3))*[1,1,9]
        a=feedback_sphericity(values)
        b=feedback_sphericity(values*10 + np.arange(24)[:,None]*300)
        self.assertAlmostEqual(a['mauchly_W'],b['mauchly_W'])
        self.assertAlmostEqual(a['epsilon_gg'],b['epsilon_gg'])
        self.assertEqual(a['sphericity_status'],'violated')
        self.assertAlmostEqual(a['epsilon_gg'],1/(2-a['mauchly_W']))

    def test_models_complete_case_but_posthoc_pairwise_complete(self):
        rng=np.random.default_rng(100)
        rows=[]
        for i in range(12):
            for condition in ['overall','distractor_present','distractor_absent']:
                for j,feedback in enumerate(['positive','negative','ambivalent']):
                    value=500+i*3+j*12+rng.normal(0,4)
                    if i==0 and feedback=='ambivalent': value=np.nan
                    rows.append(dict(subject_id=f's{i}',group='experimental',condition=condition,previous_feedback=feedback,mean_rt_ms=value))
        result=run_feedback_rt_statistics(pd.DataFrame(rows))
        self.assertTrue(result['anova_results'].n_subjects.eq(11).all())
        post=result['posthoc_comparisons']
        self.assertTrue(post.loc[post['second'].eq('positive'),'n_pairs'].eq(12).all())
        self.assertTrue(post.p_holm.ge(post.p_raw-1e-15).all())
        row=post.iloc[0]
        self.assertGreater(row.mean_difference_ms,0)
        self.assertAlmostEqual(row.cohens_dz,row.t/np.sqrt(row.n_pairs))
        overall=pd.DataFrame(rows).query("condition=='overall'").pivot(index='subject_id',columns='previous_feedback',values='mean_rt_ms').dropna().to_numpy()
        centered=overall-overall.mean(axis=1,keepdims=True)
        ss_effect=len(overall)*(centered.mean(axis=0)**2).sum()
        ss_error=((centered-centered.mean(axis=0))**2).sum()
        expected_f=(ss_effect/2)/(ss_error/(2*(len(overall)-1)))
        self.assertAlmostEqual(result['anova_results'].query("model=='overall'").F.iloc[0],expected_f)

    def test_model_specific_missingness_and_gg_correction(self):
        rng = np.random.default_rng(50)
        values = 500 + rng.normal(size=(24,3))*[1,1,9]
        rows = [dict(subject_id=f's{i}', group='experimental', condition=c,
                     previous_feedback=f, mean_rt_ms=values[i,j] + (20 if c=='distractor_present' else 0))
                for i in range(24) for c in ['overall','distractor_present','distractor_absent']
                for j,f in enumerate(['positive','negative','ambivalent'])]
        data = pd.DataFrame(rows)
        data.loc[data.subject_id.eq('s0') & data.condition.eq('distractor_present') & data.previous_feedback.eq('ambivalent'), 'mean_rt_ms'] = np.nan
        result = run_feedback_rt_statistics(data)['anova_results']
        self.assertEqual(result.query("model=='overall'").n_subjects.iloc[0],24)
        self.assertEqual(result.query("model=='feedback_x_trial_type'").n_subjects.iloc[0],23)
        self.assertEqual(result.query("model=='distractor_absent'").n_subjects.iloc[0],24)
        row = result.query("model=='overall'").iloc[0]
        self.assertEqual(row.correction,'Greenhouse-Geisser')
        self.assertAlmostEqual(row.p_reported,row.p_gg)
        self.assertAlmostEqual(row.df1_reported,row.df1*row.epsilon_gg)


if __name__ == '__main__':
    unittest.main()
