"""Retained Pd robustness, response/RT controls, and target geometry.

All analyses use real, individually aligned trials. The former alpha/theta candidate family is not part of this module.
"""


from pathlib import Path


import hashlib


import json


import numpy as np


import pandas as pd


from scipy import stats


from statsmodels.stats.multitest import multipletests


from .config import EXPECTED_SUBJECTS


from .pd_robustness_support import validate_cohort_table, compare_group_changes


KEYS = ['subject_id', 'group', 'session_id']





def load_neural_checkpoints(root, settings_path):
    """Validate every source, trial, cached array, baseline and response alignment."""
    root = Path(root)
    data = pd.read_csv(root / 'neural_trial_features.csv')
    logs = pd.read_csv(root / 'neural_run_log.csv')
    validate_cohort_table(data, ['subject_id', 'session_id', 'run_id', 'trial_index'])
    validate_cohort_table(logs, ['subject_id', 'session_id', 'run_id'])
    if len(data) != 22980 or len(logs) != 383 or not logs.status.eq('pass').all():
        raise ValueError('Require all 383 validated runs / 22,980 trials, including documented e30 exception.')
    signature = hashlib.sha256(Path(settings_path).read_bytes() +
        Path(__file__).with_name('neural_mechanisms.py').read_bytes()).hexdigest()
    arrays, frames, maximum_alignment_error = {}, [], 0.
    for entry in logs.itertuples(index=False):
        source = Path(entry.source)
        if source.stat().st_size != entry.source_size or source.stat().st_mtime_ns != entry.source_mtime_ns:
            raise ValueError(f'Raw source changed: {source}')
        for suffix, field in [('.analysis.txt', 'analysis_sha'), ('.triggers.txt', 'triggers_sha')]:
            if hashlib.sha256(source.with_suffix(suffix).read_bytes()).hexdigest() != getattr(entry, field):
                raise ValueError(f'Paired file changed: {source}{suffix}')
        if entry.signature != signature:
            raise ValueError('Stale settings/extraction signature.')
        stem = f'{entry.subject_id}_s{entry.session_id}_r{entry.run_id:02d}'
        frame = pd.read_csv(root / 'runs' / f'{stem}.csv')
        if len(frame) != 60 or frame.trial_index.tolist() != list(range(1, 61)):
            raise ValueError(f'{stem}: invalid ordered trial identifiers.')
        if not frame[KEYS + ['run_id']].eq([entry.subject_id, entry.group, entry.session_id, entry.run_id]).all().all():
            raise ValueError(f'{stem}: identity mismatch.')
        with np.load(root / 'runs' / f'{stem}.npz', allow_pickle=False) as archive:
            z = {key: archive[key] for key in archive.files}
            if z['posterior_erp_uv'].shape != (60, 2, 1024) or z['response_posterior_uv'].shape != (60, 2, 411):
                raise ValueError(f'{stem}: posterior epoch shape mismatch.')
            time, response_time = z['time'], z['response_time']
            if not np.allclose(time, np.arange(-512, 512)/512) or not np.allclose(response_time, np.arange(-205,206)/512):
                raise ValueError(f'{stem}: incorrect time axes.')
            if np.max(abs(z['posterior_erp_uv'][:, :, (time >= -.2) & (time < 0)].mean(axis=-1))) > 1e-4:
                raise ValueError(f'{stem}: baseline residual.')
            complete = np.isfinite(z['response_posterior_uv']).all(axis=(1,2))
            if not np.array_equal(complete, frame.response_epoch_available):
                raise ValueError(f'{stem}: missing response epoch flags mismatch.')
            for i in np.flatnonzero(complete):
                stim_index = np.arange(-205,206) + int(round(frame.gdf_rt_ms.iloc[i]*.512)) + 512
                overlap = (stim_index >= 0) & (stim_index < 1024)
                if not overlap.any():
                    # Very slow responses can lie wholly after the stored
                    # stimulus epoch; their continuous-data response epoch is valid.
                    continue
                error = abs(z['response_posterior_uv'][i][:, overlap] - z['posterior_erp_uv'][i][:,stim_index[overlap]]).max()
                maximum_alignment_error = max(maximum_alignment_error, float(error))
            for key in z:
                value = z[key]
                if key in ['time', 'response_time', 'channel_labels', 'spectral_frequencies']:
                    if key in arrays and not np.array_equal(arrays[key], value):
                        raise ValueError(f'{stem}: inconsistent {key}.')
                    arrays[key] = value.copy()
                else:
                    if value.shape[0] != 60 or (key != 'response_posterior_uv' and not np.isfinite(value).all()):
                        raise ValueError(f'{stem}: invalid {key}.')
                    arrays.setdefault(key, []).append(value.copy())
        frames.append(frame)
    if maximum_alignment_error > 1e-4:
        raise ValueError(f'Response/stimulus overlap mismatch: {maximum_alignment_error}')
    joined = pd.concat(frames, ignore_index=True)
    pd.testing.assert_frame_equal(joined, data, check_exact=False, rtol=1e-10, atol=1e-10)
    for key in list(arrays):
        if isinstance(arrays[key], list):
            arrays[key] = np.concatenate(arrays[key])
    for (subject, session), cell in data.groupby(['subject_id', 'session_id']):
        expected = 7 if (subject, session) == ('e30', 1) else {1:8,5:4}[session]
        if len(cell) != expected*60 or cell.run_id.nunique() != expected:
            raise ValueError(f'{subject}/{session}: unexpected coverage.')
    print(f'Validated {len(data):,} trials, all cached arrays and sources; response alignment max error={maximum_alignment_error:.3g} uV.')
    return data, arrays, dict(n_trials=len(data), n_runs=len(logs), n_subjects=data.subject_id.nunique(),
        response_alignment_max_error_uv=maximum_alignment_error,
        n_status_missing=int(data.gdf_rt_ms.isna().sum()), n_response_epochs=int(data.response_epoch_available.sum()),
        signature=signature)


def summarize_neural_candidates(data, arrays, threshold=150, correct_only=False, average_reference=False):
    """Use equal condition/side means and subject-level repeated observations."""
    if threshold not in (100,150,200,np.inf) or len(data) != len(arrays['posterior_erp_uv']):
        raise ValueError('Invalid threshold or trial-array alignment.')
    ref = 'average' if average_reference else 'recorded'
    base = data.feedback.eq(1).to_numpy() if correct_only else np.ones(len(data), bool)
    po, time = arrays['posterior_erp_uv'], arrays['time']
    rows, counts = [], []
    for identity, cell in data.groupby(KEYS, sort=True):
        ids = cell.index.to_numpy()
        row = dict(zip(KEYS, identity))
        count = dict(row)
        valid = base[ids] & (cell.pd_epoch_ptp_uv.to_numpy() <= threshold)
        sides = []
        for side, sign in [(2,1),(4,-1)]:
            chosen = ids[valid & cell.distractor_position.eq(side).to_numpy()]
            if len(chosen) < 10:
                raise ValueError(f'{identity}: <10 ERP trials for distractor side {side}.')
            sides.append(sign*(po[chosen,0]-po[chosen,1]).mean(axis=0))
            count[f'erp_n_{side}'] = len(chosen)
        wave = np.mean(sides,axis=0)
        for metric, bounds in [('late_lateral',(.5,.9)),('pd_signed',(.2,.5)),('early_distractor',(.18,.3))]:
            mask = (time>=bounds[0]) & (time<=bounds[1])
            row[metric] = wave[mask].mean()
        mask = (time>=.2)&(time<=.5)
        row['pd_positive_auc'] = np.trapz(np.maximum(wave[mask],0), time[mask])
        target_waves = []
        for side, sign in [(2,1),(4,-1)]:
            chosen = ids[valid & cell.task.eq(0).to_numpy() & cell.target_position.eq(side).to_numpy()]
            if len(chosen)<10:
                raise ValueError(f'{identity}: <10 lateral target trials for side {side}.')
            target_waves.append(sign*(po[chosen,0]-po[chosen,1]).mean(axis=0))
            count[f'target_n_{side}'] = len(chosen)
        row['target_n2pc'] = np.mean(target_waves,axis=0)[(time>=.18)&(time<=.3)].mean()
        # Geometry control: both conditions restricted to vertical targets.
        for condition in (0,1):
            chosen = ids[valid & cell.task.eq(condition).to_numpy() & cell.target_position.isin([1,3]).to_numpy()]
            if len(chosen)<20:raise ValueError(f'{identity}: too few midline-target trials.')
            row[f'late_bilateral_{condition}'] = po[chosen][:,:,(time>=.5)&(time<=.9)].mean()
        row['late_bilateral_d_minus_n'] = row['late_bilateral_1'] - row['late_bilateral_0']
        rows.append(row);counts.append(count)
    summary=pd.DataFrame(rows)
    validate_cohort_table(summary,['subject_id','session_id'])
    if not np.isfinite(summary.select_dtypes('number')).all().all():raise ValueError('Nonfinite neural summaries.')
    print(f'Summarized {len(summary)} cells: threshold {threshold}, correct-only {correct_only}, reference {ref}.')
    return summary,pd.DataFrame(counts)


def response_and_count_controls(data, arrays, seed=20261013, resamples=500):
    """RT-stratified pre/post matching, response locking, and equal-count Pd AUC."""
    if resamples<100 or len(data)!=len(arrays['posterior_erp_uv']):raise ValueError('Invalid resampling/alignment.')
    rng=np.random.default_rng(seed)
    time=arrays['time'];rtime=arrays['response_time'];po=arrays['posterior_erp_uv'];rp=arrays['response_posterior_uv']
    d=data.copy()
    d['rt_bin']=np.floor(d.gdf_rt_ms/25)
    valid=(d.feedback.eq(1)&d.pd_epoch_ptp_uv.le(150)&d.response_roi_ptp_uv.le(150)&
           d.response_epoch_available&d.gdf_rt_ms.ge(150)&d.distractor_position.isin([2,4]))
    d['matched']=False
    for subject in EXPECTED_SUBJECTS:
        for side in (2,4):
            sub=d[valid&d.subject_id.eq(subject)&d.distractor_position.eq(side)]
            for _,bin_trials in sub.groupby('rt_bin'):
                pre=bin_trials.index[bin_trials.session_id.eq(1)].to_numpy();post=bin_trials.index[bin_trials.session_id.eq(5)].to_numpy()
                n=min(len(pre),len(post))
                if n:
                    d.loc[rng.choice(pre,n,replace=False),'matched']=True
                    d.loc[rng.choice(post,n,replace=False),'matched']=True
    rows=[];waves=[];counts=[]
    for identity,cell in d.groupby(KEYS):
        row=dict(zip(KEYS,identity));count=dict(row)
        for selection in ('all_correct','matched'):
            side_waves=[];side_responses=[];side_ipsi=[];side_contra=[];side_pre=[];side_rt=[]
            for side,sign in [(2,1),(4,-1)]:
                idx=cell.index[valid[cell.index]&cell.distractor_position.eq(side)&(cell.matched if selection=='matched' else True)].to_numpy()
                if len(idx)<10:raise ValueError(f'{identity}/{selection}: fewer than 10 trials for side {side}.')
                side_waves.append(sign*(po[idx,0]-po[idx,1]).mean(axis=0))
                side_responses.append(sign*(rp[idx,0]-rp[idx,1]).mean(axis=0))
                contra,ipsi=(0,1) if side==2 else (1,0)
                side_contra.append(po[idx,contra].mean(axis=0));side_ipsi.append(po[idx,ipsi].mean(axis=0))
                count[f'{selection}_n_{side}']=len(idx)
                side_rt.append(d.loc[idx,'gdf_rt_ms'].mean())
                side_pre.append((d.loc[idx,'gdf_rt_ms']<=500).mean())
            sw=np.mean(side_waves,axis=0);rw=np.mean(side_responses,axis=0)
            row[f'late_{selection}']=sw[(time>=.5)&(time<=.9)].mean()
            row[f'response_pre_{selection}']=rw[(rtime>=-.2)&(rtime<=-.05)].mean()
            row[f'response_post_{selection}']=rw[(rtime>=.05)&(rtime<=.35)].mean()
            count[f'{selection}_rt_ms']=np.mean(side_rt)
            count[f'{selection}_fraction_responded_500']=np.mean(side_pre)
            waves.append(dict(**dict(zip(KEYS,identity)),selection=selection,stimulus=sw,response=rw,
                              contra=np.mean(side_contra,axis=0),ipsi=np.mean(side_ipsi,axis=0)))
        rows.append(row);counts.append(count)
    # Equalize pre/post trial counts separately for each side, then rectify the
    # subject-average waveform. Resamples are NOT treated as independent people.
    equal=[];m=(time>=.2)&(time<=.5)
    for subject in EXPECTED_SUBJECTS:
        ids={}
        for side in (2,4):
            for session in (1,5):
                ids[side,session]=d.index[d.subject_id.eq(subject)&d.session_id.eq(session)&d.distractor_position.eq(side)&d.pd_epoch_ptp_uv.le(150)].to_numpy()
        n={side:min(len(ids[side,1]),len(ids[side,5])) for side in (2,4)}
        if min(n.values())<10:raise ValueError('Insufficient matched counts.')
        for session in (1,5):
            values=[]
            for _ in range(resamples):
                sw=[]
                for side,sign in [(2,1),(4,-1)]:
                    chosen=rng.choice(ids[side,session],n[side],replace=False)
                    sw.append(sign*(po[chosen,0][:,m]-po[chosen,1][:,m]).mean(axis=0))
                values.append(np.trapz(np.maximum(np.mean(sw,axis=0),0),time[m]))
            equal.append(dict(subject_id=subject,session_id=session,pd_auc_equal_count=np.mean(values),equal_count_left=n[4],equal_count_right=n[2]))
    summary=pd.DataFrame(rows).merge(pd.DataFrame(equal),on=['subject_id','session_id'],validate='one_to_one')
    counts=pd.DataFrame(counts)
    balance=counts.pivot(index='subject_id',columns='session_id',values='matched_rt_ms')
    print(f'RT controls: minimum matched side count={counts[["matched_n_2","matched_n_4"]].min().min()}, max pre/post RT imbalance={abs(balance[5]-balance[1]).max():.2f} ms; {resamples} equal-count draws.')
    return summary,counts,waves,d[['subject_id','session_id','run_id','trial_index','matched']]


def reconcile_pd_and_reliability(data, arrays, output_dir, n_splits=200, seed=20261015):
    """Reproduce saved Pd, assess threshold sensitivity and disjoint-trial stability."""
    from .pd_historical import resolve_training_condition_eeg_averages_csv
    if n_splits<100 or len(data)!=len(arrays['posterior_erp_uv']):raise ValueError('Invalid reliability inputs.')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    path=resolve_training_condition_eeg_averages_csv()
    saved=pd.read_csv(path,usecols=['subject_id','group','session_id','condition','sample_index','time_sec','PO7','PO8'])
    validate_cohort_table(saved,['subject_id','session_id','condition','sample_index'])
    time=arrays['time'];po=arrays['posterior_erp_uv'].astype('float64');mask=(time>=-.5)
    reconciliation=[];raw_rows=[];sensitivities=[]
    for identity,cell in data.groupby(KEYS):
        sides=[]
        for side,sign,condition in [(2,1,'distractor_right'),(4,-1,'distractor_left')]:
            idx=cell.index[cell.distractor_position.eq(side)].to_numpy()
            wave=sign*(po[idx,0]-po[idx,1]).mean(axis=0)
            source=saved[(saved.subject_id==identity[0])&(saved.session_id==identity[2])&(saved.condition==condition)].sort_values('sample_index')
            if len(source)!=768 or not np.allclose(source.time_sec,time[mask]):raise ValueError('Saved waveform grid mismatch.')
            old=sign*(source.PO7-source.PO8).to_numpy()
            error=abs(wave[mask]-old).max()
            if error>1e-4:raise ValueError(f'{identity}/{side}: raw/saved Pd differs by {error}.')
            reconciliation.append(dict(zip(KEYS,identity))|dict(side=side,max_error_uv=error,n_trials=len(idx)))
            sides.append(wave)
        wave=np.mean(sides,axis=0);m=(time>=.2)&(time<=.5)
        raw_rows.append(dict(zip(KEYS,identity))|dict(pd_positive_auc=np.trapz(np.maximum(wave[m],0),time[m]),
            late_negative_auc=np.trapz(np.minimum(wave[(time>=.5)&(time<=.9)],0),time[(time>=.5)&(time<=.9)])))
    raw=pd.DataFrame(raw_rows)
    for threshold in (100,150,200,np.inf):
        summary,_=summarize_neural_candidates(data,arrays,threshold=threshold)
        tests,_=compare_group_changes(summary,['pd_signed','pd_positive_auc'],n_permutations=999)
        tests['threshold']=threshold;sensitivities.append(tests)
    # Repeat existing balanced 2x2 interaction: F(1,30) equals pooled t^2 on change.
    legacy=[]
    for metric in ['pd_positive_auc','late_negative_auc']:
        wide=raw.pivot(index=['subject_id','group'],columns='session_id',values=metric)
        delta=(wide[5]-wide[1]).reset_index(name='change')
        a=delta.loc[delta.group.eq('bci'),'change'];b=delta.loc[delta.group.eq('control'),'change']
        test=stats.ttest_ind(a,b,equal_var=True)
        legacy.append(dict(metric=metric,bci_change=a.mean(),control_change=b.mean(),F=test.statistic**2,df1=1,df2=30,p=test.pvalue))
    rng=np.random.default_rng(seed);reliability=[];cross=[]
    # Preserve the original RNG draw order for the retained Pd/late split identities.
    # Spectral strata only advance that historical sequence; no spectral results are emitted.
    metric_names=['pre_alpha','theta_change','pd_signed','late_lateral']
    for split in range(n_splits):
        records=[]
        for identity,cell in data.groupby(KEYS):
            half_records=[dict(zip(KEYS,identity))|dict(half=h) for h in (0,1)]
            for metric in metric_names:
                if metric=='pre_alpha':valid=cell.pre_alpha_ptp_uv.le(150);strata=cell.task;column='pre_alpha_log10_recorded'
                elif metric=='theta_change':valid=cell[['pre_theta_ptp_uv','post_theta_ptp_uv']].max(axis=1).le(150);strata=cell.task;column='theta_log10_post_pre'
                else:valid=cell.pd_epoch_ptp_uv.le(150)&cell.task.eq(1);strata=cell.distractor_position;column={'pd_signed':'pd_lateral_mean','late_lateral':'late_lateral_mean'}[metric]
                # Run-stratified splitting prevents halves representing different run periods.
                half_means=[[],[]]
                for side in sorted(strata[valid].unique()):
                    side_halves=[[],[]]
                    for _,run in cell[valid & strata.eq(side)].groupby('run_id'):
                        ids=rng.permutation(run.index.to_numpy());n=len(ids)//2
                        if n<1:raise ValueError('Insufficient split-half strata.')
                        for h in (0,1):side_halves[h].extend(ids[h*n:(h+1)*n])
                    for h in (0,1):half_means[h].append(data.loc[side_halves[h],column].mean())
                for h in (0,1):half_records[h][metric]=np.mean(half_means[h])
            records.extend(half_records)
        half=pd.DataFrame(records)
        for metric in ['pd_signed', 'late_lateral']:
            wide=half.pivot(index=['subject_id','group','session_id'],columns='half',values=metric)
            for session in (1,5):
                pair=wide.xs(session,level='session_id');r=np.corrcoef(pair[0],pair[1])[0,1]
                reliability.append(dict(split=split,metric=metric,level=f'session_{session}',r=r,spearman_brown=2*r/(1+r)))
            change=wide.xs(5,level='session_id')-wide.xs(1,level='session_id');r=np.corrcoef(change[0],change[1])[0,1]
            reliability.append(dict(split=split,metric=metric,level='post_pre_change',r=r,spearman_brown=2*r/(1+r)))
        # Independent trial sets for Pd and late negativity; shared participants remain.
        # Use one COMMON split of ERP trials below, because independent re-splits
        # per metric above would overlap and cannot assess shared trial noise.
        erp=[]
        for identity,cell in data.groupby(KEYS):
            values={0:{'pd_signed':[],'late_lateral':[]},1:{'pd_signed':[],'late_lateral':[]}}
            for side in (2,4):
                ids=cell.index[cell.pd_epoch_ptp_uv.le(150)&cell.distractor_position.eq(side)].to_numpy()
                ids=rng.permutation(ids);n=len(ids)//2
                for h in (0,1):
                    picked=ids[h*n:(h+1)*n]
                    values[h]['pd_signed'].append(data.loc[picked,'pd_lateral_mean'].mean())
                    values[h]['late_lateral'].append(data.loc[picked,'late_lateral_mean'].mean())
            for h in (0,1):erp.append(dict(zip(KEYS,identity))|dict(half=h,**{k:np.mean(v) for k,v in values[h].items()}))
        e=pd.DataFrame(erp);deltas={}
        for metric in ['pd_signed','late_lateral']:
            w=e.pivot(index=['subject_id','group','half'],columns='session_id',values=metric)
            deltas[metric]=(w[5]-w[1]).unstack('half')
        gs=(deltas['pd_signed'].index.get_level_values('group')=='bci').astype(float);X=np.c_[np.ones(len(gs)),gs]
        for h in (0,1):
            x=deltas['pd_signed'][h].to_numpy();y=deltas['late_lateral'][1-h].to_numpy()
            x=x-X@np.linalg.lstsq(X,x,rcond=None)[0];y=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
            cross.append(dict(split=split,pd_half=h,late_half=1-h,r_group_adjusted=np.corrcoef(x,y)[0,1]))
    rel=pd.DataFrame(reliability);cross=pd.DataFrame(cross)
    rel_summary=rel.groupby(['metric','level']).spearman_brown.agg(median='median',p05=lambda x:x.quantile(.05),p95=lambda x:x.quantile(.95)).reset_index()
    tables=dict(pd_source_reconciliation=pd.DataFrame(reconciliation),pd_legacy_reproduction=pd.DataFrame(legacy),
                pd_threshold_sensitivity=pd.concat(sensitivities,ignore_index=True),reliability_splits=rel,
                reliability_summary=rel_summary,disjoint_pd_late_correlations=cross)
    for name,frame in tables.items():frame.to_csv(out/f'{name}.csv',index=False)
    print(f'Saved Pd reproduced, max waveform error={max(r["max_error_uv"] for r in reconciliation):.3g} uV; {n_splits} reliability splits (descriptive, not independent tests).')
    return tables


def behavioral_geometry_controls(data, cleaned_rt_csv, output_dir):
    """Check speed/accuracy and distractor cost with identical target positions."""
    clean=pd.read_csv(cleaned_rt_csv).rename(columns={'trial':'trial_index'})
    keys=['subject_id','session_id','run_id','trial_index']
    if clean.duplicated(keys).any() or not clean.feedback.eq(1).all():raise ValueError('Invalid cleaned RT keys/correctness.')
    joined=clean[keys+['gdf_rt_ms']].merge(data[keys+['gdf_rt_ms']],on=keys,validate='one_to_one',suffixes=('_clean','_eeg'))
    if len(joined)!=len(clean) or not np.allclose(joined.gdf_rt_ms_clean,joined.gdf_rt_ms_eeg):raise ValueError('Cleaned RT alignment mismatch.')
    rows=[]
    for identity,cell in data.groupby(KEYS):
        sub=clean[(clean.subject_id==identity[0])&(clean.session_id==identity[2])]
        means={}
        for task in (0,1):
            for target in (1,3):
                q=sub[(sub.task==task)&(sub.target_position==target)]
                if len(q)<10:raise ValueError('Insufficient position-matched correct RT trials.')
                means[task,target]=q.gdf_rt_ms.mean()
        row=dict(zip(KEYS,identity))|dict(accuracy_percent=100*cell.feedback.eq(1).mean(),timeout_percent=100*cell.feedback.eq(3).mean(),
            vertical_target_cost_ms=np.mean([means[1,t]-means[0,t] for t in (1,3)]),
            vertical_target_d_rt_ms=np.mean([means[1,t] for t in (1,3)]),vertical_target_n_rt_ms=np.mean([means[0,t] for t in (1,3)]))
        rows.append(row)
    summary=pd.DataFrame(rows)
    tests,changes=compare_group_changes(summary,['accuracy_percent','timeout_percent','vertical_target_cost_ms'])
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('behavior_geometry_summary',summary),('behavior_geometry_tests',tests),('behavior_geometry_changes',changes)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    print('Validated speed/accuracy and equally weighted top/bottom target RT contrasts; same previously cleaned trials.')
    return dict(summary=summary,tests=tests,changes=changes)

