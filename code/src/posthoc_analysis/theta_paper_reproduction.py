"""Raw-data bootstrap and numerical audit for the focused midline-theta notebook.

No historical notebook, exploratory ROI extraction or behavioral CSV is required.
The optional historical Laplacian cache is an optimization, never an input need.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import shutil

import numpy as np
import pandas as pd


def prepare_theta_paper_inputs(project_root, output_dir):
    """Discover assessment runs, extract Status RT, and reproduce behavioral cleaning."""
    import mne
    from .config import EXPECTED_SUBJECTS, get_subject_group
    from .eeg import _get_training_run_gdf_files_for_session
    from .analysis import load_training_analysis_file
    from .triggers import load_training_trigger_file
    from .behavioral import _prepare_reaction_time_trial_data, _remove_rt_outliers
    from .theta_topography import theta_topography_settings
    from .theta_laplacian import make_laplacian_theta_settings

    project_root=Path(project_root).expanduser().resolve();out=Path(output_dir).resolve()
    if not project_root.is_dir():raise FileNotFoundError(f'Study root not found: {project_root}')
    rows=[];records=[];canonical=None
    for subject in EXPECTED_SUBJECTS:
        for session in (1,5):
            files=_get_training_run_gdf_files_for_session(subject,session,project_root=project_root,
                allow_incomplete=(subject=='e30' and session==1))
            expected=7 if subject=='e30' and session==1 else 8 if session==1 else 4
            if len(files)!=expected:raise ValueError(f'{subject}/S{session}: expected {expected} non-practice runs.')
            for run_id,path in enumerate(files,1):
                analysis=load_training_analysis_file(path.with_suffix('.analysis.txt')).rename(columns={'trial_index':'trial'})
                triggers=load_training_trigger_file(path.with_suffix('.triggers.txt'))
                raw=mne.io.read_raw_gdf(path,preload=False,verbose='ERROR')
                if raw.info['sfreq']!=512 or len(raw.ch_names)!=67 or len(set(raw.ch_names))!=67:
                    raise ValueError(f'{path}: expected 67 distinct channels at 512 Hz.')
                labels=[c for c in raw.ch_names if c not in ['M1','M2','EOG','sens7','sens8','Status','CPZ']]
                if len(labels)!=61 or 'FCZ' not in labels or 'CPZ' in raw.ch_names:
                    raise ValueError(f'{path}: scalp/reference layout differs from the paper.')
                if canonical is None:canonical=labels
                if labels!=canonical:raise ValueError(f'{path}: changed channel order.')
                events=mne.find_events(raw,stim_channel='Status',shortest_event=1,verbose='ERROR')
                stimuli=events[np.isin(events[:,2],[8,32,44])]
                codes=analysis.distractor_position.map({0:8,2:32,4:44}).to_numpy()
                if (stimuli.shape!=(60,3) or not np.array_equal(stimuli[:,2],codes)
                        or not np.array_equal(triggers.loc[triggers.trigger.isin([8,32,44]),'trigger'],codes)
                        or not analysis.task.eq(analysis.distractor_position.ne(0).astype(int)).all()
                        or analysis.task.value_counts().to_dict()!={0:30,1:30}):
                    raise ValueError(f'{path}: EEG/task trial alignment failed.')
                rt=np.full(60,np.nan)
                for i,onset in enumerate(stimuli[:,0]):
                    end=stimuli[i+1,0] if i<59 else raw.first_samp+raw.n_times
                    response=events[(events[:,2]==64)&(events[:,0]>onset)&(events[:,0]<end),0]
                    if len(response)>1:raise ValueError(f'{path}: multiple responses in trial {i+1}.')
                    if len(response):rt[i]=(response[0]-onset)/512*1000
                analysis=analysis.assign(subject_id=subject,group=get_subject_group(subject),session_id=session,
                    run_id=run_id,gdf_rt_ms=rt,rt_ms=rt,stimulus_code=codes,task_type='training')
                rows.append(analysis)
                stat=path.stat()
                records.append(dict(subject_id=subject,group=get_subject_group(subject),session_id=session,run_id=run_id,
                    source=str(path.resolve()),source_size=stat.st_size,source_mtime_ns=stat.st_mtime_ns,
                    analysis_sha=hashlib.sha256(path.with_suffix('.analysis.txt').read_bytes()).hexdigest(),
                    triggers_sha=hashlib.sha256(path.with_suffix('.triggers.txt').read_bytes()).hexdigest(),
                    n_status_responses=int(np.isfinite(rt).sum()),status='pass'))
                raw.close()
            print(f'{subject}/S{session}: validated {len(files)} GDF/analysis/trigger sets; extracted Status RT.',flush=True)
    data=pd.concat(rows,ignore_index=True);inventory=pd.DataFrame(records)
    if len(data)!=22980 or len(inventory)!=383:raise ValueError('Paper cohort requires 383 runs and 22,980 trials.')
    behavioral=data.copy();behavioral['group']=behavioral.group.replace({'bci':'experimental'})
    eligible,_,_=_prepare_reaction_time_trial_data(behavioral)
    cleaned,_=_remove_rt_outliers(eligible)
    keys=['subject_id','session_id','run_id','trial']
    ledger=data.merge(cleaned[keys].assign(retained=True),on=keys,how='left',validate='one_to_one')
    ledger['retained']=ledger.retained.eq(True)
    ledger['missing_status_response']=ledger.gdf_rt_ms.isna()
    valid=ledger.feedback.eq(1)&ledger.gdf_rt_ms.ge(150)
    x=ledger.loc[valid].copy();g=x.groupby(['subject_id','session_id']).gdf_rt_ms
    manual=(x.gdf_rt_ms-g.transform('mean')).abs()<=3*g.transform('std')
    if set(x.loc[manual].index)!=set(ledger.index[ledger.retained]):raise ValueError('Independent RT cleaning disagrees.')
    ledger['exclusion']=np.select([ledger.feedback.ne(1),ledger.missing_status_response,
        ledger.gdf_rt_ms.lt(150),~ledger.retained],
        ['incorrect_or_timeout','missing_status_response','rt_below_150_ms','outside_pooled_session_3sd'],default='included')
    exclusions={'included':21940,'incorrect_or_timeout':700,'missing_status_response':24,'rt_below_150_ms':3,'outside_pooled_session_3sd':313}
    if ledger.exclusion.value_counts().to_dict()!=exclusions or ledger.missing_status_response.sum()!=47:
        raise ValueError('Raw-derived exclusions differ from the published cohort; inspect instead of silently continuing.')
    ledger=ledger.sort_values(keys).reset_index(drop=True)
    cache=out/'theta_cache';cache.mkdir(parents=True,exist_ok=True)
    ledger.to_csv(cache/'trial_mask.csv',index=False)
    mask_qc=dict(n_trials=22980,n_missing_status_response=47,n_retained=21940,n_excluded=1040,exclusions=exclusions,
        rule='Correct; EEG Status RT >=150 ms; one-pass pooled participant/session +/-3 sample SD',
        mask_sha256=hashlib.sha256((cache/'trial_mask.csv').read_bytes()).hexdigest())
    (cache/'mask_validation.json').write_text(json.dumps(mask_qc,indent=2)+'\n')
    rt_dir=out/'rt';rt_dir.mkdir(exist_ok=True)
    ledger.loc[ledger.retained].to_csv(rt_dir/'trial_data_clean.csv',index=False)
    inventory.to_csv(out/'source_inventory.csv',index=False)
    settings=theta_topography_settings()
    settings.update(rt_cleaned=True,expected_excluded_trials=1040,
        trial_selection='Correct only; EEG RT >=150 ms; within participant/session mean +/-3 sample SD after minimum-RT exclusion; missing responses excluded',
        baseline_pool='retained trials of both conditions within participant/session/channel/frequency')
    settings,matrix,transform_qc=make_laplacian_theta_settings(canonical,settings)
    (cache/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    np.save(cache/'laplacian_matrix.npy',matrix)
    (cache/'transform_validation.json').write_text(json.dumps(transform_qc,indent=2)+'\n')
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','mne','matplotlib','statsmodels']}
    (out/'environment.json').write_text(json.dumps(versions,indent=2)+'\n')
    print('Raw bootstrap complete: 47 missing responses; exact 21,940-trial mask reproduced independently. No historical CSV required.')
    return inventory,ledger,settings


def build_theta_paper_cache(inventory, ledger, settings, output_dir, reuse_cache_dir=None, n_jobs=2):
    """Build portable local run checkpoints, optionally importing exact validated caches."""
    from .theta_laplacian import _cache_laplacian_run
    if n_jobs not in (1,2) or len(inventory)!=383 or len(ledger)!=22980 or ledger.retained.sum()!=21940:
        raise ValueError('Invalid paper cohort or worker count.')
    out=Path(output_dir).resolve();runs=out/'runs';runs.mkdir(parents=True,exist_ok=True)
    reused=0
    source=Path(reuse_cache_dir) if reuse_cache_dir else None
    # Cache reuse is optional. An absent cache always falls through to raw EEG.
    if source and (source/'settings.json').is_file():
        oldsettings=json.loads((source/'settings.json').read_text())
        if oldsettings!=settings:
            print('Optional cache settings differ; using raw recomputation instead.')
        elif source.resolve()!=out:
            for row in inventory.itertuples():
                key=f'{row.subject_id}_s{row.session_id}_r{row.run_id:02d}'
                oldjson=source/'runs'/f'{key}.json';oldnpz=source/'runs'/f'{key}.npz'
                if not oldjson.is_file() or not oldnpz.is_file() or (runs/f'{key}.npz').exists():continue
                meta=json.loads(oldjson.read_text())
                if (any(meta.get(k)!=getattr(row,k) for k in ['source','source_size','source_mtime_ns','analysis_sha','triggers_sha'])
                        or hashlib.sha256(oldnpz.read_bytes()).hexdigest()!=meta['npz_sha']):
                    continue
                shutil.copy2(oldnpz,runs/oldnpz.name)
                meta['npz_path']=str(runs/oldnpz.name)
                (runs/oldjson.name).write_text(json.dumps(meta,indent=2)+'\n');reused+=1
            print(f'Imported {reused} independently hash-checked optional run caches; each will also be checked against the new raw-derived mask.')
    cells={k:v for k,v in ledger.groupby(['subject_id','session_id','run_id'])}
    records=inventory.to_dict('records')
    def work(record):
        cell=cells[(record['subject_id'],record['session_id'],record['run_id'])]
        return _cache_laplacian_run(record,cell,settings,runs)
    log=[]
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        for result in pool.map(work,records):
            log.append(result)
            pd.DataFrame(log).to_csv(out/'run_log_progress.csv',index=False)
    frame=pd.DataFrame(log)
    if len(frame)!=383 or frame.n_retained.sum()!=21940 or frame.n_excluded.sum()!=1040:
        raise ValueError('Incomplete final theta cache.')
    frame.to_csv(out/'run_log.csv',index=False)
    if not all(Path(p).parent==runs for p in frame.npz_path):raise ValueError('Cache still depends on external run paths.')
    print('Complete portable theta cache: 383 local checkpoints, 21,940 retained trials.')
    return frame


def seal_theta_paper_outputs(output_dir):
    """Record actual produced file hashes for downstream loaders; no reference data."""
    root=Path(output_dir).resolve()
    files=[p for p in root.iterdir() if p.is_file() and p.name!='final_manifest.json']
    if not files:raise ValueError('Cannot seal an empty output directory.')
    manifest=dict(files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (root/'final_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Checksummed {len(files)} generated inputs in {root.name}.')
    return manifest


def summarize_theta_paper_cache(cache_dir):
    """Aggregate run caches and validate cohort before downstream paper analyses."""
    from .theta_topography import summarize_theta_topographies
    root=Path(cache_dir);table=summarize_theta_topographies(root)
    if len(table)!=11712 or table.subject_id.nunique()!=32:raise ValueError('Incomplete paper theta summary.')
    (root/'validation.json').write_text(json.dumps(dict(n_subjects=32,n_runs=383,n_retained=21940,
        checks='All checkpoint hashes, shapes, frequency/time axes, trial counts and session pairings verified by summarize_theta_topographies.'),indent=2)+'\n')
    seal_theta_paper_outputs(root)
    return table


def validate_theta_paper_results(anova, roi_changes, windows, correlations, output_dir):
    """Check computed outputs against the numbers printed in the supplied PDF.

    These expected values are assertions only; they are never analysis inputs.
    RT improvement is pre minus post, matching the PDF directly.
    """
    if ('rt_difference' not in correlations
            or not correlations.rt_difference.eq('pre_minus_post').all()
            or 'rt_measure' not in correlations
            or not correlations.rt_measure.eq('rt_improvement_ms').all()):
        raise ValueError('Expected pre-minus-post RT improvement correlations for comparison to the PDF.')
    print('PDF comparison: RT improvement uses pre minus post directly; no sign conversion.')
    rows=[]
    expected_anova=[('Group',8.86,.0057,.228),('Distractor presence',3.00,.093,.091),('Group × Distractor presence',.23,.638,.007)]
    for effect,f,p,eta in expected_anova:
        row=anova.set_index('effect').loc[effect]
        for field,target,tolerance in [('F',f,.0051),('p_value',p,.000051 if effect=='Group' else .00051),('partial_eta_sq',eta,.00051)]:
            rows.append(dict(result=f'{effect}: {field}',computed=float(row[field]),pdf_value=target,tolerance=tolerance))
    for group,target in [('bci',.520),('control',-.409)]:
        rows.append(dict(result=f'{group} condition-equal theta change',computed=float(roi_changes[roi_changes.group==group].change_db.mean()),pdf_value=target,tolerance=.00051))
    target_windows=[('bci',1,2.050,2.513,.917),('bci',5,2.240,3.070,1.440),('control',1,1.538,1.997,1.076),('control',5,1.820,2.055,.679)]
    for group,session,b,p,d in target_windows:
        row=windows.set_index(['group','session_id']).loc[group,session]
        for field,target,scale in [('baseline_power',b,1e8),('stimulus_power',p,1e8),('normalized_theta_db',d,1.)]:
            rows.append(dict(result=f'{group}/S{session}: {field}',computed=float(row[field]/scale),pdf_value=target,tolerance=.00051))
    for scope,r,p in [('bci',.506,.045),('control',-.022,.935),('pooled_unadjusted',.427,.015)]:
        row=correlations.set_index('scope').loc[scope]
        for field,target in [('pearson_r',r),('p_value',p)]:
            rows.append(dict(result=f'{scope}: {field}',computed=float(row[field]),pdf_value=target,tolerance=.00051))
    bci=correlations.set_index('scope').loc['bci']
    for field,target in [('ci95_low',.014),('ci95_high',.801),('spearman_rho',.574),('spearman_p',.020)]:
        rows.append(dict(result=f'bci: {field}',computed=float(bci[field]),pdf_value=target,tolerance=.00051))
    group=anova.set_index('effect').loc['Group']
    for field,target in [('estimate_db',.929),('ci95_low',.291),('ci95_high',1.567)]:
        rows.append(dict(result=f'group contrast: {field}',computed=float(group[field]),pdf_value=target,tolerance=.00051))
    checks=pd.DataFrame(rows);checks['matches_pdf']=(checks.computed-checks.pdf_value).abs()<=checks.tolerance
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);checks.to_csv(out/'pdf_numerical_reproduction.csv',index=False)
    if not checks.matches_pdf.all():raise ValueError('Computed results differ from the supplied PDF:\n'+checks.loc[~checks.matches_pdf].to_string(index=False))
    print(f'All {len(checks)} numerical checks match the supplied PDF at its reported precision.')
    return checks


def verify_theta_paper_raw_checkpoint(inventory, ledger, settings, cache_dir, output_dir):
    """Exercise the no-cache raw computation for one complete run and compare sums."""
    from .theta_laplacian import _cache_laplacian_run
    import tempfile
    if len(inventory)!=383 or len(ledger)!=22980:raise ValueError('Expected full-cohort bootstrap inputs.')
    record=inventory.iloc[0].to_dict()
    cell=ledger[(ledger.subject_id==record['subject_id'])&(ledger.session_id==record['session_id'])&(ledger.run_id==record['run_id'])]
    key=f"{record['subject_id']}_s{record['session_id']}_r{record['run_id']:02d}"
    with tempfile.TemporaryDirectory(prefix='theta-paper-raw-') as tmp:
        fresh=_cache_laplacian_run(record,cell,settings,Path(tmp))
        errors={}
        with np.load(fresh['npz_path'],allow_pickle=False) as a,np.load(Path(cache_dir)/'runs'/f'{key}.npz',allow_pickle=False) as b:
            for name in ['baseline_sum','post_sum']:
                errors[name]=float(np.max(abs(a[name]-b[name])/np.maximum(abs(b[name]),1e-30)))
                if not np.allclose(a[name],b[name],rtol=1e-10,atol=1e-10):raise ValueError('Raw checkpoint differs from reused power.')
            if not np.array_equal(a['retained_trial_numbers'],b['retained_trial_numbers']):raise ValueError('Raw trial selection mismatch.')
    qc=dict(run=key,no_cache_raw_path_passed=True,relative_errors=errors)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'raw_checkpoint_validation.json').write_text(json.dumps(qc,indent=2)+'\n')
    print(f'Cold-cache raw EEG computation verified against the reused checkpoint: {errors}.')
    return qc
