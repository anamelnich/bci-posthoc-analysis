"""Seven-point midfrontal theta trajectories: assessment, five interventions, assessment."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import time

import numpy as np
import pandas as pd

from .config import EXPECTED_SUBJECTS, get_subject_group
from .bci import load_decoding_analysis_file
from .triggers import load_training_trigger_file

CONDITIONS=('distractor','no_distractor','both')
ROI=('FZ','FCZ','CZ')


def inspect_intervention_theta_run(path, settings):
    """Validate complete paired task files, 60 aligned stimulus anchors and real padding."""
    import mne
    path=Path(path)
    if not path.name.endswith('_decoding.gdf') or 'practice' in path.name:
        raise ValueError(f'{path}: expected an experimental decoding recording.')
    # Check raw shape and integrality before the established range validators.
    for suffix,shape in [('.analysis.txt',(60,8)),('.triggers.txt',(180,3))]:
        values=np.loadtxt(path.with_suffix(suffix))
        if values.shape!=shape or not np.isfinite(values).all() or not np.equal(values,np.round(values)).all():
            raise ValueError(f'{path.with_suffix(suffix)}: expected finite integer data of shape {shape}.')
    analysis=load_decoding_analysis_file(path.with_suffix('.analysis.txt'))
    triggers=load_training_trigger_file(path.with_suffix('.triggers.txt'),allow_zero_latency_response=True)
    raw=mne.io.read_raw_gdf(path,preload=False,verbose='ERROR')
    try:
        labels=[c for c in raw.ch_names if c not in settings['omit_channels']]
        if (raw.info['sfreq']!=512 or len(raw.ch_names)!=67 or len(set(raw.ch_names))!=67
                or labels!=settings['surface_laplacian']['channels']):
            raise ValueError(f'{path}: expected the same 61 scalp electrodes in 67 recorded channels at 512 Hz.')
        events=mne.find_events(raw,stim_channel='Status',shortest_event=1,verbose='ERROR')
        stim=events[np.isin(events[:,2],[8,32,44])]
        expected=analysis.distractor_position.map({0:8,2:32,4:44}).to_numpy()
        if (stim.shape!=(60,3) or not np.array_equal(stim[:,2],expected)
                or not np.array_equal(triggers.loc[triggers.trigger.isin([8,32,44]),'trigger'],expected)
                or not analysis.task.eq(analysis.distractor_position.ne(0).astype(int)).all()
                or dict(zip(*np.unique(expected,return_counts=True)))!={8:30,32:15,44:15}):
            raise ValueError(f'{path}: 60-trial EEG/text/condition alignment failed.')
        fixations=events[events[:,2]==4,0]
        fix_index=np.searchsorted(fixations,stim[:,0],side='left')-1
        if (fix_index<0).any() or len(np.unique(fix_index))!=60:
            raise ValueError(f'{path}: every stimulus needs its own preceding fixation.')
        selected_fixation=fixations[fix_index]
        if (selected_fixation[1:]<=stim[:-1,0]).any():
            raise ValueError(f'{path}: fixation/stimulus trial ordering is invalid.')
        unmatched_fixations=len(fixations)-len(np.unique(fix_index))
        if unmatched_fixations:
            print(f'{path.name}: {unmatched_fixations} unmatched fixation marker(s); all 60 stimulus/text/analysis identities and preceding fixations validated.',flush=True)
        fixation=(stim[:,0]-selected_fixation)/512
        if fixation.min()<.9 or stim[0,0]-raw.first_samp<512 or stim[-1,0]-raw.first_samp+768>=raw.n_times:
            raise ValueError(f'{path}: insufficient fixation or real padding for the inherited epochs.')
        if (np.diff(stim[:,0])<=768).any():
            raise ValueError(f'{path}: next stimulus overlaps the padded epoch.')
        response_counts=[]
        for i,onset in enumerate(stim[:,0]):
            end=stim[i+1,0] if i<59 else raw.first_samp+raw.n_times
            response_counts.append(int(np.sum((events[:,2]==64)&(events[:,0]>onset)&(events[:,0]<end))))
        absent=sum(x==0 for x in response_counts);multiple=sum(x>1 for x in response_counts)
        if absent or multiple:print(f'{path.name}: {absent} missing / {multiple} multiple response anchors; all 60 valid stimulus-locked trials retained.',flush=True)
        audit=dict(n_trials=60,n_scalp_channels=61,n_status_events=len(events),n_unmatched_fixations=unmatched_fixations,
            n_missing_responses=absent,n_multiple_responses=multiple,
            fixation_min_s=float(fixation.min()),fixation_max_s=float(fixation.max()))
        analysis=analysis.assign(stimulus_sample=stim[:,0],stimulus_code=stim[:,2],retained=True)
        return analysis,audit
    finally:raw.close()


def prepare_intervention_theta_inputs(study_root, assessment_cache_dir, output_dir, missing_gdf_keys=(), roi=ROI):
    """Validate all intervention files; retain every trial and only explicit missing-file exceptions."""
    from .consolidated import validate_all_files_comprehensive
    study,assessment,out=map(Path,(study_root,assessment_cache_dir,output_dir));out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((assessment/'final_manifest.json').read_text())['files_sha256']
    for name in ['settings.json','participant_session_topographies.csv','mask_validation.json']:
        matches=[v for k,v in manifest.items() if Path(k).name==name]
        if matches!=[hashlib.sha256((assessment/name).read_bytes()).hexdigest()]:
            raise ValueError(f'Assessment input checksum mismatch: {name}')
    settings=json.loads((assessment/'settings.json').read_text());roi=tuple(roi)
    if (settings['baseline_s']!=[-.3,-.1] or settings['target_s']!=[.1,.5]
            or settings['frequencies_hz']!=np.arange(4,8.01,.5).tolist() or settings['sfreq']!=512
            or not settings.get('surface_laplacian') or settings['epoch_s']!=[-1.,1.5]
            or not roi or len(set(roi))!=len(roi) or not set(roi).issubset(settings['surface_laplacian']['channels'])):
        raise ValueError('Expected validated 4–8 Hz Laplacian assessment settings and recorded ROI channels.')
    settings.update(source_task='decoding',power_channels=list(roi),rt_cleaned=False,expected_excluded_trials=0,
        trial_selection='All 60 trials per available experimental intervention run; no RT/outcome/artifact exclusion',
        baseline_pool='All trials of both conditions pooled within participant/intervention session/channel/frequency')
    # This legacy utility covers assessment text files and has documented run-count
    # warnings for Sessions 2–4. Intervention integrity is checked independently below.
    legacy_ok=validate_all_files_comprehensive()
    missing_allowed={tuple(k) for k in missing_gdf_keys};missing_seen=set()
    records=[];trial_rows=[];session_rows=[];failures=[];practice=[]
    for subject in EXPECTED_SUBJECTS:
        sessions=sorted(p for p in (study/subject).glob(subject+'_*') if p.is_dir())
        if len(sessions)!=5:raise ValueError(f'{subject}: expected exactly five ordered session folders.')
        for session,folder in enumerate(sessions,1):
            runs=sorted(p for p in folder.glob('*_decoding') if p.is_dir())
            expected_runs=5 if (subject,session)==('e27',1) else 6 if session in (1,5) else 8
            if (subject,session) in {('e22',5),('e37',5)}:
                if len(runs)!=7:raise ValueError(f'{subject}/S5: documented practice-like exception expects seven folders.')
                practice.append(dict(subject_id=subject,session_id=session,source=str(runs[0]),reason='First extra folder is practice-like per analysis_outputs.md'))
                runs=runs[1:]
            if len(runs)!=expected_runs:raise ValueError(f'{subject}/S{session}: expected {expected_runs} experimental runs, found {len(runs)}.')
            available=0
            for run_id,run in enumerate(runs,1):
                path=run/(run.name+'.gdf');key=(subject,session,run_id)
                if not path.exists():
                    if key not in missing_allowed:raise FileNotFoundError(f'{path}: missing EEG; explicit missing-file policy required.')
                    missing_seen.add(key);print(f'{subject}/S{session}/R{run_id}: missing EEG documented; session uses remaining runs.',flush=True);continue
                try:
                    data,qc=inspect_intervention_theta_run(path,settings);stat=path.stat()
                    record=dict(subject_id=subject,group=get_subject_group(subject),session_id=session,run_id=run_id,
                        expected_n_runs=expected_runs,source=str(path.resolve()),source_size=stat.st_size,source_mtime_ns=stat.st_mtime_ns,
                        analysis_sha=hashlib.sha256(path.with_suffix('.analysis.txt').read_bytes()).hexdigest(),
                        triggers_sha=hashlib.sha256(path.with_suffix('.triggers.txt').read_bytes()).hexdigest(),**qc)
                    records.append(record);trial_rows.append(data.assign(subject_id=subject,group=get_subject_group(subject),session_id=session,run_id=run_id));available+=1
                except Exception as exc:failures.append(dict(source=str(path),error=str(exc)))
            session_rows.append(dict(subject_id=subject,group=get_subject_group(subject),session_id=session,expected_n_runs=expected_runs,n_runs=available,n_trials=60*available))
            print(f'{subject}/intervention {session}: {available}/{expected_runs} runs validated; all outcomes included.',flush=True)
    (out/'source_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    if failures:raise ValueError(f'{len(failures)} invalid intervention files; inspect source_failures.json before computing.')
    if missing_seen!=missing_allowed:raise ValueError('Requested missing-file exceptions do not match observed missing files.')
    inventory=pd.DataFrame(records);ledger=pd.concat(trial_rows,ignore_index=True);counts=pd.DataFrame(session_rows)
    if (len(counts)!=160 or counts.n_runs.le(0).any() or len(ledger)!=60*len(inventory)
            or inventory.duplicated(['subject_id','session_id','run_id']).any()
            or ledger.duplicated(['subject_id','session_id','run_id','trial_index']).any() or not ledger.retained.all()):
        raise ValueError('Incomplete participant/session or inconsistent trial inventory.')
    inventory.to_csv(out/'source_inventory.csv',index=False);ledger.to_csv(out/'trial_ledger.csv',index=False);counts.to_csv(out/'session_run_counts.csv',index=False)
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    audit=dict(n_subjects=32,n_sessions=160,n_runs=len(inventory),n_trials=len(ledger),n_excluded_trials=0,
        missing_gdf_keys=sorted(missing_seen),practice_like_folders=practice,
        e27_session1='All five experimental-labeled runs retained per documented exception',
        legacy_assessment_validator_ok=legacy_ok,
        legacy_note='Legacy assessment validator has documented Sessions 2–4 and e30 run-count warnings; every intervention file independently validated.',
        assessment_cache=str(assessment.resolve()),assessment_settings_sha256=hashlib.sha256((assessment/'settings.json').read_bytes()).hexdigest(),
        preflight_hashes={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['settings.json','trial_ledger.csv','source_inventory.csv','session_run_counts.csv']})
    (out/'input_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(f'Intervention preflight complete: {len(inventory)} runs, {len(ledger):,} trials, 160 participant/sessions; ROI {roi}.',flush=True)
    return inventory,ledger,settings


def _cache_intervention_theta_run(record,settings,output_dir):
    """Validate sources and checkpoint all-trial ROI power with a code/settings signature."""
    from .theta_topography import extract_theta_run
    from .theta_laplacian import apply_theta_laplacian,_laplacian_matrix
    source=Path(record['source']);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if (source.stat().st_size!=record['source_size'] or source.stat().st_mtime_ns!=record['source_mtime_ns']
            or any(hashlib.sha256(source.with_suffix(suffix).read_bytes()).hexdigest()!=record[key]
                   for suffix,key in [('.analysis.txt','analysis_sha'),('.triggers.txt','triggers_sha')])):
        raise ValueError(f'{source}: input changed after validation.')
    signature=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()+inspect.getsource(extract_theta_run).encode()
        +inspect.getsource(apply_theta_laplacian).encode()+inspect.getsource(_laplacian_matrix).encode()).hexdigest()
    key=f"{record['subject_id']}_s{record['session_id']}_r{record['run_id']:02d}";npz=out/f'{key}.npz';meta_path=npz.with_suffix('.json')
    if meta_path.exists() and npz.exists():
        old=json.loads(meta_path.read_text())
        if (old.get('signature')==signature and old.get('npz_sha256')==hashlib.sha256(npz.read_bytes()).hexdigest()
                and all(old.get(k)==record[k] for k in ['source','source_size','source_mtime_ns','analysis_sha','triggers_sha'])):
            print(f'{key}: validated intervention cache reused.',flush=True);return old
    start=time.monotonic();arrays,qc=extract_theta_run(source,settings)
    if (not np.array_equal(arrays['condition_counts'],[30,30]) or qc['n_excluded']!=0
            or arrays['channel_labels'].tolist()!=settings['power_channels']):
        raise ValueError(f'{key}: ROI or all-trial extraction mismatch.')
    arrays['retained_trial_numbers']=np.arange(1,61)
    temp=out/f'{key}.tmp.npz';np.savez_compressed(temp,**arrays);temp.replace(npz)
    meta={**record,**qc,'signature':signature,'n_retained':60,'npz_path':str(npz.resolve()),
          'npz_sha256':hashlib.sha256(npz.read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    print(f'{key}: intervention ROI cached, 60 trials, {meta["elapsed_s"]:.1f}s.',flush=True)
    return meta


def build_intervention_theta_cache(inventory,ledger,settings,output_dir,n_jobs=2):
    """Compute all validated intervention trials using bounded-memory resumable caches."""
    out=Path(output_dir);audit=json.loads((out/'input_validation.json').read_text())
    for name,digest in audit['preflight_hashes'].items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed preflight input: {name}')
    if n_jobs not in (1,2) or len(ledger)!=60*len(inventory) or not ledger.retained.all() or json.loads((out/'settings.json').read_text())!=settings:
        raise ValueError('Invalid all-trial intervention inputs or worker count.')
    pd.testing.assert_frame_equal(inventory,pd.read_csv(out/'source_inventory.csv'),check_dtype=False)
    pd.testing.assert_frame_equal(ledger,pd.read_csv(out/'trial_ledger.csv'),check_dtype=False)
    logs=[]
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        for row in pool.map(lambda record:_cache_intervention_theta_run(record,settings,out/'runs'),inventory.to_dict('records')):
            logs.append(row);pd.DataFrame(logs).to_csv(out/'run_log_progress.csv',index=False)
    log=pd.DataFrame(logs)
    if len(log)!=len(inventory) or not log.n_retained.eq(60).all() or log.n_excluded.sum()!=0:
        raise ValueError('Incomplete intervention cache or unexpected exclusions.')
    log.to_csv(out/'run_log.csv',index=False)
    print(f'Intervention extraction complete: {len(log)} runs; {log.n_retained.sum():,} trials; zero trial exclusions.',flush=True)
    return log


def summarize_intervention_theta_trajectory(cache_dir,assessment_cache_dir):
    """Pool trial power per session, normalize to each session baseline, and append existing endpoints."""
    root,assessment=map(Path,(cache_dir,assessment_cache_dir));settings=json.loads((root/'settings.json').read_text())
    roi=settings['power_channels'];log=pd.read_csv(root/'run_log.csv');inventory=pd.read_csv(root/'source_inventory.csv')
    audit=json.loads((root/'input_validation.json').read_text());counts=pd.read_csv(root/'session_run_counts.csv')
    for name,digest in audit['preflight_hashes'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed preflight input: {name}')
    if len(log)!=len(inventory) or log.duplicated(['subject_id','session_id','run_id']).any():raise ValueError('Incomplete run cache.')
    pd.testing.assert_frame_equal(log[['subject_id','session_id','run_id']],inventory[['subject_id','session_id','run_id']])
    rows=[]
    for (subject,group,session),cell in log.groupby(['subject_id','group','session_id']):
        bs=np.zeros((len(roi),9));ps=np.zeros((2,len(roi),9,205));n=np.zeros(2,dtype=int)
        for row in cell.itertuples():
            path=root/'runs'/Path(row.npz_path).name  # Resolve copied checkpoints locally; hash checked below.
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row.npz_sha256:raise ValueError(f'{path}: corrupted power checkpoint.')
            with np.load(path,allow_pickle=False) as z:
                b,p,nn=z['baseline_sum'],z['post_sum'],z['condition_counts']
                if (b.shape!=(len(roi),9) or p.shape!=(2,len(roi),9,205)
                        or z['channel_labels'].tolist()!=roi or not np.array_equal(z['frequencies'],settings['frequencies_hz'])
                        or not np.array_equal(z['post_times'],np.arange(52,257)/512)
                        or not np.array_equal(z['retained_trial_numbers'],np.arange(1,61))
                        or not np.array_equal(nn,[30,30]) or int(z['n_baseline_samples'])!=102
                        or not np.isfinite(b).all() or not np.isfinite(p).all() or (b<=0).any() or (p<=0).any()):
                    raise ValueError(f'{path}: invalid ROI power, axes, or trial counts.')
                bs+=b;ps+=p;n+=nn
        baseline=bs/n.sum()
        for condition,power,num in [('no_distractor',ps[0]/n[0],n[0]),('distractor',ps[1]/n[1],n[1]),('both',ps.sum(axis=0)/n.sum(),n.sum())]:
            value=float((10*np.log10(power/baseline[:,:,None])).mean())
            rows.append(dict(subject_id=subject,group=group,phase='intervention',session_id=session,session_index=session,
                session_label=f'Training {session}',condition=condition,theta_db=value,n_trials=int(num),n_runs=len(cell)))
    manifest=json.loads((assessment/'final_manifest.json').read_text())['files_sha256']
    table_path=assessment/'participant_session_topographies.csv'
    if [v for k,v in manifest.items() if Path(k).name==table_path.name]!=[hashlib.sha256(table_path.read_bytes()).hexdigest()]:
        raise ValueError('Pre/post assessment values changed or failed checksum validation.')
    reference=pd.read_csv(table_path)
    part=reference[reference.channel.isin(roi)]
    if len(part)!=32*2*3*len(roi) or not part.groupby(['subject_id','session_id','condition']).size().eq(len(roi)).all():
        raise ValueError('Missing assessment ROI/channel/condition cells.')
    for (subject,group,session,condition),cell in part.groupby(['subject_id','group','session_id','condition']):
        rows.append(dict(subject_id=subject,group=group,phase='assessment',session_id=session,session_index=0 if session==1 else 6,
            session_label='Pre' if session==1 else 'Post',condition=condition,theta_db=float(cell.theta_db.mean()),
            n_trials=int(cell.n_trials.iloc[0]),n_runs=7 if (subject,session)==('e30',1) else 8 if session==1 else 4))
    result=pd.DataFrame(rows).sort_values(['subject_id','condition','session_index']).reset_index(drop=True)
    if (len(result)!=32*7*3 or result.duplicated(['subject_id','condition','session_index']).any()
            or not np.isfinite(result.theta_db).all() or not result.groupby(['subject_id','condition']).session_index.apply(lambda x:set(x)==set(range(7))).all()
            or set(result.subject_id)!=set(EXPECTED_SUBJECTS)
            or not result.group.eq(result.subject_id.map(get_subject_group)).all()):raise ValueError('Incomplete seven-session trajectory.')
    result.to_csv(root/'participant_trajectories.csv',index=False)
    qc=dict(n_participants=32,n_per_group=16,n_points=7,n_conditions=3,n_rows=len(result),roi=roi,
        intervention_trials=int(log.n_retained.sum()),intervention_runs=len(log),intervention_exclusions=0,
        baseline_s=settings['baseline_s'],target_s=settings['target_s'],
        normalization='Session-specific common baseline across both conditions/runs. Trial mean linear power, then 10log10, then mean over times/frequencies/ROI electrodes. All-trial condition pools linear power.',
        endpoints='Existing behavior-screened distractor assessment values reused exactly; intervention retains all outcomes.',
        missing_gdf_keys=audit['missing_gdf_keys'],statistics_run=False,
        participant_sha256=hashlib.sha256((root/'participant_trajectories.csv').read_bytes()).hexdigest())
    (root/'summary_validation.json').write_text(json.dumps(qc,indent=2)+'\n')
    print(f'Validated 672 participant/condition/time values: 32 participants × 7 sessions × 3 conditions; endpoints reused.',flush=True)
    return result,qc


def plot_intervention_theta_trajectory(participants,output_dir):
    """Plot three seven-session trajectories, BCI/control means and between-subject SEM."""
    import matplotlib.pyplot as plt
    required={'subject_id','group','session_index','condition','theta_db','phase'}
    if (not required.issubset(participants) or len(participants)!=672
            or participants.duplicated(['subject_id','session_index','condition']).any()
            or not np.isfinite(participants.theta_db).all() or set(participants.condition)!=set(CONDITIONS)
            or set(participants.subject_id)!=set(EXPECTED_SUBJECTS)
            or not participants.group.eq(participants.subject_id.map(get_subject_group)).all()):
        raise ValueError('Expected complete finite 32-person, seven-session, three-condition trajectories.')
    summary=participants.groupby(['group','condition','session_index']).theta_db.agg(mean='mean',sd='std',n='count').reset_index()
    if len(summary)!=42 or not summary.n.eq(16).all() or set(summary.session_index)!=set(range(7)):
        raise ValueError('Expected 16 participants in every group/condition/session.')
    summary['sem']=summary.sd/np.sqrt(summary.n)
    low=min(0.,float((summary['mean']-summary['sem']).min()));high=max(0.,float((summary['mean']+summary['sem']).max()))
    pad=max(.12,(high-low)*.12);limits=(low-pad,high+pad)
    colors={'bci':'#DD8452','control':'#4C72B0'};labels={'bci':'BCI','control':'Mental rehearsal'}
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);stem=out/'midfrontal_theta_seven_sessions'
    rc={'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':7,
        'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(rc):
        fig,axes=plt.subplots(1,3,figsize=(7.05,3.15),sharey=True)
        fig.subplots_adjust(left=.10,right=.985,bottom=.27,top=.77,wspace=.18)
        for panel,(ax,condition,title) in enumerate(zip(axes,CONDITIONS,['Distractor','No distractor','All trials'])):
            for group in ['bci','control']:
                cell=summary[(summary.group==group)&(summary.condition==condition)].sort_values('session_index')
                ax.errorbar(cell.session_index,cell['mean'],yerr=cell['sem'],color=colors[group],marker='o',markersize=3,
                    linewidth=1.1,elinewidth=.7,capsize=2,capthick=.7,label=labels[group],zorder=3)
            ax.axhline(0,color='.7',linewidth=.5,zorder=0)
            for x in [.5,5.5]:ax.axvline(x,color='.85',linewidth=.5,linestyle='--',zorder=0)
            ax.set(xlim=(-.35,6.35),ylim=limits,xticks=range(7),xticklabels=['Pre','1','2','3','4','5','Post'],xlabel='Session',title=title)
            ax.text(-.08,1.09,chr(97+panel),transform=ax.transAxes,fontweight='bold',fontsize=9)
            ax.spines[['top','right']].set_visible(False);ax.tick_params(width=.6,length=3,labelsize=6)
        axes[0].set_ylabel('Normalized theta power (dB)')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.53,.99),fontsize=7)
        fig.text(.54,.07,'Fz, FCz, Cz | 4–8 Hz | 100–500 ms | Mean ± SEM; n = 16 per group',ha='center',fontsize=6.5)
        paths={ext:str(stem.with_suffix('.'+ext).resolve()) for ext in ['pdf','svg','png']}
        for ext,path in paths.items():fig.savefig(path,dpi=350,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    summary.to_csv(stem.with_suffix('.csv'),index=False)
    meta=dict(paths=paths,n_subjects=32,n_per_group=16,n_sessions=7,conditions=list(CONDITIONS),colors=colors,
        uncertainty='Between-subject sample SD / sqrt(16); equal participant weights',y_limits_db=list(limits),
        figure_sha256={ext:hashlib.sha256(Path(path).read_bytes()).hexdigest() for ext,path in paths.items()},
        statistics_run=False,endpoint_policy='Pre/post behavior-screened; intervention all trials; dashed separators indicate phase boundaries.')
    stem.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Saved three-panel seven-session theta figure (PDF/SVG/PNG): BCI orange, mental rehearsal blue; 16/group; mean ± SEM.',flush=True)
    return meta,summary
