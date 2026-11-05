"""Intervention RT trajectories with participant/session trimming and >50% loss rule."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .config import EXPECTED_SUBJECTS, get_subject_group

CONDITIONS=('both','distractor','no_distractor')


def load_intervention_rt_trials(theta_cache_dir, rt_cache_dir, output_dir):
    """Join verified raw GDF RTs to the intervention inventory by exact path/trial.

    All 60 trial positions are eligible, without previous-feedback selection.
    Entire missing-EEG runs are technical losses outside the behavioral-loss
    denominator; individual missing/ambiguous response RTs remain NaN.
    """
    root,rtroot,out=map(Path,(theta_cache_dir,rt_cache_dir,output_dir));out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'final_manifest.json').read_text())['files_sha256'];hashes={}
    for name in ['trial_ledger.csv','source_inventory.csv','input_validation.json']:
        digest=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if [v for k,v in manifest.items() if Path(k).name==name]!=[digest]:raise ValueError(f'{name}: checksum failed.')
        hashes[name]=digest
    meta=json.loads((rtroot/'reconstruction_provenance.json').read_text())
    if meta['version']!=1:raise ValueError('Unexpected RT reconstruction version.')
    for name in ['reconstruction_trials','reconstruction_log']:
        digest=hashlib.sha256((rtroot/f'{name}.csv').read_bytes()).hexdigest()
        if digest!=meta['tables'][name]:raise ValueError(f'{name}: RT cache checksum failed.')
        hashes[name+'.csv']=digest
    inventory=pd.read_csv(root/'source_inventory.csv');ledger=pd.read_csv(root/'trial_ledger.csv')
    sources={v['path']:v['signature'] for v in meta['sources']}
    for row in inventory.itertuples():
        for path in [Path(row.source),Path(row.source).with_suffix('.analysis.txt')]:
            stat=path.stat()
            if sources.get(str(path))!=[stat.st_size,stat.st_mtime_ns]:raise ValueError(f'{path}: source metadata changed since RT reconstruction.')
        if hashlib.sha256(Path(row.source).with_suffix('.analysis.txt').read_bytes()).hexdigest()!=row.analysis_sha:
            raise ValueError(f'{row.source}: behavioral source changed.')
    rt=pd.read_csv(rtroot/'reconstruction_trials.csv')
    required={'gdf_path','trial_id','subject_id','session_id','stimulus_sample','stimulus_trigger','response_sample','response_count','response_stop_sample','rt_ms','rt_reason','sfreq_hz','rt_source'}
    if not required.issubset(rt) or rt.duplicated(['gdf_path','trial_id']).any():raise ValueError('Missing/duplicate raw RT fields.')
    rt=rt[rt.gdf_path.isin(inventory.source)].copy()
    if len(rt)!=len(inventory)*60 or not rt.groupby('gdf_path').trial_id.apply(lambda x:sorted(x)==list(range(1,61))).all():
        raise ValueError('Expected all 60 cached RT trial positions per intervention GDF.')
    if not rt.sfreq_hz.eq(512).all() or not rt.rt_source.eq('gdf_status').all():raise ValueError('RTs must be reconstructed from 512 Hz GDF Status, without text fallback.')
    valid=rt.rt_reason.eq('ok');calc=(rt.response_sample-rt.stimulus_sample)/rt.sfreq_hz*1000
    if (not np.allclose(rt.loc[valid,'rt_ms'],calc[valid],rtol=0,atol=1e-10)
            or not rt.loc[valid,'response_count'].eq(1).all()
            or not rt.loc[valid,'response_sample'].gt(rt.loc[valid,'stimulus_sample']).all()
            or not rt.loc[valid,'response_sample'].lt(rt.loc[valid,'response_stop_sample']).all()
            or not rt.loc[~valid,'rt_ms'].isna().all()):raise ValueError('Invalid RT timing or invalid-value propagation.')
    keys=['subject_id','session_id','run_id']
    data=ledger.merge(inventory[keys+['source']],on=keys,validate='many_to_one')
    data=data.merge(rt.rename(columns={'trial_id':'trial_index','subject_id':'rt_subject','session_id':'rt_session','stimulus_sample':'rt_stimulus_sample'}),
        left_on=['source','trial_index'],right_on=['gdf_path','trial_index'],how='left',validate='one_to_one',indicator=True)
    if (not data._merge.eq('both').all() or not data.subject_id.eq(data.rt_subject).all()
            or not data.session_id.eq(data.rt_session).all() or not data.stimulus_sample.eq(data.rt_stimulus_sample).all()
            or not data.stimulus_code.eq(data.stimulus_trigger).all()
            or set(data.subject_id)!=set(EXPECTED_SUBJECTS) or not data.group.eq(data.subject_id.map(get_subject_group)).all()
            or not data.feedback.isin([1,2,3]).all() or not data.task.isin([0,1]).all()
            or not data.retained.eq(True).all()):raise ValueError('Exact-path EEG/behavior alignment failed.')
    columns=keys+['group','trial_index','task','feedback','rt_ms','rt_reason','source']
    data=data[columns].sort_values(keys+['trial_index']).reset_index(drop=True)
    audit=json.loads((root/'input_validation.json').read_text())
    qc=dict(n_trials=len(data),n_runs=len(inventory),n_subjects=32,n_sessions=160,source_cache=str(rtroot.resolve()),
        input_hashes=hashes,rt_source='GDF Status; response strictly after stimulus and before next fixation; no text fallback',
        first_trials_included=True,previous_feedback_filter=False,missing_GDF_run_keys=audit['missing_gdf_keys'],
        missing_GDF_policy='Entire missing-EEG run is a technical loss, outside the >50% behavioral/RT-loss denominator.',
        inventory_policy='Same experimental inventory as the intervention theta/response trajectory, including all five documented e27 Session 1 runs. Exact raw-path matching removes reliance on consolidated BCI run offsets.',
        missing_or_ambiguous_trial_RTs=int(data.rt_ms.isna().sum()))
    (out/'input_provenance.json').write_text(json.dumps(qc,indent=2)+'\n')
    print(f'Loaded {len(data):,} intervention trials from {len(inventory):,} runs; exact raw-path/stimulus/condition alignment passed; all trial positions included.')
    print(f'{data.rt_ms.isna().sum()} missing/ambiguous trial RTs retained as missing; missing-EEG runs are technical losses: {audit["missing_gdf_keys"]}.')
    return data,qc


def summarize_clean_intervention_rt(trials,output_dir,max_subject_loss_percent=50):
    """Sequential trial exclusions, single-pass participant/session 3 SD, subject loss rule."""
    required={'subject_id','group','session_id','run_id','trial_index','task','feedback','rt_ms'}
    if (not required.issubset(trials) or trials.duplicated(['subject_id','session_id','run_id','trial_index']).any()
            or not trials.feedback.isin([1,2,3]).all() or not trials.task.isin([0,1]).all()
            or set(trials.subject_id)!=set(EXPECTED_SUBJECTS) or not trials.group.eq(trials.subject_id.map(get_subject_group)).all()
            or set(trials.session_id)!=set(range(1,6)) or not 0<=max_subject_loss_percent<=100):raise ValueError('Invalid intervention trials or participant-loss threshold.')
    data=trials.copy();data['exclusion_reason']='retained'
    data.loc[data.feedback.eq(2),'exclusion_reason']='incorrect'
    data.loc[data.feedback.eq(3),'exclusion_reason']='timeout'
    ok=data.exclusion_reason.eq('retained')
    data.loc[ok & (~np.isfinite(data.rt_ms)|data.rt_ms.le(0)),'exclusion_reason']='missing_invalid_rt'
    data.loc[data.exclusion_reason.eq('retained') & data.rt_ms.lt(150),'exclusion_reason']='below_150_ms'
    keys=['subject_id','session_id']
    bounds=data[data.exclusion_reason.eq('retained')].groupby(keys).rt_ms.agg(n='count',mean='mean',sd='std').reset_index()
    bounds=data[keys].drop_duplicates().merge(bounds,on=keys,how='left',validate='one_to_one')
    bounds['n']=bounds.n.fillna(0).astype(int);bounds['lower_ms']=bounds['mean']-3*bounds.sd;bounds['upper_ms']=bounds['mean']+3*bounds.sd
    bounds['applied']=bounds.n.ge(2)&bounds.sd.gt(0)
    data=data.merge(bounds,on=keys,validate='many_to_one')
    outlier=data.exclusion_reason.eq('retained')&data.applied&(data.rt_ms.lt(data.lower_ms)|data.rt_ms.gt(data.upper_ms))
    data.loc[outlier,'exclusion_reason']='outlier_3sd'
    data['trial_retained']=data.exclusion_reason.eq('retained')
    subjects=data.groupby(['subject_id','group']).agg(candidate_trials=('trial_retained','size'),retained_trials=('trial_retained','sum')).reset_index()
    subjects['excluded_trials']=subjects.candidate_trials-subjects.retained_trials
    subjects['loss_percent']=100*subjects.excluded_trials/subjects.candidate_trials
    subjects['subject_excluded']=subjects.loss_percent.gt(max_subject_loss_percent)
    data=data.merge(subjects[['subject_id','subject_excluded']],on='subject_id',validate='many_to_one')
    data['final_retained']=data.trial_retained&~data.subject_excluded
    retained=data[data.final_retained];rows=[]
    for (subject,group,session),cell in data[~data.subject_excluded].groupby(['subject_id','group','session_id']):
        cell=cell[cell.trial_retained]
        for condition in CONDITIONS:
            sub=cell if condition=='both' else cell[cell.task.eq(1 if condition=='distractor' else 0)]
            rows.append(dict(subject_id=subject,group=group,session_id=session,condition=condition,n_trials=len(sub),
                mean_rt_ms=float(sub.rt_ms.mean()) if len(sub) else np.nan))
    participants=pd.DataFrame(rows)
    if participants.mean_rt_ms.isna().any():
        print('Empty participant/session/condition RT cells (remain missing):',participants.loc[participants.mean_rt_ms.isna()].to_dict('records'))
    summary=participants.groupby(['group','condition','session_id']).mean_rt_ms.agg(mean_rt_ms='mean',sd_rt_ms='std',n='count').reset_index()
    summary['sem_rt_ms']=summary.sd_rt_ms/np.sqrt(summary.n)
    if len(summary)!=30 or summary.n.lt(2).any():raise ValueError('Insufficient data for five-session group means/SEMs.')
    # Independently verify every threshold and retained trial, pooling types/runs.
    max_error=0.
    for key,cell in trials.groupby(keys):
        values=cell.loc[cell.feedback.eq(1)&np.isfinite(cell.rt_ms)&cell.rt_ms.ge(150),'rt_ms'].to_numpy()
        row=bounds[(bounds.subject_id==key[0])&(bounds.session_id==key[1])].iloc[0]
        if len(values)>=2 and values.std(ddof=1)>0:
            lo,hi=values.mean()-3*values.std(ddof=1),values.mean()+3*values.std(ddof=1)
            max_error=max(max_error,abs(lo-row.lower_ms),abs(hi-row.upper_ms))
            expected=cell.feedback.eq(1)&np.isfinite(cell.rt_ms)&cell.rt_ms.ge(150)&cell.rt_ms.ge(lo)&cell.rt_ms.le(hi)
        else:expected=cell.feedback.eq(1)&np.isfinite(cell.rt_ms)&cell.rt_ms.ge(150)
        actual=data[(data.subject_id==key[0])&(data.session_id==key[1])].set_index(['run_id','trial_index']).trial_retained.sort_index()
        expected.index=pd.MultiIndex.from_frame(cell[['run_id','trial_index']]);pd.testing.assert_series_equal(actual,expected.sort_index(),check_names=False)
    if max_error>1e-9 or retained.empty:raise ValueError('Independent threshold validation failed.')
    exclusions=data.groupby(['group','exclusion_reason']).size().rename('n_trials').reset_index()
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    for name,table in [('trial_exclusions',data),('outlier_bounds',bounds),('subject_exclusions',subjects),('exclusion_summary',exclusions),('participant_session_rt',participants),('group_session_rt',summary)]:table.to_csv(out/f'{name}.csv',index=False)
    qc=dict(n_candidate_trials=len(data),n_retained_before_subject_exclusion=int(data.trial_retained.sum()),n_final_trials=len(retained),
        n_included_by_group=subjects[~subjects.subject_excluded].group.value_counts().to_dict(),
        excluded_subjects=subjects.loc[subjects.subject_excluded,'subject_id'].tolist(),
        empty_retained_participant_cells=participants.loc[participants.mean_rt_ms.isna(),['subject_id','group','session_id','condition','n_trials']].to_dict('records'),
        missing_cell_policy='Do not impute empty cells or exclude the participant from other sessions; summarize available participant means and report actual n per group/session/condition.',
        exclusion_counts=data.exclusion_reason.value_counts().to_dict(),
        cleaning='Incorrect, timeout, missing/nonpositive/nonfinite RT, RT <150 ms, then one-pass mean ±3 sample SD among remaining trials within participant × session, pooled across conditions/runs.',
        subject_rule=f'Exclude >{max_subject_loss_percent}% loss across all five sessions after trial cleaning; exactly {max_subject_loss_percent}% retained. Technical missing runs excluded from denominator.',
        maximum_independent_bound_error_ms=float(max_error),sample_sd_ddof=1,
        group_summary='Equal participant means ± between-participant SEM; pooled condition uses all retained trials before participant averaging.',statistics_run=False)
    (out/'cleaning_validation.json').write_text(json.dumps(qc,indent=2)+'\n')
    print('Trial exclusions:',qc['exclusion_counts']);print('Excluded subjects:',qc['excluded_subjects'])
    print(f'Final retained trials: {len(retained):,}; participants by group: {qc["n_included_by_group"]}. One-pass participant/session thresholds independently verified.')
    return dict(participants=participants,summary=summary,subjects=subjects,validation=qc)


def plot_intervention_rt_trajectory(participants,summary,output_dir):
    """Plot pooled and condition-specific participant-mean RT trajectories, mean ± SEM."""
    import matplotlib.pyplot as plt
    required={'group','condition','session_id','mean_rt_ms','sem_rt_ms','n'}
    if (not required.issubset(summary) or len(summary)!=30 or summary.duplicated(['group','condition','session_id']).any()
            or not np.isfinite(summary[['mean_rt_ms','sem_rt_ms']]).all().all() or summary.n.lt(2).any()):raise ValueError('Incomplete valid RT plotting summary.')
    check=participants.groupby(['group','condition','session_id']).mean_rt_ms.agg(['mean','std','count']).reset_index()
    merged=summary.merge(check,on=['group','condition','session_id'],validate='one_to_one')
    if not np.allclose(merged.mean_rt_ms,merged['mean']) or not np.allclose(merged.sem_rt_ms,merged['std']/np.sqrt(merged['count'])):raise ValueError('Plot summaries differ from participant RTs.')
    low=max(0,50*np.floor((summary.mean_rt_ms-summary.sem_rt_ms).min()/50)-50)
    high=50*np.ceil((summary.mean_rt_ms+summary.sem_rt_ms).max()/50)+50
    sizes=summary.groupby('group').n.agg(['min','max']).to_dict('index')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);stem=out/'intervention_rt_five_sessions'
    colors={'bci':'#DD8452','control':'#4C72B0'};labels={'bci':'BCI','control':'Mental rehearsal'}
    sample_parts=[]
    for group in ['bci','control']:
        counts=summary[summary.group.eq(group)&summary.condition.eq('both')].sort_values('session_id').n.tolist()
        sample_parts.append(labels[group]+' n = '+(str(counts[0])+' throughout' if len(set(counts))==1 else ', '.join(map(str,counts))+' (sessions 1–5)'))
    sample_caption='Mean ± SEM; '+'; '.join(sample_parts)+'.'
    with plt.rc_context({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':7,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,3,figsize=(7.05,3.2),sharey=True)
        fig.subplots_adjust(left=.09,right=.985,bottom=.28,top=.77,wspace=.18)
        for index,(ax,condition,title) in enumerate(zip(axes,CONDITIONS,['All trials','Distractor','No distractor'])):
            for group in ['bci','control']:
                cell=summary[summary.group.eq(group)&summary.condition.eq(condition)].sort_values('session_id')
                if cell.session_id.tolist()!=[1,2,3,4,5]:raise ValueError('Missing plotted session.')
                ax.errorbar(cell.session_id,cell.mean_rt_ms,yerr=cell.sem_rt_ms,color=colors[group],marker='o',markersize=3,
                    linewidth=1.1,elinewidth=.7,capsize=2,capthick=.7,label=labels[group])
            ax.set(xlim=(.8,5.2),ylim=(low,high),xticks=range(1,6),xlabel='Intervention session',title=title)
            ax.spines[['top','right']].set_visible(False);ax.tick_params(width=.6,length=3,labelsize=6.5)
            ax.text(-.10,1.10,chr(97+index),transform=ax.transAxes,fontweight='bold',fontsize=9)
        axes[0].set_ylabel('Reaction time (ms)')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.54,.99),fontsize=7)
        fig.text(.54,.10,'Correct, valid RT ≥150 ms; within participant/session mean ±3 SD; participants with >50% trial loss excluded.',ha='center',fontsize=6)
        fig.text(.54,.055,sample_caption,ha='center',fontsize=6.5)
        paths={ext:str(stem.with_suffix('.'+ext)) for ext in ['pdf','svg','png']}
        for path in paths.values():fig.savefig(path,dpi=350,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    meta=dict(paths=paths,colors=colors,y_limits_ms=[low,high],n_by_group=sizes,statistics_run=False,
        figure_sha256={ext:hashlib.sha256(Path(path).read_bytes()).hexdigest() for ext,path in paths.items()})
    stem.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Saved pooled/distractor/no-distractor RT figure, PDF/SVG/PNG, common millisecond axis; BCI orange/control blue.')
    return meta
