"""Correct, incorrect and timeout response percentages over intervention sessions."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .config import EXPECTED_SUBJECTS, get_subject_group
from .bci import DECODING_ANALYSIS_COLUMNS, load_decoding_analysis_file, validate_decoding_analysis
from .triggers import load_training_trigger_file

OUTCOMES = {1:'correct', 2:'incorrect', 3:'timeout'}
CONDITIONS = ('both','distractor','no_distractor')


def prepare_intervention_response_rates(theta_cache_dir, output_dir):
    """Reuse validated behavioral rows; restore behavior-only runs omitted for missing EEG.

    Keep the established experimental/practice inventory, all response outcomes
    and all RTs. Denominator is all trials within participant/session/condition.
    Group percentages are equally weighted participant rates, not pooled counts.
    """
    root=Path(theta_cache_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'final_manifest.json').read_text())['files_sha256'];hashes={}
    for name in ['trial_ledger.csv','source_inventory.csv','session_run_counts.csv','input_validation.json']:
        digest=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if [v for k,v in manifest.items() if Path(k).name==name]!=[digest]:
            raise ValueError(f'{name}: validated intervention cache checksum mismatch.')
        hashes[name]=digest
    audit=json.loads((root/'input_validation.json').read_text())
    ledger=pd.read_csv(root/'trial_ledger.csv');inventory=pd.read_csv(root/'source_inventory.csv');counts=pd.read_csv(root/'session_run_counts.csv')
    keys=['subject_id','group','session_id','run_id'];cols=keys+list(DECODING_ANALYSIS_COLUMNS)
    if not set(cols+['retained']).issubset(ledger) or not ledger.retained.eq(True).all():
        raise ValueError('Missing behavioral ledger fields or previously excluded trials.')
    trials=ledger[cols].copy();trials['eeg_available']=True
    source_hashes={}
    for row in inventory.itertuples():
        path=Path(row.source).with_suffix('.analysis.txt');digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=row.analysis_sha:raise ValueError(f'{path}: analysis file changed since validation.')
        source_hashes[str(path)]=digest
    additions=[]
    practice_paths={p['source'] for p in audit['practice_like_folders']}
    for subject,session,run_id in audit['missing_gdf_keys']:
        source=inventory[inventory.subject_id.eq(subject)&inventory.session_id.eq(session)]
        if source.empty:raise ValueError(f'{subject}/S{session}: no sibling run to locate missing-EEG behavior.')
        folder=Path(source.source.iloc[0]).parent.parent
        runs=[p for p in sorted(folder.glob('*_decoding')) if p.is_dir() and str(p) not in practice_paths]
        expected=counts[counts.subject_id.eq(subject)&counts.session_id.eq(session)]
        if len(expected)!=1 or len(runs)!=int(expected.expected_n_runs.iloc[0]):
            raise ValueError(f'{subject}/S{session}: behavioral run inventory does not match documented expected count.')
        run=runs[run_id-1];path=run/(run.name+'.analysis.txt');trigger=run/(run.name+'.triggers.txt')
        if (run/(run.name+'.gdf')).exists():raise ValueError('Previously missing EEG now exists; refresh theta inventory before reconciling.')
        for file,shape in [(path,(60,8)),(trigger,(180,3))]:
            values=np.loadtxt(file)
            if values.shape!=shape or not np.isfinite(values).all() or not np.equal(values,np.round(values)).all():
                raise ValueError(f'{file}: expected finite integer shape {shape}.')
        behavior=load_decoding_analysis_file(path)
        triggers=load_training_trigger_file(trigger,allow_zero_latency_response=True)
        expected_stimuli=behavior.distractor_position.map({0:8,2:32,4:44}).to_numpy()
        if not np.array_equal(triggers.loc[triggers.trigger.isin([8,32,44]),'trigger'],expected_stimuli):
            raise ValueError(f'{path}: behavioral/trigger condition order mismatch.')
        addition=behavior.assign(subject_id=subject,group=get_subject_group(subject),session_id=session,run_id=run_id,eeg_available=False)
        trials=pd.concat([trials,addition[cols+['eeg_available']]],ignore_index=True)
        source_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        source_hashes[str(trigger)]=hashlib.sha256(trigger.read_bytes()).hexdigest()
        additions.append(dict(subject_id=subject,session_id=session,run_id=run_id,n_trials=len(addition),source=str(path)))
    if (trials[cols].isna().any().any() or trials.duplicated(['subject_id','session_id','run_id','trial_index']).any()
            or set(trials.subject_id)!=set(EXPECTED_SUBJECTS)
            or not trials.group.eq(trials.subject_id.map(get_subject_group)).all()
            or set(trials.session_id)!=set(range(1,6))):
        raise ValueError('Missing values, duplicate trials, incorrect groups or sessions in behavioral data.')
    for key,cell in trials.groupby(keys):
        cell=cell.sort_values('trial_index');values=cell[list(DECODING_ANALYSIS_COLUMNS)].to_numpy()
        if not np.isfinite(values).all() or not np.equal(values,np.round(values)).all():raise ValueError(f'{key}: noninteger behavioral values.')
        validate_decoding_analysis(cell[list(DECODING_ANALYSIS_COLUMNS)],str(key))
        if (not cell.task.eq(cell.distractor_position.ne(0).astype(int)).all()
                or cell.task.value_counts().to_dict()!={0:30,1:30}
                or cell.distractor_position.value_counts().to_dict()!={0:30,2:15,4:15}):
            raise ValueError(f'{key}: unexpected condition composition or task labels.')
    actual=trials.groupby(['subject_id','session_id']).agg(n_runs=('run_id','nunique'),n_trials=('trial_index','size')).reset_index()
    expected=counts[['subject_id','session_id','expected_n_runs']]
    check=actual.merge(expected,on=['subject_id','session_id'],validate='one_to_one',how='outer')
    if (len(check)!=160 or check.isna().any().any() or not check.n_runs.eq(check.expected_n_runs).all()
            or not check.n_trials.eq(60*check.n_runs).all()):raise ValueError('Participant/session behavioral run counts do not match documented inventory.')
    trials['outcome']=trials.feedback.map(OUTCOMES)
    rows=[]
    for (subject,group,session),cell in trials.groupby(['subject_id','group','session_id']):
        for condition in CONDITIONS:
            subset=cell if condition=='both' else cell[cell.task.eq(1 if condition=='distractor' else 0)]
            denominator=len(subset)
            if denominator<=0:raise ValueError('Zero trial denominator.')
            for code,outcome in OUTCOMES.items():
                count=int(subset.feedback.eq(code).sum())
                rows.append(dict(subject_id=subject,group=group,session_id=session,condition=condition,outcome=outcome,
                    count=count,total_trials=denominator,percent=100*count/denominator,n_runs=subset.run_id.nunique()))
    participants=pd.DataFrame(rows)
    totals=participants.groupby(['subject_id','session_id','condition']).agg(percent=('percent','sum'),count=('count','sum'),total=('total_trials','first'))
    if len(participants)!=1440 or not np.allclose(totals.percent,100) or not totals['count'].eq(totals.total).all():
        raise ValueError('Response categories must partition every participant/session/condition into 100%.')
    summary=participants.groupby(['group','condition','outcome','session_id']).agg(mean_percent=('percent','mean'),
        sd_percent=('percent','std'),n=('subject_id','nunique'),response_count=('count','sum'),total_trials=('total_trials','sum')).reset_index()
    summary['sem_percent']=summary.sd_percent/np.sqrt(summary.n)
    if len(summary)!=90 or not summary.n.eq(16).all():raise ValueError('Expected 16 participants in all 90 group/condition/outcome/session cells.')
    for name,table in [('participant_response_rates',participants),('group_response_rates',summary),('behavioral_session_counts',actual)]:table.to_csv(out/f'{name}.csv',index=False)
    qc=dict(n_participants=32,n_per_group=16,n_runs=int(actual.n_runs.sum()),n_trials=len(trials),
        counts_by_response=trials.outcome.value_counts().to_dict(),source_cache=str(root),cache_hashes=hashes,
        source_hashes=source_hashes,added_behavior_only_runs=additions,
        exclusions='Practice inventory inherited from validated intervention analysis; no outcome or RT exclusions.',
        rate='100 × outcome count / all trials within participant/session/condition; response categories sum to 100%.',
        group_summary='Equal participant mean percentages ± between-participant sample SD / sqrt(16).',
        condition_selection='both pools trial counts before division; separate distractor and no-distractor rates also reported.',statistics_run=False)
    (out/'response_validation.json').write_text(json.dumps(qc,indent=2)+'\n')
    print(f'Validated response rates: 32 participants, 160 participant/sessions, {qc["n_runs"]:,} runs, {len(trials):,} trials; all response types retained.')
    print(f'Included {len(additions)} behavior-only run(s) excluded from theta for missing EEG: {additions}')
    print('Response totals:',qc['counts_by_response'],'; percentages partition every participant/session/condition into 100%.')
    return dict(participants=participants,summary=summary,validation=qc)


def plot_intervention_response_rates(participants,summary,output_dir):
    """Nine-panel figure: condition columns, response-type rows; equal participant mean ± SEM."""
    import matplotlib.pyplot as plt
    required={'group','condition','outcome','session_id','mean_percent','sem_percent','n'}
    if (not required.issubset(summary) or len(summary)!=90 or not summary.n.eq(16).all()
            or not np.isfinite(summary[['mean_percent','sem_percent']]).all().all()
            or not summary.mean_percent.between(0,100).all()
            or summary.duplicated(['group','condition','outcome','session_id']).any()):
        raise ValueError('Invalid complete group response summary.')
    check=participants.groupby(['group','condition','outcome','session_id']).percent.agg(['mean','std','count']).reset_index()
    matched=summary.merge(check,on=['group','condition','outcome','session_id'],validate='one_to_one')
    if not np.allclose(matched.mean_percent,matched['mean']) or not np.allclose(matched.sem_percent,matched['std']/np.sqrt(matched['count'])):
        raise ValueError('Plot summary does not match participant rates.')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);stem=out/'intervention_response_rates'
    colors={'bci':'#DD8452','control':'#4C72B0'};labels={'bci':'BCI','control':'Mental rehearsal'}
    limits={outcome:(0,100) for outcome in OUTCOMES.values()}
    rc={'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':7,
        'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(rc):
        fig,axes=plt.subplots(3,3,figsize=(7.05,6.5),sharex=True,sharey='row')
        fig.subplots_adjust(left=.10,right=.985,bottom=.13,top=.88,wspace=.18,hspace=.30)
        for row,outcome in enumerate(OUTCOMES.values()):
            for col,condition in enumerate(CONDITIONS):
                ax=axes[row,col]
                for group in ['bci','control']:
                    cell=summary[summary.group.eq(group)&summary.condition.eq(condition)&summary.outcome.eq(outcome)].sort_values('session_id')
                    if cell.session_id.tolist()!=[1,2,3,4,5]:raise ValueError('Missing plot sessions.')
                    ax.errorbar(cell.session_id,cell.mean_percent,yerr=cell.sem_percent,color=colors[group],marker='o',
                        markersize=3,linewidth=1.1,elinewidth=.7,capsize=2,capthick=.7,label=labels[group])
                ax.set(xlim=(.8,5.2),ylim=limits[outcome],yticks=range(0,101,20),xticks=range(1,6))
                ax.spines[['top','right']].set_visible(False);ax.tick_params(width=.6,length=3,labelsize=6.5)
                ax.text(-.13,1.06,chr(97+row*3+col),transform=ax.transAxes,fontweight='bold',fontsize=9)
                if row==0:ax.set_title({'both':'All trials','distractor':'Distractor','no_distractor':'No distractor'}[condition],fontsize=8,pad=12)
                if col==0:ax.set_ylabel(outcome.capitalize()+' responses (%)',fontsize=7)
                if row==2:ax.set_xlabel('Intervention session',fontsize=7)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.54,.985),fontsize=8)
        fig.text(.54,.055,'Mean ± SEM; n = 16 per group. All panels share the same 0–100% y scale.',ha='center',fontsize=6.5)
        fig.text(.54,.028,'All experimental behavioral trials included, including the run with missing EEG; no RT exclusions.',ha='center',fontsize=6.5)
        paths={ext:str(stem.with_suffix('.'+ext)) for ext in ['pdf','svg','png']}
        for ext,path in paths.items():fig.savefig(path,dpi=350,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    meta=dict(paths=paths,colors=colors,y_limits_percent=limits,conditions=list(CONDITIONS),outcomes=list(OUTCOMES.values()),
        uncertainty='Between-participant SEM; 16 participants per group; equal weights.',statistics_run=False,
        figure_sha256={ext:hashlib.sha256(Path(path).read_bytes()).hexdigest() for ext,path in paths.items()})
    stem.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Saved PDF/SVG/PNG: pooled and separate conditions, correct/incorrect/timeout rows; mean ± SEM, BCI orange/control blue.')
    return meta
