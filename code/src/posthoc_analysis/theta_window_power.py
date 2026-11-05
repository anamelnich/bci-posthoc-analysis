"""Descriptive baseline and stimulus-window power from retained-trial caches."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def summarize_theta_roi_window_power(cache_dir, output_dir, roi=('FZ','FCZ','CZ')):
    """Pool linear trial power, summarize each window, then equally weight subjects.

    Return unnormalized power in cache units, plus the existing dB-normalized
    measure for reference. No raw EEG is recomputed and no statistics are run.
    """
    from .config import EXPECTED_SUBJECTS, get_subject_group
    root=Path(cache_dir).resolve();out=Path(output_dir).resolve()
    roi=tuple(c.upper() for c in roi)
    if not roi or len(set(roi))!=len(roi):raise ValueError('ROI requires distinct channel labels.')
    manifest=json.loads((root/'final_manifest.json').read_text());hashes={}
    for name in ['settings.json','run_log.csv','pooled_baseline_power.csv','participant_session_topographies.csv','trial_mask.csv']:
        expected=[v for k,v in manifest['files_sha256'].items() if Path(k).name==name]
        digest=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if len(expected)!=1 or expected[0]!=digest:raise ValueError(f'{name}: source checksum mismatch.')
        hashes[name]=digest
    settings=json.loads((root/'settings.json').read_text())
    if (not settings.get('surface_laplacian') or not settings.get('rt_cleaned')
            or settings['baseline_s']!=[-.3,-.1] or settings['target_s']!=[.1,.5]
            or settings['sfreq']!=512 or not np.array_equal(settings['frequencies_hz'],np.arange(4,8.01,.5))):
        raise ValueError('Expected current Laplacian theta cache and requested windows.')
    log=pd.read_csv(root/'run_log.csv');mask=pd.read_csv(root/'trial_mask.csv')
    baseline_table=pd.read_csv(root/'pooled_baseline_power.csv')
    normalized=pd.read_csv(root/'participant_session_topographies.csv')
    labels=settings['surface_laplacian']['channels'];indices=[labels.index(c) for c in roi]
    runkeys=['subject_id','group','session_id','run_id']
    if (len(log)!=383 or log.duplicated(runkeys).any() or set(log.subject_id)!=set(EXPECTED_SUBJECTS)
            or set(log.session_id)!={1,5} or log.n_retained.sum()!=21940
            or len(mask)!=22980 or mask.retained.sum()!=21940
            or mask.duplicated(['subject_id','session_id','run_id','trial']).any()):
        raise ValueError('Unexpected cohort, run inventory, or behavioral mask.')
    rows=[];baseline_errors=[];normalized_errors=[]
    for (subject,group,session),runs in log.groupby(['subject_id','group','session_id']):
        expected_runs=7 if subject=='e30' and session==1 else 8 if session==1 else 4
        if len(runs)!=expected_runs or group!=get_subject_group(subject):raise ValueError('Group/run mismatch.')
        bs=np.zeros((len(roi),9));ps=np.zeros((len(roi),9,205));count=0
        for run in runs.itertuples():
            path=root/'runs'/Path(run.npz_path).name  # Resolve copied checkpoints locally; hash checked below.
            if hashlib.sha256(path.read_bytes()).hexdigest()!=run.npz_sha:raise ValueError(f'{path}: corrupt cache.')
            cell=mask[(mask.subject_id==subject)&(mask.session_id==session)&(mask.run_id==run.run_id)].sort_values('trial')
            if cell.trial.tolist()!=list(range(1,61)):raise ValueError('Invalid trial numbering.')
            retained=cell.loc[cell.retained]
            expected_counts=np.array([(retained.task==c).sum() for c in (0,1)])
            with np.load(path,allow_pickle=False) as z:
                b=z['baseline_sum'];p=z['post_sum'];n=z['condition_counts']
                if (z['channel_labels'].tolist()!=labels or b.shape!=(61,9) or p.shape!=(2,61,9,205)
                        or not np.array_equal(z['frequencies'],settings['frequencies_hz'])
                        or not np.allclose(z['post_times'],np.arange(52,257)/512,rtol=0,atol=1e-12)
                        or int(z['n_baseline_samples'])!=102 or not np.array_equal(n,expected_counts)
                        or int(n.sum())!=run.n_retained
                        or not np.array_equal(z['retained_trial_numbers'],retained.trial.to_numpy())
                        or not np.isfinite(b).all() or not np.isfinite(p).all()
                        or np.any(b<=0) or np.any(p<0)):
                    raise ValueError(f'{path}: invalid power axes, values or retained-trial alignment.')
                bs+=b[indices];ps+=p[:,indices].sum(axis=0);count+=int(n.sum())
        b=bs/count;p=ps/count
        reference=baseline_table[(baseline_table.subject_id==subject)&(baseline_table.session_id==session)&baseline_table.channel.isin(roi)]
        ref=reference.pivot(index='channel',columns='frequency_hz',values='baseline_power').reindex(index=roi,columns=settings['frequencies_hz']).to_numpy()
        if ref.shape!=b.shape or not np.isfinite(ref).all():raise ValueError('Incomplete baseline reference.')
        baseline_errors.append(float(np.max(abs(b-ref)/b)))
        theta_db=float((10*np.log10(p/b[:,:,None])).mean())
        ref_db=normalized[(normalized.subject_id==subject)&(normalized.session_id==session)&(normalized.condition=='both')&normalized.channel.isin(roi)]
        if len(ref_db)!=len(roi):raise ValueError('Missing normalized ROI reference.')
        normalized_errors.append(abs(theta_db-float(ref_db.theta_db.mean())))
        rows.append(dict(subject_id=subject,group=group,session_id=int(session),n_trials=count,
            baseline_power=float(b.mean()),stimulus_power=float(p.mean()),normalized_theta_db=theta_db,
            baseline_log_power_db=float((10*np.log10(b)).mean()),
            stimulus_log_power_db=float((10*np.log10(p)).mean())))
    participants=pd.DataFrame(rows)
    if (len(participants)!=64 or participants.n_trials.sum()!=21940
            or max(baseline_errors)>1e-12 or max(normalized_errors)>1e-12):
        raise ValueError('Participant/window reconstruction does not match existing theta data.')
    columns=['baseline_power','stimulus_power','normalized_theta_db','baseline_log_power_db','stimulus_log_power_db']
    summary=participants.groupby(['group','session_id'])[columns].mean().reset_index()
    summary['n_participants']=participants.groupby(['group','session_id']).subject_id.nunique().to_numpy()
    if not summary.n_participants.eq(16).all():raise ValueError('Expected 16 participants in each group/session.')
    qc=dict(roi=list(roi),n_participants=32,n_retained_trials=21940,baseline_s=[-.3,-.1],stimulus_s=[.1,.5],
        pooling='Pool retained trials across runs and distractor conditions; arithmetic mean across time, nine frequencies, ROI channels; equal participant weights.',
        units='Unnormalized wavelet power in squared stored-amplitude units per metre^4; report as arbitrary units because GDF physical amplitude scaling was not independently established. Not a power spectral density.',
        normalized_measure='Mean of channel/frequency/time-specific 10log10(stimulus power / baseline power), then equal participant weights; not the log ratio of group-mean powers.',
        maximum_baseline_relative_error=max(baseline_errors),maximum_normalized_error_db=max(normalized_errors),
        input_sha256=hashes,no_inferential_statistics=True)
    out.mkdir(parents=True,exist_ok=True)
    participants.to_csv(out/'participant_window_power.csv',index=False)
    summary.to_csv(out/'group_window_power.csv',index=False)
    (out/'validation.json').write_text(json.dumps(qc,indent=2)+'\n')
    print(f'Validated {len(roi)}-electrode windows: 64 participant/session values, 16/group, 21,940 retained trials.')
    print(f'Baseline relative reconstruction error {max(baseline_errors):.3g}; normalized-theta error {max(normalized_errors):.3g} dB. No statistics run.')
    print(summary.to_string(index=False))
    return participants,summary,qc


def plot_theta_roi_window_bars(participants, output_dir, include_normalized=False):
    """Plot window means and optionally normalized dB, with SEM and paired dots."""
    import matplotlib.pyplot as plt
    from .config import EXPECTED_SUBJECTS, get_subject_group
    required={'subject_id','group','session_id','baseline_power','stimulus_power','n_trials'}
    measures=['baseline_power','stimulus_power']
    if not isinstance(include_normalized,bool):raise ValueError('include_normalized must be boolean.')
    if include_normalized:
        required.add('normalized_theta_db');measures.append('normalized_theta_db')
    if (not required.issubset(participants) or len(participants)!=64
            or participants.duplicated(['subject_id','session_id']).any()
            or set(participants.subject_id)!=set(EXPECTED_SUBJECTS)
            or set(participants.session_id)!={1,5}
            or not np.isfinite(participants[measures]).all().all()
            or (participants[['baseline_power','stimulus_power']]<=0).any().any() or participants.n_trials.sum()!=21940):
        raise ValueError('Expected 64 finite observations, positive linear power and 21,940 trials.')
    if not participants.group.eq(participants.subject_id.map(get_subject_group)).all():
        raise ValueError('Participant group assignment differs from configuration.')
    rows=[]
    for (group,session),cell in participants.groupby(['group','session_id']):
        if len(cell)!=16:raise ValueError('Each group/session requires 16 participants.')
        for measure in measures:
            rows.append(dict(group=group,session_id=int(session),measure=measure,n=16,
                mean_power=float(cell[measure].mean()),sem_power=float(cell[measure].std(ddof=1)/4)))
    summary=pd.DataFrame(rows)
    scale=1e8
    linear_summary=summary[summary.measure.isin(['baseline_power','stimulus_power'])]
    upper=max(float(participants[['baseline_power','stimulus_power']].max().max()),
        float((linear_summary.mean_power+linear_summary.sem_power).max()))/scale*1.12
    normalized_limits=None
    if include_normalized:
        normalized=summary[summary.measure=='normalized_theta_db']
        low=min(0.,float(participants.normalized_theta_db.min()),float((normalized.mean_power-normalized.sem_power).min()))
        high=max(0.,float(participants.normalized_theta_db.max()),float((normalized.mean_power+normalized.sem_power).max()))
        padding=max(.1,(high-low)*.08)
        normalized_limits=[low-padding if low<0 else 0.,high+padding]
    colors={'bci':'#3978A5','control':'#BD7548'}
    rc={'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
        'font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    stem=out/'theta_midline_laplacian_baseline_stimulus_bars'
    if include_normalized:stem=out/'theta_midline_laplacian_baseline_stimulus_normalized_bars'
    paths={ext:str(stem.with_suffix('.'+ext)) for ext in ['pdf','svg','png']}
    pairs_drawn=0
    with plt.rc_context(rc):
        fig,axes=plt.subplots(1,len(measures),figsize=(7.05,3.5),sharey=False)
        axes[1].sharey(axes[0]);axes[1].tick_params(labelleft=False)
        fig.subplots_adjust(left=.09,right=.985,bottom=.29,top=.79,wspace=.52 if include_normalized else .2)
        for panel,(ax,measure,title) in enumerate(zip(axes,measures,
                ['Baseline\n−300 to −100 ms','Stimulus\n100–500 ms','Baseline-normalized\nstimulus power'])):
            panel_scale=1. if measure=='normalized_theta_db' else scale
            for group,offset in [('bci',0),('control',2.6)]:
                pairs=participants[participants.group==group].pivot(index='subject_id',columns='session_id',values=measure).sort_index()
                if pairs.shape!=(16,2) or pairs.isna().any().any():raise ValueError('Incomplete pre/post participant pairs.')
                for index,session in enumerate([1,5]):
                    cell=summary[(summary.group==group)&(summary.session_id==session)&(summary.measure==measure)].iloc[0]
                    x=offset+index
                    ax.bar(x,cell.mean_power/panel_scale,width=.68,color=colors[group],alpha=.22,
                        edgecolor=colors[group],linewidth=.75,zorder=1)
                    ax.errorbar(x,cell.mean_power/panel_scale,yerr=cell.sem_power/panel_scale,
                        fmt='none',ecolor='black',elinewidth=.85,capsize=3,capthick=.85,zorder=5)
                # The same participant gets the same horizontal offset in both
                # sessions and panels, preserving pairing without hiding dots.
                for jitter,(_,pair) in zip(np.linspace(-.20,.20,16),pairs.iterrows()):
                    x=np.array([offset,offset+1])+jitter;y=pair[[1,5]].to_numpy()/panel_scale
                    ax.plot(x,y,color=colors[group],alpha=.32,linewidth=.55,zorder=2)
                    ax.scatter(x,y,s=10,color=colors[group],alpha=.9,edgecolors='white',linewidths=.25,zorder=4)
                    pairs_drawn+=1
                ax.text(offset+.5,-.18,'BCI' if group=='bci' else ('Mental\nrehearsal' if include_normalized else 'Mental rehearsal'),
                    transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=7,color=colors[group])
            ax.set_xticks([0,1,2.6,3.6],['Pre','Post','Pre','Post'])
            ax.set_xlim(-.6,4.2)
            if measure=='normalized_theta_db':
                ax.set_ylim(*normalized_limits)
                ax.set_ylabel('Normalized theta power (dB)',labelpad=3)
                ax.axhline(0,color='.5',linewidth=.6,linestyle='--',zorder=0)
            else:ax.set_ylim(0,upper)
            ax.set_title(title,fontsize=8,pad=8)
            ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',width=.6,length=3)
            ax.text(-.10,1.21,chr(97+panel),transform=ax.transAxes,fontweight='bold',fontsize=9)
        axes[0].set_ylabel(r'Unnormalized theta power ($10^{8}$ a.u.)')
        fig.text(.53,.965,'Laplacian theta | Fz, FCz, Cz | 4–8 Hz | Pooled trials',ha='center',fontsize=8)
        fig.text(.53,.045,'Bars: mean ± SEM; points: participants; lines: paired sessions. n = 16 per group.',ha='center',fontsize=6.5)
        if pairs_drawn!=32*len(measures):raise ValueError('Expected 32 paired participants in each panel.')
        for ext,path in paths.items():fig.savefig(path,dpi=350,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    summary.to_csv(stem.with_suffix('.csv'),index=False)
    metadata=dict(paths=paths,n_subjects=32,n_per_group=16,n_paired_lines=pairs_drawn,n_points=2*pairs_drawn,
        displayed_scale=scale,shared_linear_ylim=[0,upper],normalized_ylim_db=normalized_limits,
        include_normalized=include_normalized,uncertainty='Between-participant SEM',
        units='Arbitrary units of unnormalized Laplacian wavelet power; not microvolts squared.',
        no_inferential_statistics=True,
        figure_sha256={ext:hashlib.sha256(Path(p).read_bytes()).hexdigest() for ext,p in paths.items()})
    stem.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Validated and plotted 32 paired participants in each of {len(measures)} panels ({2*pairs_drawn} points); mean ± SEM bars.')
    if include_normalized:print('First two panels share linear-power limits; third uses existing participant-normalized dB values and a separate axis.')
    print('Saved PDF, SVG and PNG. No statistical tests or new EEG computations.')
    return metadata
