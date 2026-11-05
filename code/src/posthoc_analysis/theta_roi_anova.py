"""Participant-level theta ROI changes and balanced 2 x 2 mixed ANOVAs."""


from pathlib import Path


import hashlib


import json


import numpy as np


import pandas as pd


from scipy import stats


from statsmodels.stats.multitest import multipletests


from .config import EXPECTED_SUBJECTS, get_subject_group


from .theta_topography import summarize_theta_topographies


def prepare_theta_roi_change(cache_dir, output_dir, roi=('FZ','FCZ','CZ')):
    """Validate cached power, average electrode dB values, and pair sessions."""
    root=Path(cache_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    settings=json.loads((root/'settings.json').read_text())
    expected={'reference_channel':'CPZ','epoch_s':[-1.,1.5],'baseline_s':[-.3,-.1],
              'target_s':[.1,.5],'voltage_baseline':None,'amplitude_rejection':None}
    if any(settings.get(k)!=v for k,v in expected.items()):
        raise ValueError('Use the corrected CPz-reference, paper-epoch theta cache.')
    if not np.array_equal(settings['frequencies_hz'],np.arange(4.,8.01,.5)):
        raise ValueError('Expected the existing 4-8 Hz theta estimator.')
    roi=[str(c).upper() for c in roi]
    if not roi or len(set(roi))!=len(roi):
        raise ValueError('Provide distinct scalp electrodes for the ROI.')
    laplacian=settings.get('surface_laplacian')
    if laplacian:
        channels=laplacian.get('channels',[])
        if (len(channels)!=61 or len(set(channels))!=61 or not set(roi).issubset(channels)
                or laplacian.get('lambda2')!=1e-5 or laplacian.get('stiffness')!=4
                or laplacian.get('n_legendre_terms')!=50):
            raise ValueError('Unexpected Laplacian channel layout or transform parameters.')
        print('Using signal-level Laplacian theta; ROI is averaged after channel-specific power normalization.')
    # This existing loader verifies every run cache hash, dimensions, trial
    # counts, channel/frequency/time alignment and participant/session pairing.
    table=summarize_theta_topographies(root)
    if not set(roi).issubset(table.channel):raise ValueError('ROI contains unavailable scalp electrodes.')
    subset=table[table.channel.isin(roi)&table.condition.isin(['distractor','no_distractor'])].copy()
    keys=['subject_id','group','session_id','condition']
    groups=subset.groupby(keys)
    if len(subset)!=32*2*2*len(roi) or not groups.size().eq(len(roi)).all() or not groups.n_trials.nunique().eq(1).all():
        raise ValueError('Incomplete ROI electrodes or inconsistent per-channel trial counts.')
    session=groups.agg(theta_db=('theta_db','mean'),n_trials=('n_trials','first')).reset_index()
    wide=session.pivot(index=['subject_id','group','condition'],columns='session_id',values='theta_db')
    if set(wide.columns)!={1,5} or wide.isna().any().any():raise ValueError('Unpaired ROI sessions.')
    change=wide.rename(columns={1:'pre_db',5:'post_db'}).reset_index()
    change['change_db']=change.post_db-change.pre_db
    count=session.pivot(index=['subject_id','group','condition'],columns='session_id',values='n_trials')
    change=change.merge(count.rename(columns={1:'n_pre_trials',5:'n_post_trials'}).reset_index(),
                        on=['subject_id','group','condition'],validate='one_to_one')
    channel_pairs=subset.pivot(index=['subject_id','group','condition','channel'],columns='session_id',values='theta_db')
    alternative=(channel_pairs[5]-channel_pairs[1]).groupby(['subject_id','group','condition']).mean()
    error=float(abs(change.set_index(['subject_id','group','condition']).change_db-alternative).max())
    if error>1e-12 or len(change)!=64:
        raise ValueError('ROI/session averaging mismatch or unexpected participant cells.')
    n_trials=int(change.n_pre_trials.sum()+change.n_post_trials.sum())
    input_names=['settings.json','participant_session_topographies.csv','run_log.csv','validation.json']
    if settings.get('rt_cleaned',False):
        mask_path=root/'trial_mask.csv'
        mask_qc=json.loads((root/'mask_validation.json').read_text())
        if hashlib.sha256(mask_path.read_bytes()).hexdigest()!=mask_qc['mask_sha256']:
            raise ValueError('RT trial mask checksum mismatch.')
        mask=pd.read_csv(mask_path)
        required={'subject_id','session_id','run_id','trial','task','retained','missing_status_response'}
        if not required.issubset(mask) or mask.duplicated(['subject_id','session_id','run_id','trial']).any():
            raise ValueError('Invalid RT mask fields or duplicate trials.')
        if (len(mask)!=22980 or mask.retained.dtype!=bool or not mask.task.isin([0,1]).all()
                or mask.missing_status_response.sum()!=47 or mask.retained.sum()!=mask_qc['n_retained']
                or n_trials!=mask_qc['n_retained'] or len(mask)-n_trials!=settings['expected_excluded_trials']):
            raise ValueError('RT mask counts do not reconcile with validated theta settings.')
        retained=mask.loc[mask.retained].copy()
        retained['condition']=retained.task.map({0:'no_distractor',1:'distractor'})
        expected_counts=retained.groupby(['subject_id','session_id','condition']).size().rename('expected_trials').reset_index()
        matched=session.merge(expected_counts,on=['subject_id','session_id','condition'],how='outer',validate='one_to_one',indicator=True)
        if not matched._merge.eq('both').all() or not matched.n_trials.eq(matched.expected_trials).all():
            raise ValueError('ROI cell counts disagree with retained behavioral trial mask.')
        input_names+=['trial_mask.csv','mask_validation.json']
    elif n_trials!=22980 or (change.n_post_trials!=120).any():
        raise ValueError('Unscreened ROI trial counts do not reconcile with the source cohort.')
    for row in change.itertuples():
        expected_pre=210 if row.subject_id=='e30' else 240
        if (not settings.get('rt_cleaned',False) and row.n_pre_trials!=expected_pre) or row.group!=get_subject_group(row.subject_id):
            raise ValueError(f'{row.subject_id}: unexpected trial count or group.')
    session.to_csv(out/'roi_session_means.csv',index=False)
    change.to_csv(out/'roi_subject_changes.csv',index=False)
    provenance=dict(roi=roi,signal_space='laplacian' if laplacian else 'voltage',n_subjects=32,n_trials=n_trials,n_rejected=22980-n_trials,n_original_trials=22980,
        maximum_averaging_order_error_db=error,settings=settings,
        averaging=f"Arithmetic mean of channel-specific normalized dB power across {'/'.join(roi)}, then Session 5 minus Session 1; two conditions kept separate",
        input_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                      input_names})
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(f'ROI {roi} validated: 32 subjects, 16/group, 64 paired condition changes; {n_trials:,} trials; no additional exclusions. Averaging-order error {error:.3g} dB.')
    return change


def run_theta_roi_mixed_anova(change, output_dir, correction='holm', include_diagnostics=True):
    """Run classical mixed ANOVA with separate subject and within error terms.

    Independently check sums-of-squares F statistics using subject-contrast OLS.
    Set correction=None to report unadjusted p values only. The historical
    midline analysis retains its default Holm option for reproducibility.
    """
    import statsmodels.api as sm
    if correction not in (None,'holm'):raise ValueError('correction must be None or holm.')
    required={'subject_id','group','condition','change_db'}
    if not required.issubset(change) or change.duplicated(['subject_id','condition']).any():
        raise ValueError('Missing fields or duplicate participant-condition observations.')
    wide=change.pivot(index=['subject_id','group'],columns='condition',values='change_db')
    if len(wide)!=32 or set(wide.columns)!={'distractor','no_distractor'} or not np.isfinite(wide.to_numpy()).all():
        raise ValueError('Mixed ANOVA requires 32 complete, finite condition pairs.')
    subjects=wide.index.get_level_values('subject_id');group=wide.index.get_level_values('group').to_numpy()
    if set(subjects)!=set(EXPECTED_SUBJECTS) or pd.Series(group).value_counts().to_dict()!={'bci':16,'control':16}:
        raise ValueError('Unexpected cohort or unbalanced group sizes.')
    if any(g!=get_subject_group(s) for s,g in zip(subjects,group)):
        raise ValueError('Participant group assignment mismatch.')
    y=wide[['distractor','no_distractor']].to_numpy();n=32;b=2
    subject_mean=y.mean(axis=1);grand=float(y.mean());condition_mean=y.mean(axis=0)
    gm={g:float(y[group==g].mean()) for g in ['bci','control']}
    cell={g:y[group==g].mean(axis=0) for g in gm}
    ss_group=b*sum(16*(gm[g]-grand)**2 for g in gm)
    ss_subject=b*sum(np.sum((subject_mean[group==g]-gm[g])**2) for g in gm)
    ss_condition=n*np.sum((condition_mean-grand)**2)
    ss_interaction=sum(16*np.sum((cell[g]-gm[g]-condition_mean+grand)**2) for g in gm)
    ss_error=sum(np.sum((y[group==g]-subject_mean[group==g,None]-cell[g]+gm[g])**2) for g in gm)
    ss_total=float(np.sum((y-grand)**2));parts=ss_group+ss_subject+ss_condition+ss_interaction+ss_error
    if min(ss_subject,ss_error)<=0 or not np.isclose(parts,ss_total,rtol=1e-12,atol=1e-12):
        raise ValueError('Invalid error variance or failed ANOVA SS decomposition.')
    design=np.column_stack([np.ones(n),np.where(group=='bci',.5,-.5)])
    difference=y[:,0]-y[:,1]
    models={'mean':sm.OLS(subject_mean,design).fit(),'difference':sm.OLS(difference,design).fit()}
    specifications=[('Group',ss_group,ss_subject,'mean',1,'BCI minus control, averaged across conditions'),
                    ('Distractor presence',ss_condition,ss_error,'difference',0,'Distractor minus no distractor, averaged across groups'),
                    ('Group × Distractor presence',ss_interaction,ss_error,'difference',1,'BCI minus control difference in distractor-minus-no-distractor change')]
    rows=[];errors=[]
    for effect,ss,denominator,model_name,coefficient,contrast in specifications:
        model=models[model_name];f=float(ss/(denominator/30));independent=float(model.tvalues[coefficient]**2)
        if model.df_resid!=30 or not np.isclose(f,independent,rtol=1e-10,atol=1e-10):
            raise ValueError(f'Independent subject-contrast validation failed for {effect}.')
        errors.append(abs(f-independent));ci=model.conf_int()[coefficient]
        rows.append(dict(effect=effect,SS=float(ss),df_num=1,df_den=30,F=f,
            p_value=float(stats.f.sf(f,1,30)),partial_eta_sq=float(ss/(ss+denominator)),
            estimate_db=float(model.params[coefficient]),ci95_low=float(ci[0]),ci95_high=float(ci[1]),
            contrast=contrast,error_term='Subject(Group)' if model_name=='mean' else 'Distractor presence × Subject(Group)'))
    anova=pd.DataFrame(rows)
    if correction=='holm':anova['p_holm_three_effects']=multipletests(anova.p_value,method='holm')[1]
    descriptives=[]
    for (g,c),cell_data in change.groupby(['group','condition']):
        values=cell_data.change_db;mean=float(values.mean());sd=float(values.std(ddof=1));se=sd/np.sqrt(len(values));half=stats.t.ppf(.975,len(values)-1)*se
        descriptives.append(dict(group=g,condition=c,n=len(values),mean_change_db=mean,sd_db=sd,
                                 sem_db=se,ci95_low=mean-half,ci95_high=mean+half))
    descriptives=pd.DataFrame(descriptives)
    diagnostics=[];sensitivity=[]
    for name,values in ([('mean',subject_mean),('difference',difference)] if include_diagnostics else []):
        left=values[group=='bci'];right=values[group=='control'];lev=stats.levene(left,right,center='median')
        for g,v in [('bci',left),('control',right)]:
            sh=stats.shapiro(v)
            diagnostics.append(dict(contrast=name,group=g,n=len(v),shapiro_W=float(sh.statistic),shapiro_p=float(sh.pvalue),
                                    sd=float(np.std(v,ddof=1)),levene_W=float(lev.statistic),levene_p=float(lev.pvalue)))
        # A labeled variance-robust check; it does not replace the requested ANOVA.
        welch=stats.ttest_ind(left,right,equal_var=False)
        se2=np.var(left,ddof=1)/16+np.var(right,ddof=1)/16
        df=se2**2/((np.var(left,ddof=1)/16)**2/15+(np.var(right,ddof=1)/16)**2/15)
        estimate=float(left.mean()-right.mean());half=stats.t.ppf(.975,df)*np.sqrt(se2)
        sensitivity.append(dict(effect='Group' if name=='mean' else 'Group × Distractor presence',
            method='Welch sensitivity, two-sided',estimate_db=estimate,t=float(welch.statistic),df=float(df),p_value=float(welch.pvalue),
            ci95_low=estimate-half,ci95_high=estimate+half))
    diagnostics=pd.DataFrame(diagnostics);sensitivity=pd.DataFrame(sensitivity)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('mixed_anova',anova),('descriptives',descriptives),('assumption_checks',diagnostics),('welch_sensitivity',sensitivity)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    qc=dict(n_subjects=32,n_per_group=16,n_observations=64,ss_total=ss_total,ss_components_sum=float(parts),
        ss_subject_group=float(ss_subject),ss_within_error=float(ss_error),maximum_independent_F_error=max(errors),
        sphericity='Automatically satisfied with two within-subject levels; no correction needed',
        multiplicity=('Raw omnibus p values plus Holm across the three model effects; CIs are pointwise, not simultaneous' if correction=='holm' else 'Unadjusted p values only, as requested; CIs are pointwise'),
        correction=correction,
        include_diagnostics=include_diagnostics,
        scope='User-requested ROI analysis within an ongoing exploratory investigation; ROI selected after topography inspection',
        dv='Post-pre change in session-baseline-normalized theta dB, not absolute theta power')
    (out/'anova_validation.json').write_text(json.dumps(qc,indent=2))
    print('Mixed ANOVA complete: Group between subjects; distractor presence within subjects; all effects F(1,30).')
    columns=['effect','F','p_value','partial_eta_sq']+(['p_holm_three_effects'] if correction=='holm' else [])
    print(anova[columns].to_string(index=False))
    print(f'Independent contrast check: maximum F error {max(errors):.3g}; no participant exclusions.')
    return dict(anova=anova,descriptives=descriptives,diagnostics=diagnostics,sensitivity=sensitivity,validation=qc)


prepare_midline_theta_change = prepare_theta_roi_change


run_midline_theta_mixed_anova = run_theta_roi_mixed_anova
