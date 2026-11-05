"""Independent regression audit and exploratory intervention theta linear trends."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from .config import EXPECTED_SUBJECTS, get_subject_group


def audit_intervention_theta_statistics(data, omnibus, trajectory_root, output_dir):
    """Check plotted inputs and reconstruct all seven effects with fixed-subject OLS.

    Separate models for subject means, session-averaged condition values,
    condition-averaged session values, and condition-difference session values
    supply the appropriate independent subject error strata. No calls to the
    original ANOVA estimator or its orthonormal-contrast code are used.
    """
    required = {'subject_id','group','condition','session_id','theta_db'}
    if not required.issubset(data):
        raise ValueError(f'Missing fields: {sorted(required-set(data))}')
    if (len(data)!=320 or data[list(required)].isna().any().any()
            or data.duplicated(['subject_id','condition','session_id']).any()
            or not np.isfinite(data.theta_db).all()
            or set(data.subject_id)!=set(EXPECTED_SUBJECTS)
            or not data.group.eq(data.subject_id.map(get_subject_group)).all()
            or set(data.condition)!={'distractor','no_distractor'}
            or set(data.session_id)!=set(range(1,6))):
        raise ValueError('Expected 32 configured participants × 2 conditions × 5 sessions; complete finite data.')
    if not data.groupby(['subject_id','condition']).size().eq(5).all():
        raise ValueError('Incomplete participant condition cells.')
    if not {'effect','F','df_num','df_den','p_uncorrected'}.issubset(omnibus) or len(omnibus)!=7:
        raise ValueError('Expected all seven original omnibus effects and unadjusted statistics.')
    root=Path(trajectory_root);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'cache/final_manifest.json').read_text())['files_sha256']
    path=root/'cache/participant_trajectories.csv';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if [v for k,v in manifest.items() if Path(k).name==path.name]!=[digest]:
        raise ValueError('Participant trajectory checksum failed.')
    trajectory=pd.read_csv(path)
    figure_csv=root/'figures/midfrontal_theta_seven_sessions.csv'
    plot=pd.read_csv(figure_csv)
    meta=json.loads((root/'figures/midfrontal_theta_seven_sessions.json').read_text())
    for ext,expected in meta['figure_sha256'].items():
        figure=root/f'figures/midfrontal_theta_seven_sessions.{ext}'
        if hashlib.sha256(figure.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Plotted {ext} artifact has changed.')
    selected=trajectory[trajectory.phase.eq('intervention') & trajectory.condition.isin(['distractor','no_distractor'])]
    keys=['subject_id','condition','session_id']
    pd.testing.assert_frame_equal(data.sort_values(keys).reset_index(drop=True),
        selected[data.columns].sort_values(keys).reset_index(drop=True),check_dtype=False)
    if not selected.session_index.eq(selected.session_id).all():
        raise ValueError('Intervention figure x positions do not match session labels.')
    if not selected.session_label.eq('Training '+selected.session_id.astype(str)).all():
        raise ValueError('Unexpected plotted session labels.')
    expected=data.groupby(['group','condition','session_id']).theta_db.agg(mean='mean',sd='std',n='count').reset_index()
    expected['sem']=expected.sd/np.sqrt(expected.n)
    actual=plot[plot.session_index.isin(range(1,6)) & plot.condition.isin(['distractor','no_distractor'])].rename(columns={'session_index':'session_id'})
    cols=['group','condition','session_id']
    pd.testing.assert_frame_equal(expected.sort_values(cols).reset_index(drop=True),actual[expected.columns].sort_values(cols).reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-12)
    # Sum-coded numeric contrasts avoid group/subject aliasing in formula models.
    d=data.copy();d['g']=d.group.map({'bci':.5,'control':-.5});d['c']=d.condition.map({'distractor':.5,'no_distractor':-.5})
    condition=d.groupby(['subject_id','g','c'],as_index=False).theta_db.mean()
    session=d.groupby(['subject_id','g','session_id'],as_index=False).theta_db.mean()
    delta=d.pivot(index=['subject_id','g','session_id'],columns='condition',values='theta_db').reset_index()
    delta['theta_db']=delta.distractor-delta.no_distractor
    for frame in [session,delta]:
        for i in range(1,5):frame[f's{i}']=np.where(frame.session_id==i,1,np.where(frame.session_id==5,-1,0))
    subject=d.groupby(['subject_id','g'],as_index=False).theta_db.mean()
    session_formula='theta_db ~ C(subject_id) + s1+s2+s3+s4 + g:s1+g:s2+g:s3+g:s4'
    fits={
        'subject':smf.ols('theta_db ~ g',data=subject).fit(),
        'condition':smf.ols('theta_db ~ C(subject_id) + c + g:c',data=condition).fit(),
        'session':smf.ols(session_formula,data=session).fit(),
        'condition_difference':smf.ols(session_formula,data=delta).fit(),
    }
    specs=[('Group','subject','g=0'),('Distractor presence','condition','c=0'),
        ('Group × Distractor presence','condition','g:c=0'),
        ('Session','session','s1=0,s2=0,s3=0,s4=0'),
        ('Group × Session','session','g:s1=0,g:s2=0,g:s3=0,g:s4=0'),
        ('Distractor presence × Session','condition_difference','s1=0,s2=0,s3=0,s4=0'),
        ('Group × Distractor presence × Session','condition_difference','g:s1=0,g:s2=0,g:s3=0,g:s4=0')]
    rows=[]
    for effect,model,hypothesis in specs:
        fit=fits[model]
        if np.linalg.matrix_rank(fit.model.exog)!=fit.model.exog.shape[1]:
            raise ValueError(f'{model}: regression is rank deficient.')
        test=fit.f_test(hypothesis);old=omnibus[omnibus.effect.eq(effect)]
        if len(old)!=1:raise ValueError(f'{effect}: missing/duplicate original ANOVA effect.')
        old=old.iloc[0];F=float(np.asarray(test.fvalue).item());p=float(test.pvalue)
        if (abs(F-old.F)>1e-10 or abs(p-old.p_uncorrected)>1e-10
                or test.df_num!=old.df_num or test.df_denom!=old.df_den):
            raise ValueError(f'{effect}: independent regression disagrees with original ANOVA.')
        rows.append(dict(effect=effect,model=model,F_OLS=F,F_original=old.F,df_num=test.df_num,
            df_den=test.df_denom,p_OLS=p,p_original=old.p_uncorrected,F_absolute_error=abs(F-old.F),
            model_rank=int(np.linalg.matrix_rank(fit.model.exog)),model_n=int(fit.nobs)))
    comparisons=pd.DataFrame(rows)
    comparisons.to_csv(out/'independent_OLS_ANOVA.csv',index=False)
    expected.to_csv(out/'verified_plot_values.csv',index=False)
    qc=dict(n_subjects=32,n_per_group=16,n_cells=320,plot_means_and_SEMs_match=True,
        subject_values_match=True,session_numbers_match=True,group_assignments_match=True,
        seven_omnibus_effects_reproduced=True,maximum_F_error=float(comparisons.F_absolute_error.max()),
        method='Independent statsmodels formula OLS/F tests with subject intercepts and sum-coded group/condition/session terms; separate error models for subject means, condition values, session values and condition-difference session values.',
        residual_df={name:int(fit.df_resid) for name,fit in fits.items()},
        plot_uncertainty='Between-participant SEM at each session, not uncertainty of individual changes or group slope differences.',
        scope='Intervention sessions 1–5 only. Pre/Post assessment points are not in this five-session ANOVA.',
        missing_data='No missing participant-condition-session cells. Previously approved e52 Session 3 seven-run mean retained.',
        source_hashes={'participant_trajectories.csv':digest,'figure_summary.csv':hashlib.sha256(figure_csv.read_bytes()).hexdigest()},
        correction=None,statistical_implementation_bug_found=False)
    (out/'audit_validation.json').write_text(json.dumps(qc,indent=2)+'\n')
    print('Audit passed: all 320 participant values, 20 plotted intervention group means/SEMs, session/group labels, and seven ANOVA effects agree.')
    print(f'Independent regression maximum F discrepancy: {comparisons.F_absolute_error.max():.3g}; error dfs: 30, 30, 120, 120.')
    print(comparisons[['effect','F_OLS','df_num','df_den','p_OLS']].to_string(index=False))
    return dict(comparisons=comparisons,plot_values=expected,validation=qc)


def analyze_intervention_theta_linear_trends(data,output_dir):
    """Exploratory participant slopes across sessions 1–5, separate by condition.

    Test each group's slopes against zero and compare BCI minus control slopes
    directly using Welch t tests. All six p values are two-sided/unadjusted.
    """
    required={'subject_id','group','condition','session_id','theta_db'}
    if (not required.issubset(data) or len(data)!=320
            or data[list(required)].isna().any().any()
            or data.duplicated(['subject_id','condition','session_id']).any()
            or set(data.subject_id)!=set(EXPECTED_SUBJECTS)
            or not data.group.eq(data.subject_id.map(get_subject_group)).all()
            or not np.isfinite(data.theta_db).all()):
        raise ValueError('Expected 320 complete configured participant/condition/session values.')
    x=np.arange(1,6,dtype=float);weights=(x-x.mean())/np.sum((x-x.mean())**2)
    rows=[];max_error=0.
    for (subject,group,condition),cell in data.groupby(['subject_id','group','condition']):
        cell=cell.sort_values('session_id')
        if cell.session_id.tolist()!=[1,2,3,4,5] or condition not in ['distractor','no_distractor']:
            raise ValueError(f'{subject}/{condition}: unexpected session set or condition.')
        y=cell.theta_db.to_numpy();slope=float(y@weights)
        independent=stats.linregress(x,y).slope;max_error=max(max_error,abs(slope-independent))
        rows.append(dict(subject_id=subject,group=group,condition=condition,slope_db_per_session=slope,
            session1_db=y[0],session5_db=y[-1],session5_minus_session1_db=y[-1]-y[0]))
    slopes=pd.DataFrame(rows)
    if len(slopes)!=64 or not slopes.groupby(['group','condition']).size().eq(16).all() or max_error>1e-12:
        raise ValueError('Invalid slope alignment, sample counts, or independent slope reconstruction.')
    within=[];between=[]
    for condition in ['distractor','no_distractor']:
        values={}
        for group in ['bci','control']:
            cell=slopes[slopes.condition.eq(condition)&slopes.group.eq(group)]
            v=cell.slope_db_per_session.to_numpy();values[group]=v
            if v.std(ddof=1)<=0:raise ValueError('Degenerate slope variance.')
            test=stats.ttest_1samp(v,0);ci=test.confidence_interval()
            within.append(dict(condition=condition,group=group,n=len(v),slope_db_per_session=float(v.mean()),
                sd_slope=float(v.std(ddof=1)),ci95_low=float(ci.low),ci95_high=float(ci.high),
                t=float(test.statistic),df=float(test.df),p_uncorrected=float(test.pvalue),cohen_dz=float(v.mean()/v.std(ddof=1)),
                session1_mean_db=float(cell.session1_db.mean()),session5_mean_db=float(cell.session5_db.mean()),
                mean_session5_minus_session1_db=float(cell.session5_minus_session1_db.mean())))
        a,b=values['bci'],values['control'];test=stats.ttest_ind(a,b,equal_var=False);ci=test.confidence_interval()
        between.append(dict(condition=condition,contrast='BCI slope minus control slope',n_bci=len(a),n_control=len(b),
            difference_db_per_session=float(a.mean()-b.mean()),ci95_low=float(ci.low),ci95_high=float(ci.high),
            t=float(test.statistic),df=float(test.df),p_uncorrected=float(test.pvalue),
            cohen_d=float((a.mean()-b.mean())/np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2))))
    within=pd.DataFrame(within);between=pd.DataFrame(between)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    for name,table in [('participant_slopes',slopes),('within_group_trends',within),('between_group_trends',between)]:table.to_csv(out/f'{name}.csv',index=False)
    qc=dict(exploratory=True,reason='Follow-up to the visually observed intervention-session decline; not prespecified.',
        sessions=[1,2,3,4,5],slope_weights=weights.tolist(),units='dB per intervention session',n_subjects=32,
        n_tests=6,correction=None,tests='Two-sided one-sample t tests on participant slopes and Welch between-group slope tests.',
        no_independence_assumption_between_repeated_sessions='Each participant contributes one slope per condition; group inference uses independent participants.',
        independent_linregress_maximum_error=max_error,pre_post_included=False,additional_exclusions=0,
        interpretation='A significant decrease in one group and nonsignificant decrease in another does not establish a between-group difference; use the direct slope contrast. Nonsignificance is not evidence of stability/equivalence.')
    (out/'trend_validation.json').write_text(json.dumps(qc,indent=2)+'\n')
    print('Exploratory linear trends: sessions 1–5, one slope per participant/condition, 16/group, six unadjusted two-sided tests.')
    print(within.to_string(index=False));print(between.to_string(index=False))
    return dict(participant_slopes=slopes,within_group=within,between_group=between,validation=qc)
