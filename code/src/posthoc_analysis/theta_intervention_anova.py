"""Balanced Group × Distractor presence × Intervention session mixed ANOVA."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import helmert

from .config import EXPECTED_SUBJECTS, get_subject_group


def prepare_intervention_theta_anova(cache_dir, output_dir):
    """Load checksummed ROI values and select the two conditions and five intervention sessions."""
    root,out=map(Path,(cache_dir,output_dir));out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'final_manifest.json').read_text())['files_sha256'];hashes={}
    for name in ['settings.json','participant_trajectories.csv','summary_validation.json','session_run_counts.csv']:
        expected=[v for k,v in manifest.items() if Path(k).name==name]
        digest=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if expected!=[digest]:raise ValueError(f'{name}: intervention input checksum mismatch.')
        hashes[name]=digest
    settings=json.loads((root/'settings.json').read_text())
    if (settings['power_channels']!=['FZ','FCZ','CZ'] or settings['target_s']!=[.1,.5]
            or settings['baseline_s']!=[-.3,-.1] or settings['frequencies_hz']!=np.arange(4,8.01,.5).tolist()
            or settings['source_task']!='decoding' or settings['rt_cleaned'] or settings['expected_excluded_trials']!=0):
        raise ValueError('Expected the all-trial, 100–500 ms, Fz/FCz/Cz intervention theta estimator.')
    full=pd.read_csv(root/'participant_trajectories.csv')
    if len(full)!=672 or full.duplicated(['subject_id','condition','session_index']).any():
        raise ValueError('Expected the complete 32 × 7 × 3 trajectory table.')
    data=full[full.phase.eq('intervention') & full.condition.isin(['distractor','no_distractor'])].copy()
    data=data[['subject_id','group','session_id','condition','theta_db','n_trials','n_runs']].sort_values(
        ['subject_id','condition','session_id']).reset_index(drop=True)
    if len(data)!=320 or set(data.session_id)!=set(range(1,6)) or data.n_trials.sum()!=69000:
        raise ValueError('ANOVA requires 320 participant/condition/session values covering 69,000 trials.')
    data.to_csv(out/'anova_input.csv',index=False)
    provenance=dict(input_hashes=hashes,source_cache=str(root.resolve()),n_subjects=32,n_per_group=16,
        between_factor='Group: BCI versus mental rehearsal',within_factors={'Distractor presence':['distractor','no_distractor'],'Session':[1,2,3,4,5]},
        dependent_variable='Normalized Laplacian theta power (dB), Fz/FCz/Cz, 4–8 Hz, 100–500 ms; baseline −300 to −100 ms',
        observations='One equally weighted participant/condition/session mean; pre/post assessment and pooled-all-trial values are not ANOVA observations.',
        missing_recording='e52 Session 3 Run 6 unavailable; the previously approved seven-run session mean is retained. No missing ANOVA cells or imputation.',
        trial_exclusions=0,participant_exclusions=0,
        input_sha256=hashlib.sha256((out/'anova_input.csv').read_bytes()).hexdigest())
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Prepared 2 × 2 × 5 mixed ANOVA: 32 participants, 16/group, 320 complete repeated-measures cells; intervention only.',flush=True)
    return data,provenance


def run_intervention_theta_mixed_anova(data, output_dir, correction="GG"):
    """Use orthonormal repeated contrasts and group-adjusted subject error strata.

    Set correction=None for unadjusted reporting, or "GG" for the historical
    Greenhouse–Geisser reporting. For the latter, estimate epsilon
    estimated from pooled within-group residual contrast covariance. Independently
    verify every SS/F using classical marginal-mean decompositions.
    """
    if correction not in (None, 'GG'):
        raise ValueError('correction must be None or GG.')
    required={'subject_id','group','condition','session_id','theta_db'}
    keys=['subject_id','condition','session_id']
    if (not required.issubset(data) or len(data)!=320 or data.duplicated(keys).any()
            or data[list(required)].isna().any().any() or not np.isfinite(data.theta_db).all()
            or set(data.condition)!={'distractor','no_distractor'} or set(data.session_id)!=set(range(1,6))
            or set(data.subject_id)!=set(EXPECTED_SUBJECTS)
            or not data.group.eq(data.subject_id.map(get_subject_group)).all()):
        raise ValueError('Expected complete finite 32 × 2 × 5 data, unique cells and configured groups.')
    columns=pd.MultiIndex.from_product([['distractor','no_distractor'],range(1,6)],names=['condition','session_id'])
    wide=data.pivot(index=['subject_id','group'],columns=['condition','session_id'],values='theta_db').reindex(columns=columns).sort_index()
    if wide.shape!=(32,10) or wide.isna().any().any():raise ValueError('All participants need all 10 repeated cells.')
    group=wide.index.get_level_values('group').to_numpy();groups=['bci','control']
    if [np.sum(group==g) for g in groups]!=[16,16]:raise ValueError('This balanced ANOVA requires 16 participants/group.')
    y=wide.to_numpy().reshape(32,2,5);grand=float(y.mean());N=32;G=2;df_subject=N-G
    # Orthonormal within-subject basis: overall, condition, session, condition×session.
    cm=np.ones(2)/np.sqrt(2);cc=np.array([1.,-1.])/np.sqrt(2)
    sm=np.ones(5)/np.sqrt(5);sc=helmert(5,full=False).T
    scores={
        'Subject':np.einsum('ncs,c,s->n',y,cm,sm)[:,None],
        'Distractor presence':np.einsum('ncs,c,s->n',y,cc,sm)[:,None],
        'Session':np.einsum('ncs,c,sk->nk',y,cm,sc),
        'Distractor presence × Session':np.einsum('ncs,c,sk->nk',y,cc,sc),
    }
    rows=[];errors=[];epsilons=[];partition=0.;strata={}
    for name,z in scores.items():
        k=z.shape[1];mean=z.mean(axis=0);group_means={g:z[group==g].mean(axis=0) for g in groups}
        residual=np.vstack([z[group==g]-group_means[g] for g in groups])
        ss_error=float(np.sum(residual**2));df_error=df_subject*k
        if ss_error<=np.finfo(float).eps:raise ValueError(f'{name}: zero or degenerate subject error variance.')
        covariance=residual.T@residual/df_subject
        epsilon=1. if k==1 else float(np.clip(np.trace(covariance)**2/(k*np.trace(covariance@covariance)),1/k,1))
        interaction=float(sum(16*np.sum((group_means[g]-mean)**2) for g in groups))
        error_name='Subject(Group)' if name=='Subject' else f'{name} × Subject(Group)'
        terms=[('Group',interaction)] if name=='Subject' else [(name,float(N*np.sum(mean**2))),(f'Group × {name}',interaction)]
        partition+=ss_error+sum(ss for _,ss in terms)
        strata[name]=(ss_error,df_error,epsilon)
        errors.append(dict(error_term=error_name,SS_error=ss_error,df_error=df_error,MS_error=ss_error/df_error))
        epsilons.append(dict(within_effect=name,contrast_dimension=k,epsilon_GG=epsilon,
            residual_df=df_subject,covariance_eigenvalues=np.linalg.eigvalsh(covariance).tolist()))
        for effect,ss in terms:
            F=(ss/k)/(ss_error/df_error)
            rows.append(dict(effect=effect,SS=ss,df_num=k,df_den=df_error,MS=ss/k,
                error_term=error_name,SS_error=ss_error,F=float(F),p_uncorrected=float(stats.f.sf(F,k,df_error)),
                epsilon_GG=epsilon,df_num_GG=k*epsilon,df_den_GG=df_error*epsilon,
                p_GG=float(stats.f.sf(F,k*epsilon,df_error*epsilon)),partial_eta_squared=ss/(ss+ss_error)))
    anova=pd.DataFrame(rows)
    if correction is None:
        anova=anova.drop(columns=['epsilon_GG','df_num_GG','df_den_GG','p_GG'])
    # Independent classical marginal-means sums of squares in the original 2×5 array.
    subject_mean=y.mean(axis=(1,2));subject_c=y.mean(axis=2);subject_s=y.mean(axis=1)
    cmarg=y.mean(axis=(0,2));smarg=y.mean(axis=(0,1));cell=y.mean(axis=0)
    cs=cell-cmarg[:,None]-smarg[None,:]+grand
    gm={g:y[group==g].mean() for g in groups};gc={g:y[group==g].mean(axis=(0,2)) for g in groups}
    gs={g:y[group==g].mean(axis=(0,1)) for g in groups};gcell={g:y[group==g].mean(axis=0) for g in groups}
    independent={
        'Group':10*sum(16*(gm[g]-grand)**2 for g in groups),
        'Distractor presence':N*5*np.sum((cmarg-grand)**2),
        'Session':N*2*np.sum((smarg-grand)**2),
        'Distractor presence × Session':N*np.sum(cs**2),
        'Group × Distractor presence':5*sum(16*np.sum((gc[g]-gm[g]-cmarg+grand)**2) for g in groups),
        'Group × Session':2*sum(16*np.sum((gs[g]-gm[g]-smarg+grand)**2) for g in groups),
        'Group × Distractor presence × Session':sum(16*np.sum((gcell[g]-gc[g][:,None]-gs[g][None,:]+gm[g]-cs)**2) for g in groups),
    }
    independent_errors={
        'Subject':10*sum(np.sum((subject_mean[group==g]-gm[g])**2) for g in groups),
        'Distractor presence':5*sum(np.sum((subject_c[group==g]-subject_mean[group==g,None]-gc[g]+gm[g])**2) for g in groups),
        'Session':2*sum(np.sum((subject_s[group==g]-subject_mean[group==g,None]-gs[g]+gm[g])**2) for g in groups),
        'Distractor presence × Session':sum(np.sum((y[group==g]-subject_c[group==g,:,None]-subject_s[group==g,None,:]+subject_mean[group==g,None,None]
            -gcell[g]+gc[g][:,None]+gs[g][None,:]-gm[g])**2) for g in groups),
    }
    ss_errors=[];f_errors=[]
    for row in anova.itertuples():
        stratum='Subject' if row.effect=='Group' else row.effect.removeprefix('Group × ')
        numerator=float(independent[row.effect]);denominator=float(independent_errors[stratum])
        check_f=(numerator/row.df_num)/(denominator/row.df_den)
        ss_errors.extend([abs(row.SS-numerator),abs(row.SS_error-denominator)])
        f_errors.append(abs(row.F-check_f))
    total=float(np.sum((y-grand)**2));partition_error=abs(total-partition)
    if not np.allclose(partition,total,rtol=1e-12,atol=1e-12) or max(ss_errors)>1e-10 or max(f_errors)>1e-10:
        raise ValueError('Independent ANOVA SS/F reconstruction failed.')
    if not np.allclose(anova.partial_eta_squared,anova.F*anova.df_num/(anova.F*anova.df_num+anova.df_den),atol=1e-12):
        raise ValueError('Partial eta-squared identity failed.')
    order=['Group','Distractor presence','Session','Group × Distractor presence','Group × Session',
           'Distractor presence × Session','Group × Distractor presence × Session']
    anova=anova.set_index('effect').loc[order].reset_index()
    descriptives=data.groupby(['group','condition','session_id']).theta_db.agg(mean_db='mean',sd_db='std',n='count').reset_index()
    descriptives['sem_db']=descriptives.sd_db/np.sqrt(descriptives.n)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    anova.to_csv(out/'mixed_anova_2x2x5.csv',index=False);pd.DataFrame(errors).to_csv(out/'error_strata.csv',index=False)
    descriptives.to_csv(out/'descriptives.csv',index=False)
    qc=dict(n_subjects=32,n_per_group=16,n_observations=320,df_subject_within_group=30,
        model='Group between subjects; Distractor presence and intervention Session within subjects; all seven fixed effects.',
        total_SS=total,partition_SS=partition,SS_partition_error=partition_error,
        maximum_independent_SS_error=max(ss_errors),maximum_independent_F_error=max(f_errors),
        correction=correction, epsilon_estimates=epsilons if correction == 'GG' else [],
        primary_reporting=('Unadjusted numerator/denominator degrees of freedom and p values; no GG correction.' if correction is None else 'GG-adjusted degrees of freedom and p values for every term containing Session; epsilon=1 for remaining terms. Uncorrected results also retained.'),
        GG_method=('Not applied.' if correction is None else 'Pooled within-group residual covariance in each orthonormal repeated-contrast subspace; epsilon = tr(S)^2 / [k tr(S^2)].'),
        multiplicity_adjustment='None across the seven omnibus effects. GG addresses sphericity, not multiple comparisons.',
        missing_cells=0,participant_exclusions=0,posthoc_tests_run=False,
        methods_reference='https://search.r-project.org/CRAN/refmans/car/html/Anova.html')
    (out/'anova_validation.json').write_text(json.dumps(qc,indent=2)+'\n')
    print('2 × 2 × 5 mixed ANOVA complete: 32 participants, 320 cells; seven effects, separate subject-error strata.',flush=True)
    print('No GG correction; all p values and degrees of freedom are unadjusted.' if correction is None else 'Session terms use Greenhouse–Geisser correction; other terms have epsilon=1.',flush=True)
    columns=['effect','F','df_num','df_den','p_uncorrected','partial_eta_squared'] if correction is None else ['effect','F','df_num_GG','df_den_GG','p_uncorrected','p_GG','partial_eta_squared']
    print(anova[columns].to_string(index=False))
    return dict(anova=anova,descriptives=descriptives,error_strata=pd.DataFrame(errors),validation=qc)
