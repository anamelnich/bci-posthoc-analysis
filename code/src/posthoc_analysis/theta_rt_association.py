"""Exploratory association of pooled midline Laplacian theta and cleaned RT changes."""


from pathlib import Path


import hashlib


import json


import numpy as np


import pandas as pd


from scipy import stats


def analyze_theta_rt_changes(theta_dir, rt_dir, cache_dir, output_dir, paper_only=False):
    """Correlate post-minus-pre theta with pre-minus-post RT improvement."""
    import statsmodels.api as sm
    from .config import EXPECTED_SUBJECTS, get_subject_group
    theta_dir=Path(theta_dir);rt_dir=Path(rt_dir);cache_dir=Path(cache_dir);out=Path(output_dir)
    manifest=json.loads((theta_dir/'final_manifest.json').read_text())
    theta_path=theta_dir/'participant_window_power.csv'
    expected=[v for k,v in manifest['files_sha256'].items() if k.endswith('/participant_window_power.csv')]
    if len(expected)!=1 or hashlib.sha256(theta_path.read_bytes()).hexdigest()!=expected[0]:
        raise ValueError('Participant theta table checksum mismatch.')
    theta=pd.read_csv(theta_path);qc=json.loads((theta_dir/'validation.json').read_text())
    if qc['roi']!=['FZ','FCZ','CZ'] or qc['n_retained_trials']!=21940:
        raise ValueError('Expected the current pooled midline Laplacian theta measure.')
    mask_path=cache_dir/'trial_mask.csv';mask_qc=json.loads((cache_dir/'mask_validation.json').read_text())
    if hashlib.sha256(mask_path.read_bytes()).hexdigest()!=mask_qc['mask_sha256']:
        raise ValueError('Theta behavioral mask checksum mismatch.')
    mask=pd.read_csv(mask_path);rt_path=rt_dir/'trial_data_clean.csv';rt=pd.read_csv(rt_path)
    keys=['subject_id','session_id','run_id','trial']
    required=set(keys+['group','task','feedback','gdf_rt_ms','rt_ms'])
    if (not required.issubset(rt) or len(rt)!=21940 or rt.duplicated(keys).any()
            or len(mask)!=22980 or mask.retained.sum()!=21940 or mask.missing_status_response.sum()!=47
            or not rt.feedback.eq(1).all() or not rt.task.isin([0,1]).all()
            or not np.isfinite(rt.gdf_rt_ms).all() or rt.gdf_rt_ms.lt(150).any()
            or not np.allclose(rt.rt_ms,rt.gdf_rt_ms,rtol=0,atol=1e-10)):
        raise ValueError('Invalid cleaned EEG-Status RT table or theta eligibility mask.')
    rt['group']=rt.group.replace({'experimental':'bci'})
    aligned=rt[keys+['group','task','feedback','gdf_rt_ms']].merge(
        mask.loc[mask.retained,keys+['group','task','feedback','gdf_rt_ms']],on=keys,
        suffixes=('_rt','_theta'),how='outer',validate='one_to_one',indicator=True)
    if (not aligned._merge.eq('both').all()
            or any(not aligned[f'{c}_rt'].eq(aligned[f'{c}_theta']).all() for c in ['group','task','feedback'])
            or not np.allclose(aligned.gdf_rt_ms_rt,aligned.gdf_rt_ms_theta,rtol=0,atol=1e-10)):
        raise ValueError('Behavioral RT and theta do not use identical trial keys/values.')
    sessions=rt.groupby(['subject_id','group','session_id']).agg(mean_rt_ms=('gdf_rt_ms','mean'),n_rt_trials=('trial','size')).reset_index()
    merged=theta.merge(sessions,on=['subject_id','group','session_id'],validate='one_to_one',how='outer',indicator=True)
    if (len(merged)!=64 or not merged._merge.eq('both').all() or not merged.n_trials.eq(merged.n_rt_trials).all()
            or set(merged.subject_id)!=set(EXPECTED_SUBJECTS)
            or not merged.group.eq(merged.subject_id.map(get_subject_group)).all()
            or not np.isfinite(merged[['normalized_theta_db','mean_rt_ms']]).all().all()):
        raise ValueError('Unpaired subject/session theta and RT summaries.')
    wide=merged.pivot(index=['subject_id','group'],columns='session_id',values=['normalized_theta_db','mean_rt_ms'])
    if wide.shape!=(32,4) or wide.isna().any().any():raise ValueError('Expected 32 complete pre/post pairs.')
    changes=pd.DataFrame(dict(theta_pre_db=wide['normalized_theta_db'][1],theta_post_db=wide['normalized_theta_db'][5],
        rt_pre_ms=wide['mean_rt_ms'][1],rt_post_ms=wide['mean_rt_ms'][5])).reset_index()
    changes['theta_change_db']=changes.theta_post_db-changes.theta_pre_db
    changes['rt_change_ms']=changes.rt_post_ms-changes.rt_pre_ms
    # Keep the signed change for compatibility; the paper analyzes improvement.
    changes['rt_improvement_ms']=-changes.rt_change_ms
    if not np.allclose(changes.rt_change_ms, changes.rt_post_ms-changes.rt_pre_ms, rtol=0, atol=1e-10):
        raise ValueError('RT change must equal Session 5 minus Session 1.')
    if changes.groupby('group').size().to_dict()!={'bci':16,'control':16}:raise ValueError('Expected 16/group.')
    rows=[]
    scopes=['bci','control','pooled_unadjusted'] if paper_only else ['bci','control','pooled_unadjusted','group_adjusted']
    for scope in scopes:
        sub=changes if scope in ['pooled_unadjusted','group_adjusted'] else changes[changes.group==scope]
        x=sub.theta_change_db.to_numpy();y=sub.rt_improvement_ms.to_numpy();n=len(sub);k=0
        if scope=='group_adjusted':
            x=x-sub.groupby('group').theta_change_db.transform('mean').to_numpy()
            y=y-sub.groupby('group').rt_improvement_ms.transform('mean').to_numpy();k=1
        if min(np.std(x),np.std(y))<=0:raise ValueError('A change measure is constant.')
        r=float(stats.pearsonr(x,y).statistic);df=n-k-2
        t=r*np.sqrt(df/(1-r*r));p=float(2*stats.t.sf(abs(t),df))
        ci=np.tanh(np.arctanh(r)+np.array([-1,1])*stats.norm.ppf(.975)/np.sqrt(n-k-3))
        if k:
            design=sm.add_constant(np.c_[sub.theta_change_db,(sub.group=='bci').astype(float)])
        else:design=sm.add_constant(sub.theta_change_db.to_numpy())
        model=sm.OLS(sub.rt_improvement_ms.to_numpy(),design).fit()
        if not np.isclose(model.tvalues[1],t,atol=1e-10) or not np.isclose(model.pvalues[1],p,atol=1e-10):
            raise ValueError('Independent regression check disagrees with correlation test.')
        spearman=stats.spearmanr(x,y) if not k else None
        loo=[]
        for omit in ([] if paper_only else range(n)):
            left=sub.drop(sub.index[omit]);lx=left.theta_change_db;ly=left.rt_improvement_ms
            if k:
                lx=lx-left.groupby('group').theta_change_db.transform('mean')
                ly=ly-left.groupby('group').rt_improvement_ms.transform('mean')
            loo.append(float(stats.pearsonr(lx,ly).statistic))
        slope_ci=model.conf_int()[1]
        rows.append(dict(scope=scope,rt_difference='pre_minus_post',rt_measure='rt_improvement_ms',n=n,pearson_r=r,df=df,p_value=p,ci95_low=float(ci[0]),ci95_high=float(ci[1]),
            slope_ms_per_db=float(model.params[1]),slope_ci95_low=float(slope_ci[0]),slope_ci95_high=float(slope_ci[1]),
            spearman_rho=float(spearman.statistic) if spearman else np.nan,
            spearman_p=float(spearman.pvalue) if spearman else np.nan,
            leave_one_out_r_min=min(loo) if loo else np.nan,leave_one_out_r_max=max(loo) if loo else np.nan))
    results=pd.DataFrame(rows)
    validation=dict(rt_difference='pre_minus_post',rt_measure='rt_improvement_ms',n_subjects=32,n_per_group=16,n_trials=21940,identical_RT_theta_trial_keys=True,
        theta='Pooled-trial Fz/FCz/Cz Laplacian normalized theta, 4-8 Hz, 100-500 ms; post minus pre.',
        rt='Correct EEG-Status RTs >=150 ms and within one-pass participant/session +/-3 SD; improvement = pre minus post, positive = faster post-training responses.',
        compatibility_column='rt_change_ms remains post minus pre and equals -rt_improvement_ms; not used as the correlation or plot outcome.',
        pooling='Both conditions pooled within participant/session; trial-weighted session RT and linear-power pooling before theta dB.',
        tests='Two-sided exploratory Pearson correlations; group-adjusted partial correlation controls binary group (df=29). Within-group and pooled rank correlations are sensitivity checks.',
        inference='Unadjusted exploratory p values and pointwise approximate Fisher-z 95% correlation CIs; no mediation or causality test.',
        regression_validation='Independent OLS reproduces all correlation t statistics and p values.',
        no_new_exclusions=True,input_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [theta_path,mask_path,rt_path]})
    if paper_only:
        validation['tests']='PDF scope: two-sided within-group and pooled Pearson correlations; within-group/pooled Spearman sensitivity. No group-adjusted or leave-one-out analyses.'
    out.mkdir(parents=True,exist_ok=True)
    merged.drop(columns='_merge').to_csv(out/'participant_sessions.csv',index=False)
    changes.to_csv(out/'participant_changes.csv',index=False);results.to_csv(out/'correlations.csv',index=False)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('Theta change is post minus pre; RT improvement is pre minus post (positive = faster responses).')
    print('Matched all 21,940 cleaned RT/theta trial keys; 32 paired participants; no additional exclusions.')
    print(results[['scope','n','pearson_r','p_value','ci95_low','ci95_high','spearman_rho','spearman_p']].to_string(index=False))
    return changes,results,validation


def plot_theta_rt_changes(changes, results, output_dir, pooled_only=False):
    """Plot post-minus-pre theta versus pre-minus-post RT improvement."""
    import matplotlib.pyplot as plt
    required = {'subject_id', 'group', 'theta_change_db', 'rt_change_ms', 'rt_improvement_ms', 'rt_pre_ms', 'rt_post_ms'}
    if not isinstance(pooled_only, bool):
        raise ValueError('pooled_only must be a boolean.')
    scopes = ['pooled_unadjusted'] if pooled_only else ['bci', 'control', 'pooled_unadjusted']
    if (not required.issubset(changes) or len(changes) != 32
            or changes.subject_id.duplicated().any()
            or changes.groupby('group').size().to_dict() != {'bci': 16, 'control': 16}
            or 'rt_difference' not in results or not results.rt_difference.eq('pre_minus_post').all()
            or 'rt_measure' not in results or not results.rt_measure.eq('rt_improvement_ms').all()
            or results.scope.duplicated().any() or not set(scopes).issubset(results.scope)
            or not np.isfinite(changes[['theta_change_db', 'rt_change_ms', 'rt_improvement_ms', 'rt_pre_ms', 'rt_post_ms']]).all().all()
            or not np.allclose(changes.rt_improvement_ms, changes.rt_pre_ms-changes.rt_post_ms, rtol=0, atol=1e-10)
            or not np.allclose(changes.rt_change_ms, changes.rt_post_ms-changes.rt_pre_ms, rtol=0, atol=1e-10)):
        raise ValueError('Plot requires 32 paired participants and pre-minus-post RT improvement results; rerun the analysis.')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    rc={'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
        'font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}
    colors={'bci':'#DD8452','control':'#4C72B0'}
    names={'bci':'BCI','control':'Mental rehearsal','pooled_unadjusted':'Both groups pooled'}
    with plt.rc_context(rc):
        fig,axes=plt.subplots(1,len(scopes),figsize=(3.5,3.5) if pooled_only else (7.05,3.25),
                              sharex=True,sharey=True,squeeze=False)
        axes=axes.ravel()
        fig.subplots_adjust(left=.18 if pooled_only else .10,right=.98,bottom=.28 if pooled_only else .25,
                            top=.86 if pooled_only else .80,wspace=.25)
        for i,(ax,scope) in enumerate(zip(axes,scopes)):
            sub=changes if scope=='pooled_unadjusted' else changes[changes.group==scope]
            res=results.set_index('scope').loc[scope]
            x=sub.theta_change_db;y=sub.rt_improvement_ms
            fit=stats.linregress(x,y)
            if (res.n != len(sub) or not np.isclose(fit.rvalue,res.pearson_r,rtol=0,atol=1e-10)
                    or not np.isclose(fit.pvalue,res.p_value,rtol=0,atol=1e-10)):
                raise ValueError(f'{scope}: plotted participants disagree with correlation results.')
            for group in ['bci','control']:
                points=sub[sub.group==group]
                if not points.empty:
                    ax.scatter(points.theta_change_db,points.rt_improvement_ms,s=18,color=colors[group],
                               edgecolor='white',linewidth=.4,zorder=3,label=names[group])
            grid=np.linspace(x.min(),x.max(),100)
            ax.plot(grid,fit.intercept+fit.slope*grid,color=colors.get(scope,'.25'),linewidth=1)
            ax.axhline(0,color='.6',linewidth=.6,linestyle='--');ax.axvline(0,color='.6',linewidth=.6,linestyle='--')
            title=(f'n = {len(sub)}, Pearson r = {res.pearson_r:.3f}, p = {res.p_value:.4f}' if pooled_only
                   else names[scope]+f' (n = {len(sub)})\nr = {res.pearson_r:.3f}, p = {res.p_value:.3f}')
            ax.set_title(title,fontsize=7 if pooled_only else 8)
            ax.set_xlabel('Post − pre theta (dB)')
            ax.spines[['top','right']].set_visible(False);ax.tick_params(direction='out',width=.6)
            if not pooled_only:
                ax.text(-.15,1.20,chr(97+i),transform=ax.transAxes,fontsize=9,fontweight='bold')
        axes[0].set_ylabel('RT improvement: pre − post (ms)')
        handles,labels=axes[-1].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.55,.10 if pooled_only else .075),ncol=2,frameon=False,fontsize=7)
        note=('Positive RT improvement indicates faster responses.\nPooled fit and two-sided p are unadjusted for group.' if pooled_only
              else 'Positive RT improvement indicates faster responses; pooled fit is unadjusted for group.')
        fig.text(.55,.035,note,ha='center',fontsize=6.5)
        stem=out/('theta_rt_change_association_pooled' if pooled_only else 'theta_rt_change_association');paths={ext:str(stem.with_suffix('.'+ext).resolve()) for ext in ['pdf','png','svg']}
        for ext,path in paths.items():fig.savefig(path,dpi=350,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    print(f'Saved {len(scopes)} theta-change/RT-improvement scatterplot panel(s): 16 BCI (orange), 16 mental rehearsal (blue); no omitted points.')
    return paths

