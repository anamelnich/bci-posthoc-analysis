"""Compact pooled-trial Laplacian theta maps across training sessions."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def plot_pooled_laplacian_theta_sessions(cache_dir, output_dir, color_limit_db=1.5,
                                        spline_regularization=1e-5, band='theta',
                                        change_color_limit_db=None, condition='both',
                                        pdf_pages=None, electrode_labels=None,
                                        group_panels=False, session_term='training',
                                        electrode_markers_only=False,
                                        selected_electrodes_only=False):
    """Plot two groups by pre/post/change, reusing validated pooled-power data.

    The cached 'both' condition pools retained trials in linear power before
    dB normalization. This function never recomputes EEG or changes its montage.
    An optional change limit gives pre/post and change separate labeled scales.
    electrode_labels selects display names only; all 61 channels enter each map.
    electrode_markers_only shows dots only at the selected electrodes, without names.
    selected_electrodes_only hides dots outside electrode_labels while preserving names.
    group_panels separates pre/post (a) from change (b); session_term controls titles.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects
    import mne
    from .config import EXPECTED_SUBJECTS, get_subject_group
    from .spherical_topography import prepare_spherical_topography, spherical_spline_grid
    from .theta_topography import _space_theta_electrode_labels

    root=Path(cache_dir).resolve();out=Path(output_dir).resolve()
    if (not np.isfinite(color_limit_db) or color_limit_db<=0
            or not np.isfinite(spline_regularization) or spline_regularization<0):
        raise ValueError('Require positive color limit and nonnegative spline regularization.')
    change_limit=color_limit_db if change_color_limit_db is None else change_color_limit_db
    if not np.isfinite(change_limit) or change_limit<=0:
        raise ValueError('Require a positive finite change color limit.')
    split_scales=change_limit!=color_limit_db
    limits={'pre':float(color_limit_db),'post':float(color_limit_db),'change':float(change_limit)}
    print(f'Display limits (dB): pre/post ±{color_limit_db:g}; change ±{change_limit:g}.')
    manifest=json.loads((root/'final_manifest.json').read_text())
    hashes={}
    for name in ('settings.json','participant_session_topographies.csv','mask_validation.json'):
        matches=[v for k,v in manifest['files_sha256'].items() if Path(k).name==name]
        digest=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if len(matches)!=1 or digest!=matches[0]:raise ValueError(f'Cache integrity mismatch: {name}')
        hashes[name]=digest
    settings=json.loads((root/'settings.json').read_text())
    bands={'theta':(4.,8.),'alpha':(8.,12.)}
    if band not in bands or not np.array_equal(settings['frequencies_hz'],
            np.arange(bands[band][0],bands[band][1]+.01,.5)):
        raise ValueError('Requested band label does not match the cached frequency grid.')
    frequency_label=f'{bands[band][0]:g}–{bands[band][1]:g} Hz'
    stroop=settings.get('task')=='stroop'
    target=np.asarray(settings['target_s'],dtype=float)
    if (target.shape!=(2,) or not np.isfinite(target).all()
            or not 0<=target[0]<target[1]<=settings['epoch_s'][1]):
        raise ValueError('Invalid cached poststimulus target window.')
    target_label=f'{1000*target[0]:g}–{1000*target[1]:g} ms'
    print(f'Plotting cached {band} over {target_label}.')
    side_conditions={'right_distractor','left_distractor','no_distractor'}
    side_split=settings.get('split_distractor_side',False)
    expected_conditions=({'both','congruent','incongruent'} if stroop else
                         side_conditions if side_split else {'both','distractor','no_distractor'})
    contrast=stroop and condition=='incongruent_minus_congruent'
    if condition not in expected_conditions and not contrast:
        raise ValueError('Requested condition is not available in this cache.')
    condition_label={'incongruent_minus_congruent':'Incongruent − congruent','both':'All retained trials','congruent':'Congruent','incongruent':'Incongruent','distractor':'Distractor',
        'right_distractor':'Right distractor','left_distractor':'Left distractor',
        'no_distractor':'No distractor'}[condition]
    mask=json.loads((root/'mask_validation.json').read_text())
    if (not settings.get('surface_laplacian') or not settings.get('rt_cleaned')
            or (not stroop and settings['target_s']!=[.1,.5]) or settings['baseline_s']!=[-.3,-.1]
            or settings['surface_laplacian']['montage']!='standard_1020'):
        raise ValueError('Expected the existing RT-screened, template-based Laplacian theta cache.')
    table=pd.read_csv(root/'participant_session_topographies.csv')
    keys=['subject_id','session_id','condition','channel']
    required=set(keys+['group','theta_db','n_trials'])
    labels=sorted(settings['surface_laplacian']['channels'])
    if session_term not in ('training', 'intervention') or not isinstance(group_panels, bool):
        raise ValueError('Use training/intervention session terminology and a boolean group_panels.')
    display_labels=labels if electrode_labels is None else list(electrode_labels)
    if (not display_labels or any(not isinstance(label, str) for label in display_labels)
            or len({label.casefold() for label in display_labels}) != len(display_labels)
            or not {label.casefold() for label in display_labels}.issubset(
                {label.casefold() for label in labels})):
        raise ValueError('Electrode labels must be unique recorded channel names (case insensitive).')
    if not isinstance(electrode_markers_only, bool) or not isinstance(selected_electrodes_only, bool):
        raise ValueError('Electrode display flags must be boolean.')
    label_lookup={label.casefold():label for label in display_labels}
    map_names=[label_lookup.get(label.casefold(), '') for label in labels]
    selected_mask=np.array([label.casefold() in label_lookup for label in labels])
    print(f'Display {"dots only" if electrode_markers_only else "labels"} ({len(display_labels)}/{len(labels)}): {", ".join(display_labels)}; all channels retained in maps.')
    if (not required.issubset(table) or len(table)!=32*2*3*61
            or table.duplicated(keys).any() or table[list(required)].isna().any().any()
            or not np.isfinite(table.theta_db).all() or (table.n_trials<=0).any()
            or set(table.channel)!=set(labels) or len(labels)!=61
            or set(table.condition)!=expected_conditions
            or set(table.subject_id)!=set(EXPECTED_SUBJECTS)):
        raise ValueError('Incomplete, duplicate, or invalid cached theta values.')
    if any(g!=get_subject_group(s) for s,g in table[['subject_id','group']].drop_duplicates().itertuples(index=False,name=None)):
        raise ValueError('Subject group mismatch.')
    if contrast:
        identity=['subject_id','group','session_id','channel']
        paired_conditions=table[table.condition.isin(['congruent','incongruent'])].pivot(
            index=identity,columns='condition',values='theta_db')
        counts=table[table.condition.isin(['congruent','incongruent'])].pivot(
            index=identity,columns='condition',values='n_trials')
        if (len(paired_conditions)!=32*2*61 or paired_conditions.isna().any().any()
                or counts.isna().any().any()):
            raise ValueError('Condition contrast requires paired congruent/incongruent values at every channel.')
        pooled=(paired_conditions['incongruent']-paired_conditions['congruent']).rename('theta_db').reset_index()
        pooled['n_trials']=counts.sum(axis=1).to_numpy()
        pooled['condition']=condition
        print('Contrast: incongruent minus congruent dB within each participant/session/channel; equal participant weights.')
    else:
        pooled=table[table.condition==condition].copy()
    cells=pooled.groupby(['subject_id','group','session_id'])
    if len(cells)!=64 or not cells.size().eq(61).all() or not cells.n_trials.nunique().eq(1).all():
        raise ValueError('Pooled cells must contain 61 channels and consistent trial counts.')
    n_trials=int(cells.n_trials.first().sum())
    cohort_trials=int(table.groupby(['subject_id','session_id','condition']).n_trials.first().sum()) if side_split else int(table[table.condition=='both'].groupby(['subject_id','session_id']).n_trials.first().sum())
    expected_trials=settings['n_retained'] if stroop else 21940
    if cohort_trials!=expected_trials or cohort_trials!=mask['n_retained']:
        raise ValueError('Pooled trial counts differ from the validated behavioral mask.')
    paired=pooled.pivot(index=['subject_id','group','channel'],columns='session_id',values='theta_db')
    if set(paired.columns)!={1,5} or paired.isna().any().any():raise ValueError('Unpaired sessions.')
    values=paired.rename(columns={1:'pre',5:'post'}).reset_index()
    values['change']=values['post']-values['pre']
    groups=['bci','control'];comparisons=['pre','post','change'];maps=[];records=[]
    for group in groups:
        sub=values[values.group==group]
        if sub.subject_id.nunique()!=16 or len(sub)!=16*61:raise ValueError('Expected 16 participants/group.')
        means=sub.groupby('channel')[comparisons].mean().reindex(labels)
        maps.append(means.to_numpy().T)
        for comparison in comparisons:
            records.extend(dict(group=group,comparison=comparison,channel=ch,value_db=float(value))
                           for ch,value in means[comparison].items())
    maps=np.asarray(maps)
    error=float(np.max(abs(maps[:,2]-(maps[:,1]-maps[:,0]))))
    if maps.shape!=(2,3,61) or error>1e-12:raise ValueError('Group/session map reconstruction failed.')
    print(f'Validated pooled maps: 32 paired participants, 16/group, {n_trials:,} retained trials; change error {error:.3g} dB.')

    info=mne.create_info(labels,512,'eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'),match_case=False)
    info,directions=prepare_spherical_topography(info)
    rc={'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
        'font.size':7,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}
    out.mkdir(parents=True,exist_ok=True)
    condition_stem='pooled' if condition=='both' else condition
    task_prefix='stroop_' if stroop else ''
    stem=out/f'{task_prefix}{band}_laplacian_{condition_stem}_pre_post_change'
    paths={ext:str(stem.with_suffix('.'+ext)) for ext in ('pdf','svg','png')}
    checks=[];weights=None;images={}
    with plt.rc_context(rc):
        if group_panels:
            fig=plt.figure(figsize=(7.05,5.5 if split_scales else 4.9))
            grid=fig.add_gridspec(2,4,width_ratios=[1,1,.12,1],left=.17,right=.87,
                bottom=.26 if split_scales else .16,top=.87,wspace=.15,hspace=.16)
            axes=np.array([[fig.add_subplot(grid[i,j]) for j in (0,1,3)] for i in range(2)])
        else:
            fig,axes=plt.subplots(2,3,figsize=(7.05,5.5 if split_scales else 4.9))
            fig.subplots_adjust(left=.17,right=.86,bottom=.26 if split_scales else .16,top=.87,wspace=.15,hspace=.16)
        for i,group in enumerate(groups):
            for j,comparison in enumerate(comparisons):
                ax=axes[i,j];v=maps[i,j]
                im,_=mne.viz.plot_topomap(v,info,axes=ax,show=False,cmap='RdBu_r',
                    vlim=(-limits[comparison],limits[comparison]),contours=0,sensors=not electrode_markers_only,
                    names=None if electrode_markers_only else map_names,
                    mask=selected_mask if electrode_markers_only else None,
                    mask_params=dict(marker='o',markerfacecolor='black',markeredgecolor='black',
                                     markersize=1.5,markeredgewidth=0,linestyle='None'),
                    res=128,image_interp='linear',sphere=(0.,0.,0.,.095))
                images[comparison]=im
                shape=im.get_array().shape
                if weights is None:
                    weights,node_weights,spline_qc=spherical_spline_grid(
                        directions,im.get_extent(),shape,regularization=spline_regularization)
                xmin,xmax,ymin,ymax=im.get_extent()
                gx,gy=np.meshgrid(np.linspace(xmin,xmax,shape[1]),np.linspace(ymin,ymax,shape[0]))
                clip=im.get_clip_path()
                if clip is None:raise ValueError('Missing scalp clipping boundary.')
                visible=clip.get_fully_transformed_path().contains_points(
                    ax.transData.transform(np.c_[gx.ravel(),gy.ravel()])).reshape(shape)
                pixels=(weights@v).reshape(shape)
                if not visible.any() or not np.isfinite(pixels).all():raise ValueError('Invalid interpolated map.')
                im.set_data(np.ma.array(pixels,mask=~visible));im.set_interpolation('nearest')
                for t in ax.texts:
                    if t.get_text() in display_labels:
                        t.set_fontsize(5.0 if electrode_labels is not None else 4.3);t.set_color('black')
                        t.set_path_effects([effects.withStroke(linewidth=.65,foreground='white')])
                for marker in ax.collections:
                    marker.set_sizes([1.])
                    if selected_electrodes_only and not electrode_markers_only:
                        positions=marker.get_offsets()
                        if positions.shape!=(len(labels),2):
                            raise ValueError('Sensor dots must align with all recorded channels before selection.')
                        marker.set_offsets(positions[selected_mask])
                if not electrode_markers_only:
                    _space_theta_electrode_labels(ax,display_labels)
                if not group_panels:
                    ax.text(-.03,1.03,chr(97+3*i+j),transform=ax.transAxes,fontsize=9,fontweight='bold')
                if i==0:ax.set_title([f'Pre-{session_term}',f'Post-{session_term}','Post − pre'][j],fontsize=8,pad=10)
                checks.append(dict(group=group,comparison=comparison,
                    color_limit_db=limits[comparison],
                    fraction_visible_pixels_outside_scale=float(np.mean(abs(pixels[visible])>limits[comparison])),
                    electrode_min_db=float(v.min()),electrode_max_db=float(v.max()),
                    rendered_min_db=float(pixels[visible].min()),rendered_max_db=float(pixels[visible].max()),
                    maximum_regularized_node_deviation_db=float(np.max(abs(node_weights@v-v)))))
        if group_panels:
            from matplotlib.lines import Line2D
            first=axes[0,0].get_position();post=axes[0,1].get_position()
            change=axes[0,2].get_position();bottom=axes[1,2].get_position()
            divider_x=(post.x1+change.x0)/2
            fig.add_artist(Line2D([divider_x,divider_x],[bottom.y0-.01,change.y1+.035],
                transform=fig.transFigure,color='.75',linewidth=.6))
            for label,pos in [('a',first),('b',change)]:
                fig.text(pos.x0-.006,pos.y1+.025,label,fontsize=9,fontweight='bold',va='bottom')
            print('Panel a: pre/post for both groups; panel b: paired post–pre changes; gray divider.')
        for i,name in enumerate(['BCI\n(n = 16)','Mental rehearsal\n(n = 16)']):
            pos=axes[i,0].get_position();fig.text(.012,(pos.y0+pos.y1)/2,name,ha='left',va='center',fontsize=8)
        fig.text(.52,.965,f'{"Stroop | " if stroop else ""}Laplacian {band} | {frequency_label} | {target_label} | {condition_label}',ha='center',fontsize=8)
        # Preserve the requested scales, but explicitly mark any clipped map
        # colors. Use the rendered scalp pixels, which are the colorbar's data.
        colorbar_extensions={}
        for scale_name,comparisons_for_scale in (
                [('pre',['pre','post']),('change',['change'])] if split_scales
                else [('shared',comparisons)]):
            selected_checks=[check for check in checks
                             if check['comparison'] in comparisons_for_scale]
            below=any(check['rendered_min_db'] < -check['color_limit_db']
                      for check in selected_checks)
            above=any(check['rendered_max_db'] > check['color_limit_db']
                      for check in selected_checks)
            colorbar_extensions[scale_name]=('both' if below and above else
                                             'min' if below else 'max' if above else 'neither')
            print(f'{scale_name} colorbar overflow: {colorbar_extensions[scale_name]}; '
                  'fixed display limits unchanged.')
        if split_scales:
            for comparison,position,label in [
                    ('pre',[.17,.17,.44,.018],'Pre/post: incongruent − congruent (dB)' if contrast else 'Pre/post: baseline-normalized power (dB)'),
                    ('change',[.67,.17,.19,.018],'Post − pre change (dB)')]:
                limit=limits[comparison];cax=fig.add_axes(position)
                cb=fig.colorbar(images[comparison],cax=cax,orientation='horizontal',ticks=[-limit,0,limit],
                                extend=colorbar_extensions[comparison])
                cb.set_label(label,fontsize=7,labelpad=3)
                cb.ax.tick_params(labelsize=6,width=.6,length=2,pad=2)
        else:
            cax=fig.add_axes([.91,.24,.018,.52]);cb=fig.colorbar(im,cax=cax,ticks=[-color_limit_db,0,color_limit_db],
                                                            extend=colorbar_extensions['shared'])
            cb.set_label('Incongruent − congruent / post–pre change (dB)' if contrast else 'Baseline-normalized power / post–pre change (dB)',fontsize=7)
            cb.ax.tick_params(labelsize=6,width=.6)
        baseline_caption=('Within-participant dB contrast; common baseline: −300 to −100 ms' if contrast else
            'Baseline: −300 to −100 ms, pooled across retained congruent and incongruent trials' if stroop else
            'Baseline: −300 to −100 ms, pooled across right, left and no-distractor trials' if side_split else
            'Baseline: −300 to −100 ms; distractor and no-distractor trials pooled before dB normalization')
        fig.text(.52,.06 if split_scales else .095,baseline_caption,ha='center',fontsize=6.3)
        fig.text(.52,.025 if split_scales else .06,f'Same RT-screened trials and template geometry; regularized spherical display (λ = {spline_regularization:g})',ha='center',fontsize=6.3)
        for ext,path in paths.items():fig.savefig(path,dpi=350,bbox_inches='tight',facecolor='white')
        if pdf_pages is not None:pdf_pages.savefig(fig,dpi=350,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    pd.DataFrame(records).to_csv(stem.with_suffix('.csv'),index=False)
    metadata=dict(paths=paths,band=band,target_s=settings['target_s'],baseline_s=settings['baseline_s'],frequencies_hz=settings['frequencies_hz'],signal_space='laplacian',condition=condition,n_subjects=32,n_per_group=16,
        n_retained_trials=n_trials,color_limit_db=float(color_limit_db),montage='standard_1020',
        n_cohort_retained_trials=cohort_trials,
        color_limits_db=limits,separate_change_scale=bool(split_scales),
        colorbar_extensions=colorbar_extensions,
        electrode_labels=[] if electrode_markers_only else display_labels,
        electrode_markers=display_labels if electrode_markers_only or selected_electrodes_only else labels,
        selected_electrodes_only=selected_electrodes_only,
        electrode_markers_only=electrode_markers_only,all_channels_used_for_maps=labels,
        session_term=session_term,group_panels=group_panels,
        panel_layout={'a':['pre','post'],'b':['change']} if group_panels else 'individual a–f',
        maximum_change_reconstruction_error_db=error,interpolation=spline_qc,panel_checks=checks,
        input_sha256=hashes,figure_sha256={ext:hashlib.sha256(Path(p).read_bytes()).hexdigest() for ext,p in paths.items()},
        pooling='Retained trial linear power pooled before dB; equal participant weights; within-participant Session 5 minus Session 1.',
        scope='Display of cached results only; no EEG, montage, or statistics recomputed.')
    if contrast:
        participant_path=stem.with_name(stem.name+'_participants').with_suffix('.csv')
        values.to_csv(participant_path,index=False)
        metadata.update(contrast_definition='Within participant/session/channel: incongruent dB minus congruent dB; change = contrast S5 minus contrast S1.',
            pooling='Condition dB values subtracted within participant; equal participant weights within group.',
            participant_values_path=str(participant_path),
            participant_values_sha256=hashlib.sha256(participant_path.read_bytes()).hexdigest())
    stem.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Saved six-panel pooled Laplacian {band} figure (PDF, SVG, PNG); original EEG values and montage unchanged.')
    return metadata
