"""Full-scalp theta maps with a pooled prestimulus power baseline.

User-requested descriptive analysis, 2026-10-15. No voltage baseline,
amplitude rejection, ERP subtraction, ICA, or additional spatial reference.
"""


from pathlib import Path


from concurrent.futures import ThreadPoolExecutor


import hashlib


import json


import numpy as np


import pandas as pd


from .config import EXPECTED_SUBJECTS, PROJECT_ROOT, get_subject_group


from .analysis import load_training_analysis_file


from .triggers import load_training_trigger_file


from .eeg import _get_training_run_gdf_files_for_session


def theta_topography_settings(epoch_s=(-1., 1.5), reference_channel='CPZ'):
    """Return explicit parameters for the requested 4-8 Hz, 100-500 ms maps."""
    frequencies = np.arange(4., 8.01, .5)
    # Interpolate the paper's logarithmic 3-to-10-cycle / 2-to-60-Hz rule.
    cycles = 3 * (frequencies / 2) ** (np.log(10 / 3) / np.log(60 / 2))
    if tuple(epoch_s) not in ((-1.,1.5),(-1.5,1.5)):
        raise ValueError('Supported epochs: paper -1 to +1.5 s or previous -1.5 to +1.5 s.')
    if reference_channel not in ('CPZ','FCZ'):
        raise ValueError('CPZ is the corrected reference; FCZ is supported only for historical reproduction.')
    settings = dict(version=3, sfreq=512, filter_hz=[.1, 20.],
        epoch_s=list(epoch_s), baseline_s=[-.3, -.1], target_s=[.1, .5],
        maximum_boundary_kernel_energy=1e-8,
        frequencies_hz=frequencies.tolist(), n_cycles=cycles.tolist(),
        zero_mean_wavelets=True, voltage_baseline=None, amplitude_rejection=None,
        trial_selection='all outcomes, non-practice assessment training runs',
        reference=f'recorded {reference_channel}; no additional rereferencing',
        reference_channel=reference_channel,
        expected_scalp_channels=61 if reference_channel=='CPZ' else 60,
        omit_channels=['M1', 'M2', 'EOG', 'sens7', 'sens8', 'Status', reference_channel],
        normalization='10*log10(condition mean single-trial power / pooled baseline power)',
        baseline_pool='all trials of both conditions within participant/session/channel/frequency',
        averaging='mean linear power across trials, then dB, then mean across time and frequencies; equal participants',
        source_doi='10.1152/jn.00479.2013',
        method_scope='paper-style wavelets and dB baseline; not a reproduction of its artifact correction or reference')
    if len(frequencies) != 9 or not np.all(np.diff(cycles) > 0):
        raise ValueError('Invalid theta frequency/cycle grid.')
    print(f'Theta settings: epochs {list(epoch_s)} s; 4-8 Hz; 100-500 ms; pooled -300 to -100 ms power baseline; no amplitude rejection or voltage baseline.')
    return settings


def validate_theta_inventory(inventory_csv, project_root=PROJECT_ROOT, run_overrides=None):
    """Validate cohort, source inventory, paired files and fresh run discovery."""
    frame = pd.read_csv(inventory_csv)
    needed = {'subject_id', 'group', 'session_id', 'run_id', 'source', 'status'}
    if not needed.issubset(frame) or frame.duplicated(['subject_id','session_id','run_id']).any():
        raise ValueError('Missing inventory fields or duplicate run identifiers.')
    if set(frame.subject_id) != set(EXPECTED_SUBJECTS) or set(frame.session_id) != {1,5}:
        raise ValueError('Expected all configured participants and Sessions 1/5.')
    if not frame.status.eq('pass').all():
        raise ValueError('Source inventory contains failed runs.')
    overrides = {('e30', 1): 7} if run_overrides is None else run_overrides
    for subject in EXPECTED_SUBJECTS:
        for session in (1,5):
            cell = frame[(frame.subject_id == subject) & (frame.session_id == session)].sort_values('run_id')
            expected = overrides.get((subject, session), {1:8,5:4}[session])
            files = _get_training_run_gdf_files_for_session(subject, session,
                project_root=project_root, allow_incomplete=(subject,session) in overrides)
            if len(cell) != expected or cell.run_id.tolist() != list(range(1,expected+1)):
                raise ValueError(f'{subject}/S{session}: expected {expected} ordered runs.')
            if cell.source.tolist() != [str(p) for p in files] or not cell.group.eq(get_subject_group(subject)).all():
                raise ValueError(f'{subject}/S{session}: source discovery/group assignment mismatch.')
    print(f'Validated inventory: {len(frame)} runs, {len(EXPECTED_SUBJECTS)} participants, 60 trials/run; documented e30/S1 seven-run exception.')
    return frame.sort_values(['subject_id','session_id','run_id']).reset_index(drop=True)


def extract_theta_run(gdf_path, settings, trial_mask=None, expected_rt_ms=None):
    """Return linear-power sufficient statistics; preserve every valid trial.

    Arrays: baseline_sum (channel, frequency), post_sum
    (condition [no-distractor, distractor], channel, frequency, time).
    Baseline_sum sums per-trial temporal-mean power, not log power.
    """
    import mne
    path = Path(gdf_path)
    task=settings.get('source_task','training')
    if task not in ('training','decoding') or not path.name.endswith(f'_{task}.gdf') or 'practice' in str(path).lower():
        raise ValueError(f'Not a non-practice {task} run: {path}')
    if task=='decoding':
        from .bci import load_decoding_analysis_file
        analysis=load_decoding_analysis_file(path.with_suffix('.analysis.txt'))
        if trial_mask is not None and not np.asarray(trial_mask).all():
            raise ValueError('Intervention theta must retain all trials.')
    else:
        analysis=load_training_analysis_file(path.with_suffix('.analysis.txt'))
    triggers=load_training_trigger_file(path.with_suffix('.triggers.txt'),allow_zero_latency_response=(task=='decoding'))
    raw = mne.io.read_raw_gdf(path, preload=True, verbose='ERROR')
    fs = settings['sfreq']
    if raw.info['sfreq'] != fs or len(raw.ch_names) != 67 or len(set(raw.ch_names)) != 67:
        raise ValueError(f'{path}: expected 512 Hz / 67 unique recorded channels.')
    events = mne.find_events(raw, stim_channel='Status', shortest_event=1, verbose='ERROR')
    stimuli = events[np.isin(events[:,2], [8,32,44])]
    expected = analysis.distractor_position.map({0:8,2:32,4:44}).to_numpy()
    text_codes = triggers.loc[triggers.trigger.isin([8,32,44]), 'trigger'].to_numpy()
    if stimuli.shape != (60,3) or not np.array_equal(stimuli[:,2],expected) or not np.array_equal(expected,text_codes):
        raise ValueError(f'{path}: Status/text/analysis stimulus order mismatch.')
    if not analysis.task.eq(analysis.distractor_position.ne(0).astype(int)).all():
        raise ValueError(f'{path}: task/position mismatch.')
    if dict(zip(*np.unique(expected,return_counts=True))) != {8:30,32:15,44:15}:
        raise ValueError(f'{path}: expected 30 no-distractor and 15/15 distractor-side trials.')
    keep=np.ones(60,dtype=bool) if trial_mask is None else np.asarray(trial_mask)
    if keep.shape!=(60,) or keep.dtype!=bool or not keep.any():
        raise ValueError('trial_mask must select at least one of the 60 trials with boolean values.')
    if expected_rt_ms is not None:
        rt=np.full(60,np.nan)
        for i,onset in enumerate(stimuli[:,0]):
            end=stimuli[i+1,0] if i<59 else raw.first_samp+raw.n_times
            response=events[(events[:,2]==64)&(events[:,0]>onset)&(events[:,0]<end),0]
            if len(response)>1:raise ValueError(f'{path}: multiple responses in trial {i+1}.')
            if len(response):rt[i]=(response[0]-onset)/fs*1000
        if np.shape(expected_rt_ms)!=(60,) or not np.allclose(rt,expected_rt_ms,equal_nan=True,rtol=0,atol=1e-10):
            raise ValueError(f'{path}: freshly extracted Status RTs disagree with the trial mask source.')
    fixation = events[events[:,2] == 4,0]
    fix_idx = np.searchsorted(fixation, stimuli[:,0], side='left') - 1
    if (fix_idx < 0).any() or ((stimuli[:,0]-fixation[fix_idx])/fs < .9).any():
        raise ValueError(f'{path}: missing/short prestimulus fixation.')
    labels = [c for c in raw.ch_names if c not in settings['omit_channels']]
    n_channels=settings['expected_scalp_channels']
    if len(labels) != n_channels or (settings['reference_channel']=='CPZ' and ('CPZ' in raw.ch_names or 'FCZ' not in labels)):
        raise ValueError(f'{path}: unexpected scalp/reference-channel inventory.')
    data = raw.get_data(picks=labels)
    if not np.isfinite(data).all():
        raise ValueError(f'{path}: nonfinite EEG samples.')
    filtered = mne.filter.filter_data(data, fs, *settings['filter_hz'],
        method='fir', phase='zero', fir_design='firwin', verbose='ERROR')
    spatial_qc=None
    if settings.get('surface_laplacian'):
        from .theta_laplacian import apply_theta_laplacian
        filtered,spatial_qc=apply_theta_laplacian(filtered,labels,settings['surface_laplacian'])
    # Compute the spatial transform using all scalp channels, then optionally
    # restrict wavelet power to the requested ROI; the underlying signal is identical.
    if 'power_channels' in settings:
        selected=settings['power_channels']
        if not selected or len(set(selected))!=len(selected) or not set(selected).issubset(labels):
            raise ValueError('Power ROI must contain distinct recorded scalp electrodes.')
        filtered=filtered[[labels.index(ch) for ch in selected]]
        labels=list(selected);n_channels=len(labels)
    offsets = np.arange(round(settings['epoch_s'][0]*fs), round(settings['epoch_s'][1]*fs)+1)
    indices = stimuli[:,0,None] - raw.first_samp + offsets[None,:]
    if indices.min() < 0 or indices.max() >= raw.n_times:
        raise ValueError(f'{path}: insufficient real-data epoch padding.')
    indices=indices[keep]
    tasks=analysis.task.to_numpy()[keep]
    counts=np.array([(tasks==c).sum() for c in (0,1)])
    times = offsets / fs
    bmask = (times >= settings['baseline_s'][0]) & (times <= settings['baseline_s'][1])
    pmask = (times >= settings['target_s'][0]) & (times <= settings['target_s'][1])
    if bmask.sum() != 102 or pmask.sum() != 205:
        raise ValueError('Unexpected baseline/target sample counts at 512 Hz.')
    freqs = np.array(settings['frequencies_hz']); cycles = np.array(settings['n_cycles'])
    kernels = mne.time_frequency.morlet(fs, freqs, n_cycles=cycles, zero_mean=settings['zero_mean_wavelets'])
    half_support = max((len(w)-1)//2 for w in kernels)
    # The paper's -1 s start truncates only tiny outer tails of the 4/4.5 Hz
    # kernels at the earliest baseline centers. Explicitly quantify this;
    # standard FFT convolution uses zero outside the real epoch, not invented EEG.
    edge_fractions=[]
    for wave in kernels:
        lag=np.arange(-(len(wave)//2),len(wave)//2+1)
        for mask in (bmask,pmask):
            selected=np.flatnonzero(mask)
            for center in (selected[0],selected[-1]):
                outside=((center-lag)<0)|((center-lag)>=len(times))
                edge_fractions.append(float(np.sum(abs(wave[outside])**2)/np.sum(abs(wave)**2)))
    max_edge_energy=max(edge_fractions)
    if max_edge_energy > settings['maximum_boundary_kernel_energy']:
        raise ValueError(f'Excessive wavelet kernel energy outside epoch: {max_edge_energy:.3g}')
    baseline_sum = np.empty((n_channels,len(freqs)))
    post_sum = np.empty((2,n_channels,len(freqs),pmask.sum()))
    for start in range(0,n_channels,10):
        stop = min(start+10,n_channels)
        epochs = filtered[start:stop,indices].transpose(1,0,2)
        # No mean subtraction, detrending, amplitude rejection or rereferencing.
        power = mne.time_frequency.tfr_array_morlet(epochs, sfreq=fs, freqs=freqs,
            n_cycles=cycles, zero_mean=settings['zero_mean_wavelets'], use_fft=True,
            output='power', n_jobs=1, verbose='ERROR')
        if power.shape != (int(keep.sum()),stop-start,len(freqs),len(times)) or not np.isfinite(power).all():
            raise ValueError(f'{path}: invalid time-frequency output.')
        baseline_sum[start:stop] = power[:,:,:,bmask].mean(axis=-1).sum(axis=0)
        for condition in (0,1):
            post_sum[condition,start:stop] = power[tasks==condition][:,:,:,pmask].sum(axis=0)
    if np.any(baseline_sum <= 0) or any(np.any(post_sum[c]<=0) for c in (0,1) if counts[c]>0):
        raise ValueError(f'{path}: nonpositive power; cannot normalize to dB.')
    arrays = dict(baseline_sum=baseline_sum, post_sum=post_sum,
        channel_labels=np.array(labels), frequencies=freqs, post_times=times[pmask],
        condition_counts=counts, n_baseline_samples=np.array(bmask.sum()))
    if spatial_qc is not None:
        arrays['voltage_rms']=spatial_qc.pop('voltage_rms')
        arrays['laplacian_rms']=spatial_qc.pop('laplacian_rms')
    qc = dict(n_trials=60,n_excluded=int((~keep).sum()),n_channels=n_channels,n_baseline_samples=int(bmask.sum()),
        n_target_samples=int(pmask.sum()),wavelet_max_half_support_s=half_support/fs,
        maximum_outside_epoch_kernel_energy=max_edge_energy,
        baseline_power_min=float(baseline_sum.min()/keep.sum()),baseline_power_max=float(baseline_sum.max()/keep.sum()),
        n_correct=int(analysis.feedback.eq(1).sum()),n_other=int(analysis.feedback.ne(1).sum()))
    if spatial_qc is not None:qc.update(spatial_qc)
    print(f'{path.name}: {keep.sum()}/60 trials, {n_channels} channels; padded wavelets; no voltage baseline/amplitude rejection.', flush=True)
    raw.close()
    return arrays, qc


def summarize_theta_topographies(cache_dir):
    """Pool linear power within session, normalize per frequency, then average.

    The 'both' map pools linear power across all trials before taking the log;
    it is deliberately not the arithmetic average of the two condition dB maps.
    """
    root = Path(cache_dir); log = pd.read_csv(root/'run_log.csv')
    settings=json.loads((root/'settings.json').read_text())
    n_channels=settings['expected_scalp_channels']
    rt_cleaned=settings.get('rt_cleaned',False)
    expected_excluded=settings.get('expected_excluded_trials',0)
    if len(log)!=383 or log.n_trials.sum()!=22980 or log.n_excluded.sum()!=expected_excluded:
        raise ValueError('Incomplete full-cohort theta cache.')
    rows=[];baseline_rows=[]; canonical=None
    for (subject,group,session),cell in log.groupby(['subject_id','group','session_id']):
        if group != get_subject_group(subject):
            raise ValueError('Wrong group label.')
        expected = 7 if (subject,session)==('e30',1) else {1:8,5:4}[session]
        if len(cell)!=expected:
            raise ValueError(f'{subject}/S{session}: incomplete cached session.')
        bs=None;ps=None;counts=np.zeros(2,dtype=int)
        for run in cell.itertuples():
            path=root/'runs'/Path(run.npz_path).name  # Resolve copied checkpoints locally; hash checked below.
            if hashlib.sha256(path.read_bytes()).hexdigest()!=run.npz_sha:
                raise ValueError(f'{path}: cache hash mismatch.')
            with np.load(path,allow_pickle=False) as z:
                axes=(z['channel_labels'],z['frequencies'],z['post_times'])
                if canonical is None: canonical=tuple(a.copy() for a in axes)
                if any(not np.array_equal(a,b) for a,b in zip(axes,canonical)):
                    raise ValueError('Inconsistent channel/frequency/time axes.')
                b=z['baseline_sum'];p=z['post_sum'];n=z['condition_counts']
                counts_valid=(n.shape==(2,) and np.all((n>0)&(n<=30)) and n.sum()==60-run.n_excluded) if rt_cleaned else np.array_equal(n,[30,30])
                if b.shape!=(n_channels,9) or p.shape!=(2,n_channels,9,205) or not counts_valid:
                    raise ValueError(f'{path}: unexpected cached dimensions/counts.')
                if not np.isfinite(b).all() or not np.isfinite(p).all() or np.any(b<=0) or np.any(p<=0):
                    raise ValueError(f'{path}: invalid cached power.')
                bs=b.copy() if bs is None else bs+b
                ps=p.copy() if ps is None else ps+p
                counts+=n
        baseline=bs/counts.sum()
        labels,freqs,times=canonical
        for channel,label in enumerate(labels):
            for fi,freq in enumerate(freqs):
                baseline_rows.append(dict(subject_id=subject,group=group,session_id=session,
                    channel=label,frequency_hz=freq,baseline_power=baseline[channel,fi],n_trials=int(counts.sum())))
        for name,linear,n in [('distractor',ps[1]/counts[1],counts[1]),
                              ('no_distractor',ps[0]/counts[0],counts[0]),
                              ('both',ps.sum(axis=0)/counts.sum(),counts.sum())]:
            db=10*np.log10(linear/baseline[:,:,None])
            values=db.mean(axis=(1,2))
            if not np.isfinite(values).all():raise ValueError('Nonfinite normalized power.')
            for label,value in zip(labels,values):
                rows.append(dict(subject_id=subject,group=group,session_id=session,condition=name,
                    channel=label,theta_db=float(value),n_trials=int(n)))
    table=pd.DataFrame(rows)
    if len(table)!=32*2*3*n_channels or table.duplicated(['subject_id','session_id','condition','channel']).any():
        raise ValueError('Incomplete/duplicate participant maps.')
    table.to_csv(root/'participant_session_topographies.csv',index=False)
    pd.DataFrame(baseline_rows).to_csv(root/'pooled_baseline_power.csv',index=False)
    print(f'Validated {len(table):,} participant/session/condition/electrode values; all 32 paired participants.')
    return table


def _space_theta_electrode_labels(ax, labels):
    """Place all electrode names without text collisions; tether moved labels."""
    from matplotlib.transforms import offset_copy
    text_items=[t for t in ax.texts if t.get_text() in labels]
    if sorted(t.get_text() for t in text_items)!=sorted(labels):
        raise ValueError('Each map must label every scalp electrode exactly once.')
    # Keep central labels anchored first; resolve dense peripheral rows last.
    text_items.sort(key=lambda t:(abs(t.get_position()[0]),t.get_position()[1],t.get_text()))
    ax.figure.canvas.draw()
    renderer=ax.figure.canvas.get_renderer()
    candidates=sorted([(x,y) for x in (0,3,-3,6,-6) for y in (0,3,-3,6,-6,9,-9)],
                      key=lambda p:(p[0]**2+p[1]**2,abs(p[0]),p))
    placed=[];moved=0
    for text in text_items:
        for dx,dy in candidates:
            text.set_transform(offset_copy(ax.transData,fig=ax.figure,x=dx,y=dy,units='points'))
            bounds=text.get_window_extent(renderer).expanded(1.15,1.25)
            if not any(bounds.overlaps(previous) for previous in placed):
                placed.append(bounds)
                if dx or dy:
                    ax.annotate('',xy=text.get_position(),xytext=(dx,dy),textcoords='offset points',
                        arrowprops=dict(arrowstyle='-',color='.35',linewidth=.3,shrinkA=0,shrinkB=0),zorder=2)
                    moved+=1
                break
        else:
            raise ValueError(f'Cannot place electrode label without overlap: {text.get_text()}')
    if len(placed)!=len(labels):raise ValueError('Missing electrode label placement.')
    print(f'Validated {len(labels)} nonoverlapping electrode labels; {moved} use short location guides.')

