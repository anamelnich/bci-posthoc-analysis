"""Posterior seven-point theta; adapted from the validated midfrontal pipeline.

Preserves full-montage Laplacian, wavelets, pooled linear-power normalization,
and phase-specific trial inclusion. Splits stimulus power into codes 8/32/44.
"""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time

import numpy as np
import pandas as pd

from .analysis import load_training_analysis_file
from .triggers import load_training_trigger_file
from .config import EXPECTED_SUBJECTS, get_subject_group

CHANNELS = ('PO7', 'P7', 'P5', 'PO8', 'P8', 'P6')
ROIS = {'left_posterior': CHANNELS[:3], 'right_posterior': CHANNELS[3:], 'bilateral_posterior': CHANNELS}
CONDITIONS = {'right_distractor': (1,), 'left_distractor': (2,),
              'both_distractors': (1, 2), 'no_distractor': (0,), 'all_trials': (0, 1, 2)}

@lru_cache(maxsize=4)
def _laplacian_matrix(labels, sphere, lambda2, stiffness, terms):
    """Construct MNE's transform and check an independent augmented-system solution."""
    import mne
    from numpy.polynomial.legendre import legval
    if len(labels)!=61 or len(set(labels))!=61 or 'CPZ' in labels or 'FCZ' not in labels:
        raise ValueError('Laplacian requires the same 61 recorded scalp channels.')
    info=mne.create_info(list(labels),512,'eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'),match_case=False)
    basis=mne.io.RawArray(np.eye(len(labels)),info,verbose='ERROR')
    matrix=mne.preprocessing.compute_current_source_density(basis,sphere=sphere,
        lambda2=lambda2,stiffness=stiffness,n_legendre_terms=terms,verbose='ERROR').get_data()
    xyz=np.array([ch['loc'][:3] for ch in info['chs']])-np.array(sphere[:3])
    xyz/=np.linalg.norm(xyz,axis=1)[:,None];cosine=np.clip(xyz@xyz.T,-1,1)
    n=np.arange(1,terms+1,dtype=float)
    g=legval(cosine,np.r_[0,(2*n+1)/(4*np.pi*(n*(n+1))**stiffness)])
    h=legval(cosine,np.r_[0,(2*n+1)/(4*np.pi*(n*(n+1))**(stiffness-1))])
    count=len(labels);system=np.block([[g+lambda2*np.eye(count),np.ones((count,1))],[np.ones((1,count)),np.zeros((1,1))]])
    coef=np.linalg.solve(system,np.r_[np.eye(count),np.zeros((1,count))])
    independent=h@coef[:-1]/sphere[3]**2
    error=float(np.max(abs(matrix-independent))/np.max(abs(matrix)))
    common=float(np.max(abs(matrix@np.ones(count)))/np.max(abs(matrix)))
    if not np.isfinite(matrix).all() or max(error,common)>1e-9:
        raise ValueError(f'Laplacian validation failed: independent={error}, common-mode={common}.')
    qc=dict(relative_independent_matrix_error=error,relative_common_mode_residual=common,
        shape=list(matrix.shape),sphere_m=list(sphere),lambda2=lambda2,stiffness=stiffness,n_legendre_terms=terms,
        interpretation='Annihilates common reference; regularized spherical second spatial derivative, not image interpolation')
    matrix.setflags(write=False)
    print(f'Laplacian operator validated against independent equations; relative error {error:.3g}, common-mode residual {common:.3g}.',flush=True)
    return matrix,qc


def apply_theta_laplacian(filtered, labels, settings):
    """Spatially transform filtered voltage samples before nonlinear power estimation."""
    if labels!=settings['channels'] or filtered.ndim!=2 or filtered.shape[0]!=61 or not np.isfinite(filtered).all():
        raise ValueError('Invalid full-montage input or changed channel order.')
    matrix,qc=_laplacian_matrix(tuple(labels),tuple(settings['sphere_m']),settings['lambda2'],settings['stiffness'],settings['n_legendre_terms'])
    result=matrix@filtered
    voltage_rms=np.sqrt(np.mean(filtered**2,axis=1));laplacian_rms=np.sqrt(np.mean(result**2,axis=1))
    if not np.isfinite(result).all() or min(voltage_rms.min(),laplacian_rms.min())<=0:
        raise ValueError('Nonfinite or flat voltage/Laplacian channel.')
    audit=dict(laplacian_common_mode_residual=qc['relative_common_mode_residual'],
        voltage_rms=voltage_rms,laplacian_rms=laplacian_rms,
        maximum_voltage_rms_to_channel_median=float(max(voltage_rms/np.median(voltage_rms))),
        maximum_laplacian_rms_to_channel_median=float(max(laplacian_rms/np.median(laplacian_rms))))
    print(f'Applied signal-level Laplacian to {filtered.shape[1]:,} samples; all 61 channels finite.',flush=True)
    return result,audit



def extract_posterior_theta_run(gdf_path, settings, trial_mask=None, expected_rt_ms=None):
    """Return linear-power sufficient statistics; preserve every valid trial.

    Arrays: baseline_sum (channel, frequency), post_sum
    (condition [no-distractor, right-distractor, left-distractor], channel, frequency, time).
    Baseline_sum sums per-trial temporal-mean power, not log power.
    """
    import mne
    path = Path(gdf_path)
    for suffix, shape in [('.analysis.txt', (60, 8)), ('.triggers.txt', (180, 3))]:
        values = np.loadtxt(path.with_suffix(suffix))
        if values.shape != shape or not np.isfinite(values).all() or not np.equal(values, np.round(values)).all():
            raise ValueError(f'{path}{suffix}: malformed task file; expected integer {shape}.')
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
    ranges = {'trial_index': range(1, 61), 'task': [0, 1], 'feedback': [1, 2, 3],
              'target_position': [1, 2, 3, 4], 'distractor_position': [0, 2, 4],
              'dot_side': [0, 1], 'bci_output': [99] if task == 'training' else [0, 1, 3]}
    for col, allowed in ranges.items():
        if not analysis[col].isin(allowed).all():
            raise ValueError(f'{path}: invalid {col}.')
    if (analysis.trial_index.tolist() != list(range(1, 61))
            or not analysis.intertrial_interval_ms.gt(0).all()):
        raise ValueError(f'{path}: invalid trial order or intertrial interval.')
    tv = triggers[['trial', 'trigger', 'time']].to_numpy() if 'trial' in triggers else triggers.to_numpy()
    if not np.array_equal(tv[:, 0], np.repeat(np.arange(1, 61), 3)):
        raise ValueError(f'{path}: invalid text trial numbers.')
    if not (np.all(tv[::3, 1] == 4) and np.all(tv[2::3, 1] == 64)):
        raise ValueError(f'{path}: invalid fixation/stimulus/response sequence.')
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
    tasks=expected[keep]
    counts=np.array([(tasks==c).sum() for c in (8,32,44)])
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
    post_sum = np.empty((3,n_channels,len(freqs),pmask.sum()))
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
        for condition, code in enumerate((8,32,44)):
            post_sum[condition,start:stop] = power[tasks==code][:,:,:,pmask].sum(axis=0)
    if np.any(baseline_sum <= 0) or any(np.any(post_sum[c]<=0) for c in range(3) if counts[c]>0):
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



def prepare_posterior_theta_inputs(reference_root, cache_dir):
    """Validate and snapshot the original theta inventories, settings and trial masks."""
    reference = Path(reference_root).expanduser().resolve()
    out = Path(cache_dir).resolve()
    assessment = reference / 'theta_cache'
    intervention = reference / 'intervention_trajectory/cache'
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = json.loads((assessment / 'final_manifest.json').read_text())['files_sha256']
    for name in ['settings.json', 'trial_mask.csv', 'run_log.csv', 'participant_session_topographies.csv']:
        if [v for k, v in manifest.items() if Path(k).name == name] != [digest(assessment / name)]:
            raise ValueError(f'Assessment checksum mismatch: {name}')
    audit = json.loads((intervention / 'input_validation.json').read_text())
    for name, expected in audit['preflight_hashes'].items():
        if digest(intervention / name) != expected:
            raise ValueError(f'Intervention checksum mismatch: {name}')
    settings = json.loads((assessment / 'settings.json').read_text())
    other = json.loads((intervention / 'settings.json').read_text())
    for key in ['sfreq', 'filter_hz', 'epoch_s', 'baseline_s', 'target_s',
                'frequencies_hz', 'n_cycles', 'surface_laplacian', 'zero_mean_wavelets']:
        if settings[key] != other[key]:
            raise ValueError(f'Phase settings disagree: {key}')
    if (settings['sfreq'] != 512 or settings['target_s'] != [.1, .5]
            or settings['baseline_s'] != [-.3, -.1] or settings['filter_hz'] != [.1, 20.]
            or settings['epoch_s'] != [-1., 1.5]
            or settings['frequencies_hz'] != np.arange(4, 8.01, .5).tolist()
            or settings['amplitude_rejection'] is not None):
        raise ValueError('Unexpected reference theta parameters.')
    settings['power_channels'] = list(CHANNELS)
    settings['stimulus_codes'] = [8, 32, 44]
    frames, ledgers = [], []
    for phase, root, inventory_name, ledger_name in [
            ('assessment', assessment, 'run_log.csv', 'trial_mask.csv'),
            ('intervention', intervention, 'source_inventory.csv', 'trial_ledger.csv')]:
        frame = pd.read_csv(root / inventory_name)
        ledger = pd.read_csv(root / ledger_name).rename(columns={'trial': 'trial_index'})
        expected_runs, expected_trials, expected_retained = (383, 22980, 21940) if phase == 'assessment' else (1150, 69000, 69000)
        keys = ['subject_id', 'session_id', 'run_id']
        required = keys + ['source', 'source_size', 'source_mtime_ns', 'analysis_sha', 'triggers_sha', 'group']
        if (not set(required).issubset(frame) or len(frame) != expected_runs
                or len(ledger) != expected_trials or ledger.retained.dtype != bool
                or int(ledger.retained.sum()) != expected_retained
                or frame.duplicated(keys).any() or ledger.duplicated(keys + ['trial_index']).any()
                or set(frame.subject_id) != set(EXPECTED_SUBJECTS)
                or not frame.group.eq(frame.subject_id.map(get_subject_group)).all()):
            raise ValueError(f'{phase}: invalid cohort, inventory or retained mask.')
        for (subject, session), cell in frame.groupby(keys[:2]):
            expected = ((7 if subject == 'e30' and session == 1 else 8 if session == 1 else 4)
                        if phase == 'assessment' else
                        (5 if subject == 'e27' and session == 1 else 7 if subject == 'e52' and session == 3 else 6 if session in (1, 5) else 8))
            expected_ids = [1, 2, 3, 4, 5, 7, 8] if phase == 'intervention' and (subject, session) == ('e52', 3) else list(range(1, expected + 1))
            if len(cell) != expected or sorted(cell.run_id) != expected_ids:
                raise ValueError(f'{phase}/{subject}/S{session}: unexpected run inventory.')
        for key, cell in ledger.groupby(keys):
            if (sorted(cell.trial_index) != list(range(1, 61))
                    or cell.stimulus_code.value_counts().to_dict() != {8: 30, 32: 15, 44: 15}
                    or not cell.stimulus_code.eq(cell.distractor_position.map({0: 8, 2: 32, 4: 44})).all()):
                raise ValueError(f'{phase}/{key}: invalid trial ordering/conditions.')
        if phase == 'assessment':
            eligible = ledger.feedback.eq(1) & ledger.gdf_rt_ms.ge(150) & np.isfinite(ledger.gdf_rt_ms)
            x = ledger.loc[eligible]
            grouped = x.groupby(['subject_id', 'session_id']).gdf_rt_ms
            kept = (x.gdf_rt_ms - grouped.transform('mean')).abs() <= 3 * grouped.transform('std')
            if set(x.index[kept]) != set(ledger.index[ledger.retained]):
                raise ValueError('Saved assessment mask disagrees with independently reconstructed exclusions.')
        for row in frame.itertuples():
            path = Path(row.source)
            stat = path.stat()
            if 'practice' in str(path).lower() or stat.st_size != row.source_size or stat.st_mtime_ns != row.source_mtime_ns:
                raise ValueError(f'{path}: changed source or practice run.')
            for suffix, expected in [('.analysis.txt', row.analysis_sha), ('.triggers.txt', row.triggers_sha)]:
                if digest(path.with_suffix(suffix)) != expected:
                    raise ValueError(f'{path}: paired file changed: {suffix}')
        frames.append(frame[required].assign(phase=phase))
        ledgers.append(ledger.assign(phase=phase))
        print(f'{phase}: validated {len(frame)} sources; {expected_retained:,}/{len(ledger):,} retained trials.', flush=True)
    inventory = pd.concat(frames, ignore_index=True)
    ledger = pd.concat(ledgers, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(out / 'source_inventory.csv', index=False)
    ledger.to_csv(out / 'trial_mask.csv', index=False)
    (out / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    provenance = dict(reference_root=str(reference), reference_settings_sha=digest(assessment / 'settings.json'),
        original_intervention_audit=audit, n_assessment_retained=21940, n_intervention_retained=69000,
        input_hashes={name: digest(out / name) for name in ['source_inventory.csv', 'trial_mask.csv', 'settings.json']})
    (out / 'input_validation.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Posterior inputs ready: 32 participants × seven time points; unchanged phase-specific eligibility.', flush=True)
    return inventory, ledger, settings


def _cache_posterior_run(record, cell, settings, cache_dir):
    """Extract one validated run, or reuse an intact source/mask/code-matched checkpoint."""
    import inspect
    source = Path(record['source'])
    out = Path(cache_dir) / 'runs' / record['phase']
    out.mkdir(parents=True, exist_ok=True)
    cell = cell.sort_values('trial_index')
    keep = cell.retained.to_numpy()
    if cell.trial_index.tolist() != list(range(1, 61)) or keep.dtype != bool:
        raise ValueError(f'{source}: invalid 60-trial mask.')
    stat = source.stat()
    if stat.st_size != record['source_size'] or stat.st_mtime_ns != record['source_mtime_ns']:
        raise ValueError(f'{source}: source changed since preflight.')
    for suffix, key in [('.analysis.txt', 'analysis_sha'), ('.triggers.txt', 'triggers_sha')]:
        if hashlib.sha256(source.with_suffix(suffix).read_bytes()).hexdigest() != record[key]:
            raise ValueError(f'{source}: changed paired input.')
    params = dict(settings, source_task='training' if record['phase'] == 'assessment' else 'decoding')
    signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode() + keep.tobytes()
        + cell.stimulus_code.to_numpy(dtype=np.int64).tobytes()
        + cell.gdf_rt_ms.to_numpy(dtype=float).tobytes()
        + inspect.getsource(extract_posterior_theta_run).encode()
        + inspect.getsource(apply_theta_laplacian).encode()
        + inspect.getsource(_laplacian_matrix).encode()).hexdigest()
    path = out / f"{record['subject_id']}_s{record['session_id']}_r{record['run_id']:02d}.npz"
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text())
        if (old.get('signature') == signature and all(old.get(k) == v for k, v in record.items())
                and old['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()):
            print(f'{record["phase"]}/{path.stem}: validated checkpoint reused.', flush=True)
            return old
    start = time.monotonic()
    arrays, qc = extract_posterior_theta_run(source, params, trial_mask=keep,
        expected_rt_ms=cell.gdf_rt_ms.to_numpy() if record['phase'] == 'assessment' else None)
    counts = np.array([int((keep & cell.stimulus_code.eq(code).to_numpy()).sum()) for code in (8, 32, 44)])
    if not np.array_equal(counts, arrays['condition_counts']):
        raise ValueError(f'{source}: counts fail trial-mask reconciliation.')
    arrays['retained_trial_numbers'] = cell.loc[keep, 'trial_index'].to_numpy(dtype=int)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    meta = dict(record, **qc, signature=signature, n_retained=int(keep.sum()),
                npz_path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                elapsed_s=time.monotonic() - start)
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    print(f'{record["phase"]}/{path.stem}: {keep.sum()}/60 trials cached in {meta["elapsed_s"]:.1f}s.', flush=True)
    return meta


def build_posterior_theta_cache(inventory, ledger, settings, cache_dir, n_jobs=2):
    """Build resumable side-specific posterior power, retaining the approved trial masks."""
    out = Path(cache_dir).resolve()
    audit = json.loads((out / 'input_validation.json').read_text())
    for name, expected in audit['input_hashes'].items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Input cache changed: {name}')
    if n_jobs not in (1, 2) or len(inventory) != 1533 or int(ledger.retained.sum()) != 90940:
        raise ValueError('Expected 1,533 runs, 90,940 retained trials and one/two workers.')
    pd.testing.assert_frame_equal(inventory, pd.read_csv(out / 'source_inventory.csv'), check_dtype=False)
    pd.testing.assert_frame_equal(ledger, pd.read_csv(out / 'trial_mask.csv'), check_dtype=False)
    if settings != json.loads((out / 'settings.json').read_text()):
        raise ValueError('Settings differ from saved preflight.')
    keys = ['phase', 'subject_id', 'session_id', 'run_id']
    grouped = {key: cell for key, cell in ledger.groupby(keys)}
    def work(record):
        return _cache_posterior_run(record, grouped[tuple(record[k] for k in keys)], settings, out)
    logs = []
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        for row in pool.map(work, inventory.to_dict('records')):
            logs.append(row)
            if len(logs) % 25 == 0:
                pd.DataFrame(logs).to_csv(out / 'run_log_progress.csv', index=False)
                print(f'PROGRESS {len(logs)}/{len(inventory)} posterior runs complete.', flush=True)
    log = pd.DataFrame(logs)
    if len(log) != 1533 or log.n_retained.sum() != 90940 or log.n_excluded.sum() != 1040:
        raise ValueError('Extraction totals disagree with approved inventory.')
    log.to_csv(out / 'run_log.csv', index=False)
    print('All posterior power complete: 1,533 runs; 90,940 retained trials.', flush=True)
    return log


def summarize_posterior_theta(cache_dir):
    """Pool linear trial power before dB; verify endpoints against existing theta results."""
    root = Path(cache_dir)
    log = pd.read_csv(root / 'run_log.csv')
    settings = json.loads((root / 'settings.json').read_text())
    inputs = json.loads((root / 'input_validation.json').read_text())
    if len(log) != 1533 or log.n_retained.sum() != 90940 or log.duplicated(['phase', 'subject_id', 'session_id', 'run_id']).any():
        raise ValueError('Incomplete posterior run cache.')
    rows, checks = [], []
    ref = pd.read_csv(Path(inputs['reference_root']) / 'theta_cache/participant_session_topographies.csv')
    for (phase, subject, group, session), cell in log.groupby(['phase', 'subject_id', 'group', 'session_id']):
        baseline_sum = np.zeros((6, 9))
        post_sum = np.zeros((3, 6, 9, 205))
        counts = np.zeros(3, dtype=int)
        for row in cell.itertuples():
            path = Path(row.npz_path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
                raise ValueError(f'{path}: checkpoint hash mismatch.')
            with np.load(path, allow_pickle=False) as z:
                b, p, n = z['baseline_sum'], z['post_sum'], z['condition_counts']
                if (b.shape != (6, 9) or p.shape != (3, 6, 9, 205) or n.shape != (3,)
                        or z['channel_labels'].tolist() != list(CHANNELS)
                        or not np.array_equal(z['frequencies'], settings['frequencies_hz'])
                        or not np.array_equal(z['post_times'], np.arange(52, 257) / 512)
                        or int(z['n_baseline_samples']) != 102 or n.sum() != row.n_retained
                        or not np.isfinite(b).all() or not np.isfinite(p).all() or (b <= 0).any()
                        or any((p[i] <= 0).any() for i in range(3) if n[i] > 0)):
                    raise ValueError(f'{path}: malformed power statistics or axes.')
                baseline_sum += b
                post_sum += p
                counts += n
        if (counts <= 0).any():
            raise ValueError(f'{phase}/{subject}/{session}: empty stimulus condition.')
        baseline = baseline_sum / counts.sum()
        for condition, indices in CONDITIONS.items():
            n = int(counts[list(indices)].sum())
            power = post_sum[list(indices)].sum(axis=0) / n
            channel_db = (10 * np.log10(power / baseline[:, :, None])).mean(axis=(1, 2))
            if phase == 'assessment' and condition in ['both_distractors', 'no_distractor', 'all_trials']:
                old_condition = {'both_distractors': 'distractor', 'no_distractor': 'no_distractor', 'all_trials': 'both'}[condition]
                old = ref[(ref.subject_id == subject) & (ref.session_id == session) & (ref.condition == old_condition)].set_index('channel').reindex(CHANNELS)
                error = float(np.max(np.abs(old.theta_db.to_numpy() - channel_db)))
                if not np.isfinite(error) or error > 1e-9 or not old.n_trials.eq(n).all():
                    raise ValueError(f'{subject}/S{session}/{condition}: original posterior endpoint mismatch {error}.')
                checks.append(error)
            for roi, channels in ROIS.items():
                value = float(channel_db[[CHANNELS.index(ch) for ch in channels]].mean())
                rows.append(dict(subject_id=subject, group=group, phase=phase, session_id=session,
                    session_index=(0 if session == 1 else 6) if phase == 'assessment' else session,
                    roi=roi, condition=condition, theta_db=value, n_trials=n, n_runs=len(cell)))
    participants = pd.DataFrame(rows).sort_values(['roi', 'condition', 'subject_id', 'session_index']).reset_index(drop=True)
    if (len(participants) != 32 * 7 * 3 * 5 or not np.isfinite(participants.theta_db).all()
            or participants.duplicated(['subject_id', 'session_index', 'roi', 'condition']).any()
            or not participants.group.eq(participants.subject_id.map(get_subject_group)).all()
            or not participants.groupby(['roi', 'condition', 'group', 'session_index']).size().eq(16).all()):
        raise ValueError('Incomplete seven-point posterior trajectories.')
    # Equal electrode weighting implies bilateral dB equals the two hemisphere means.
    wide = participants.pivot(index=['subject_id', 'session_index', 'condition'], columns='roi', values='theta_db')
    np.testing.assert_allclose(wide.bilateral_posterior, (wide.left_posterior + wide.right_posterior) / 2, atol=1e-12)
    qc = dict(n_participants=32, n_per_group=16, n_values=len(participants),
              assessment_comparisons=len(checks), maximum_endpoint_error_db=max(checks), statistics_run=False)
    (root / 'summary_validation.json').write_text(json.dumps(qc, indent=2) + '\n')
    print(f'Validated {len(participants):,} posterior values; {len(checks)} pooled endpoint comparisons reproduce prior results (max {max(checks):.3g} dB).', flush=True)
    return participants, qc


def plot_posterior_theta(participants, output_dir):
    """Export three five-panel ROI figures with common limits and participant SEM."""
    import matplotlib.pyplot as plt
    keys = ['roi', 'condition', 'group', 'session_index']
    if (len(participants) != 3360 or not np.isfinite(participants.theta_db).all()
            or participants.duplicated(['subject_id', 'session_index', 'roi', 'condition']).any()
            or set(participants.roi) != set(ROIS) or set(participants.condition) != set(CONDITIONS)
            or set(participants.subject_id) != set(EXPECTED_SUBJECTS)
            or not participants.group.eq(participants.subject_id.map(get_subject_group)).all()):
        raise ValueError('Expected 32 × 7 × 3 × 5 complete finite posterior observations.')
    summary = participants.groupby(keys).theta_db.agg(mean='mean', sd='std', n='count').reset_index()
    if len(summary) != 210 or not summary.n.eq(16).all() or set(summary.session_index) != set(range(7)):
        raise ValueError('Every plotted cell must contain 16 participants.')
    summary['sem'] = summary.sd / np.sqrt(summary.n)
    low = min(0., float((summary['mean'] - summary['sem']).min()))
    high = max(0., float((summary['mean'] + summary['sem']).max()))
    padding = max(.1, .08 * (high - low))
    limits = (low - padding, high + padding)
    titles = ['Right distractor', 'Left distractor', 'Both distractors (pooled)', 'No distractor', 'All trials']
    colors = {'bci': '#DD8452', 'control': '#4C72B0'}
    labels = {'bci': 'BCI', 'control': 'Mental rehearsal'}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exports = {}
    rc = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
          'font.size': 7, 'axes.linewidth': .6, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
    with plt.rc_context(rc):
        for roi, channels in ROIS.items():
            fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.75), sharey=True)
            fig.subplots_adjust(left=.09, right=.985, bottom=.17, top=.87, wspace=.22, hspace=.53)
            for panel, (ax, condition, title) in enumerate(zip(axes.flat, CONDITIONS, titles)):
                for group in ('bci', 'control'):
                    cell = summary[(summary.roi == roi) & (summary.condition == condition) & (summary.group == group)].sort_values('session_index')
                    ax.errorbar(cell.session_index, cell['mean'], yerr=cell['sem'], color=colors[group],
                                marker='o', markersize=3, linewidth=1.1, elinewidth=.7, capsize=2, label=labels[group])
                ax.axhline(0, color='.75', linewidth=.5, zorder=0)
                for x in (.5, 5.5):
                    ax.axvline(x, color='.8', linewidth=.5, linestyle='--', zorder=0)
                ax.set(xlim=(-.35, 6.35), ylim=limits, xticks=range(7),
                       xticklabels=['Pre', '1', '2', '3', '4', '5', 'Post'], xlabel='Session', title=title)
                ax.text(-.10, 1.10, chr(97 + panel), transform=ax.transAxes, fontweight='bold', fontsize=9)
                ax.spines[['top', 'right']].set_visible(False)
                ax.tick_params(width=.6, length=3, labelsize=6)
            axes[1, 2].axis('off')
            for ax in axes[:, 0]:
                ax.set_ylabel('Normalized theta power (dB)')
            axes[1, 2].text(.05, .94, '\n'.join([', '.join(channels), '', '4–8 Hz; 100–500 ms', 'Baseline: −300 to −100 ms',
                'Full-scalp surface Laplacian', '', 'Mean ± SEM; n = 16/group', 'Pre/post: behavior-screened',
                'Sessions 1–5: all outcomes', '', 'Both distractors = left + right trials']),
                transform=axes[1, 2].transAxes, va='top', fontsize=6.5, linespacing=1.45)
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False,
                       bbox_to_anchor=(.53, .985), fontsize=7)
            fig.text(.53, .04, 'Dashed separators mark assessment/intervention boundaries; shared y-axis limits across all ROIs.',
                     ha='center', fontsize=6.3)
            stem = out / f'theta_{roi}_seven_points'
            paths = {ext: str(stem.with_suffix('.' + ext).resolve()) for ext in ('pdf', 'svg', 'png')}
            for path in paths.values():
                fig.savefig(path, dpi=350, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            exports[roi] = paths
    metadata = dict(paths=exports, channels={k: list(v) for k, v in ROIS.items()}, y_limits_db=list(limits),
        n_per_group=16, uncertainty='Between-participant SEM; equal participant weights', statistics_run=False,
        figure_sha256={roi: {ext: hashlib.sha256(Path(path).read_bytes()).hexdigest() for ext, path in paths.items()} for roi, paths in exports.items()})
    (out / 'posterior_theta_figures.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved three five-panel posterior ROI figures (PDF/SVG/PNG); shared y limits {limits}; no inferential statistics.', flush=True)
    return exports, summary


def plot_posterior_theta_condition_trajectories(participants, output_dir):
    """Export separate present/absent seven-point PDFs in the midfrontal style.

    Use the six-electrode posterior ROI, equal participant means, between-person
    SEM, and common limits across conditions. Reuse pooled linear-power-derived
    distractor estimates; do not average separate left/right condition dB values.
    """
    import matplotlib.pyplot as plt
    required = {'subject_id', 'group', 'phase', 'session_id', 'session_index',
                'roi', 'condition', 'theta_db', 'n_trials'}
    if required - set(participants):
        raise ValueError(f'Missing trajectory fields: {required-set(participants)}')
    conditions = {'both_distractors': 'Distractor', 'no_distractor': 'No distractor'}
    selected = participants[participants.roi.eq('bilateral_posterior')
                            & participants.condition.isin(conditions)].copy()
    keys = ['subject_id', 'session_index', 'condition']
    if (len(selected) != 448 or selected[list(required)].isna().any().any()
            or selected.duplicated(keys).any() or not np.isfinite(selected.theta_db).all()
            or set(selected.subject_id) != set(EXPECTED_SUBJECTS)
            or set(selected.session_index) != set(range(7))
            or not selected.group.eq(selected.subject_id.map(get_subject_group)).all()
            or not selected.phase.eq(np.where(selected.session_index.isin([0, 6]), 'assessment', 'intervention')).all()
            or not selected.session_id.eq(selected.session_index.replace({0: 1, 6: 5})).all()
            or not selected.n_trials.gt(0).all()
            or selected.groupby('phase').n_trials.sum().to_dict() != {'assessment': 21940, 'intervention': 69000}):
        raise ValueError('Expected complete bilateral 32 × 7 × 2 trajectories with original phase mapping and trial counts.')
    summary = selected.groupby(['group', 'condition', 'session_index']).theta_db.agg(
        mean='mean', sd='std', n='count').reset_index()
    if len(summary) != 28 or not summary.n.eq(16).all():
        raise ValueError('Every condition/group/time point must contain all 16 participants.')
    summary['sem'] = summary.sd / np.sqrt(summary.n)
    low = min(0., float((summary['mean'] - summary['sem']).min()))
    high = max(0., float((summary['mean'] + summary['sem']).max()))
    pad = max(.12, (high-low)*.12)
    limits = (low-pad, high+pad)
    colors = {'bci': '#DD8452', 'control': '#4C72B0'}
    labels = {'bci': 'BCI', 'control': 'Mental rehearsal'}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exports = {}
    rc = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
          'font.size': 7, 'axes.linewidth': .6, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
    with plt.rc_context(rc):
        for condition, title in conditions.items():
            fig, ax = plt.subplots(figsize=(3.5, 3.15))
            fig.subplots_adjust(left=.19, right=.97, bottom=.29, top=.78)
            for group in ('bci', 'control'):
                cell = summary[summary.group.eq(group) & summary.condition.eq(condition)].sort_values('session_index')
                ax.errorbar(cell.session_index, cell['mean'], yerr=cell['sem'], color=colors[group],
                    marker='o', markersize=3, linewidth=1.1, elinewidth=.7, capsize=2,
                    capthick=.7, label=labels[group], zorder=3)
            ax.axhline(0, color='.7', linewidth=.5, zorder=0)
            for x in (.5, 5.5):
                ax.axvline(x, color='.85', linewidth=.5, linestyle='--', zorder=0)
            ax.set(xlim=(-.35, 6.35), ylim=limits, xticks=range(7),
                xticklabels=['Pre', '1', '2', '3', '4', '5', 'Post'], xlabel='Session',
                ylabel='Normalized theta power (dB)', title=title)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(width=.6, length=3, labelsize=6, direction='out')
            fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=2,
                       frameon=False, bbox_to_anchor=(.55, .98), fontsize=7)
            fig.text(.55, .13, 'PO7, P7, P5, PO8, P8, P6 | 4–8 Hz | 100–500 ms', ha='center', fontsize=6)
            fig.text(.55, .085, 'Mean ± SEM; n = 16 per group', ha='center', fontsize=6.5)
            fig.text(.55, .04, 'Pre/post: behavior-screened; sessions 1–5: all outcomes', ha='center', fontsize=5.8)
            suffix = 'distractor' if condition == 'both_distractors' else 'no_distractor'
            stem = out / f'posterior_theta_{suffix}_seven_sessions'
            paths = {ext: str(stem.with_suffix('.'+ext).resolve()) for ext in ('pdf', 'svg', 'png')}
            for path in paths.values():
                fig.savefig(path, dpi=350, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            exports[condition] = paths
    metadata = dict(paths=exports, roi='bilateral_posterior', channels=list(CHANNELS),
        n_per_group=16, session_labels=['Pre', '1', '2', '3', '4', '5', 'Post'],
        y_limits_db=list(limits), uncertainty='Between-participant SEM; equal participant weights',
        condition_pooling='Distractor = left/right trial linear power pooled before dB',
        input_sha256=hashlib.sha256(selected.sort_values(keys).to_csv(index=False).encode()).hexdigest(),
        figure_sha256={c: {ext: hashlib.sha256(Path(p).read_bytes()).hexdigest() for ext, p in paths.items()}
                       for c, paths in exports.items()},
        endpoint_policy='Pre/post behavior-screened; intervention all outcomes; dashed lines mark phase boundaries.',
        statistics_run=False)
    (out / 'posterior_theta_condition_trajectories.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Validated 448 observations (32 participants × 7 points × 2 conditions); 16/group; original trial totals retained.')
    print(f'Saved separate distractor/no-distractor PDFs, SVGs and PNGs; shared y limits {limits}; mean ± SEM.')
    return metadata, summary


def verify_posterior_theta_intervention_reference(reference_root, cache_dir):
    """Verify side pooling against one original midfrontal all-trial power checkpoint."""
    root = Path(reference_root) / 'intervention_trajectory/cache'
    settings = json.loads((root / 'settings.json').read_text())
    inventory = pd.read_csv(root / 'source_inventory.csv')
    if len(inventory) != 1150 or settings['power_channels'] != ['FZ', 'FCZ', 'CZ']:
        raise ValueError('Expected the original complete midfrontal intervention cache.')
    row = inventory.iloc[0]
    path = root / 'runs' / f'{row.subject_id}_s{row.session_id}_r{row.run_id:02d}.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta['npz_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest() or meta['source'] != row.source:
        raise ValueError('Reference checkpoint checksum/source mismatch.')
    arrays, _ = extract_posterior_theta_run(row.source, settings)
    with np.load(path, allow_pickle=False) as old:
        combined = np.stack([arrays['post_sum'][0], arrays['post_sum'][1:].sum(axis=0)])
        errors = {}
        for field, new in [('baseline_sum', arrays['baseline_sum']), ('post_sum', combined)]:
            np.testing.assert_allclose(new, old[field], rtol=1e-12, atol=1e-9)
            errors[field] = float(np.max(np.abs(new - old[field]) / old[field]))
        np.testing.assert_array_equal(arrays['condition_counts'], [30, 15, 15])
        np.testing.assert_array_equal(old['condition_counts'], [30, 30])
    qc = dict(source=row.source, maximum_relative_errors=errors,
              check='New side-specific extraction pooled to original midfrontal two-condition sufficient statistics')
    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'intervention_reference_validation.json').write_text(json.dumps(qc, indent=2) + '\n')
    print(f'Intervention regression check passed: side pooling matches original midfrontal power, maximum relative error {max(errors.values()):.3g}.', flush=True)
    return qc
