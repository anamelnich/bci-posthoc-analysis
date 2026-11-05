"""Descriptive Stroop Laplacian theta with the current session-pooled RT mask."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import inspect
import json
import time

import numpy as np
import pandas as pd

from .config import EXPECTED_SUBJECTS, get_subject_group
from .consolidated import collect_all_stroop_runs, load_stroop_behoutput_file, validate_all_stroop_files
from .theta_laplacian import apply_theta_laplacian, _laplacian_matrix
from .theta_paper_reproduction import seal_theta_paper_outputs

CONDITIONS = ('congruent', 'incongruent', 'both')
TRIAL_KEYS = ['subject_id', 'session_id', 'run_id', 'trial_number']
SOURCE_URL = 'https://github.com/anamelnich/distractor-classification/blob/bdf2a9ff03119b94826d9865eff5e0e4190836ad/visualInterface/Stroop.py'


def select_stroop_theta_trials(data):
    """Retain correct trials within one participant/session +/-3 sample-SD pass."""
    required = set(TRIAL_KEYS + ['group', 'trial_type', 'response', 'reaction_time_ms'])
    if (data.empty or not required.issubset(data) or data.duplicated(TRIAL_KEYS).any()
            or data[list(required - {'reaction_time_ms'})].isna().any().any()
            or not data.response.isin([1, 2, 3]).all()
            or not data.trial_type.isin(CONDITIONS[:2]).all()
            or not data.session_id.isin([1, 5]).all()):
        raise ValueError('Stroop mask requires unique, valid participant/session/run/trial rows and response/congruency fields.')
    observed = data.loc[data.response.ne(3), 'reaction_time_ms']
    if (not np.isfinite(observed).all() or observed.le(0).any()
            or data.loc[data.response.eq(3), 'reaction_time_ms'].notna().any()):
        raise ValueError('Non-timeout Stroop RT must be finite and positive; timeout RT must be missing.')
    correct = data.loc[data.response.eq(1)]
    thresholds = correct.groupby(['subject_id', 'session_id']).agg(
        session_mean_rt_ms=('reaction_time_ms', 'mean'),
        session_sd_rt_ms=('reaction_time_ms', 'std'), n_correct=('reaction_time_ms', 'size'),
        n_runs=('run_id', 'nunique')).reset_index()
    if len(thresholds) != len(data[['subject_id', 'session_id']].drop_duplicates()) or thresholds.n_correct.lt(2).any():
        raise ValueError('Every participant/session needs at least two correct trials to estimate sample SD.')
    ledger = data.merge(thresholds, on=['subject_id', 'session_id'], how='left', validate='many_to_one')
    ledger['rt_outlier'] = ledger.response.eq(1) & (
        (ledger.reaction_time_ms-ledger.session_mean_rt_ms).abs() > 3*ledger.session_sd_rt_ms)
    ledger['retained'] = ledger.response.eq(1) & ~ledger.rt_outlier
    ledger['exclusion_reason'] = np.select(
        [ledger.response.eq(3), ledger.response.eq(2), ledger.rt_outlier],
        ['timeout', 'incorrect', 'outside_session_3sd'], default='retained')
    # Independent NumPy reconstruction protects ddof, pooling, and one-pass semantics.
    for key, cell in ledger.groupby(['subject_id', 'session_id']):
        x = cell.loc[cell.response.eq(1), 'reaction_time_ms'].to_numpy()
        keep = cell.response.eq(1) & ((cell.reaction_time_ms-x.mean()).abs() <= 3*x.std(ddof=1))
        if not np.array_equal(keep.to_numpy(), cell.retained.to_numpy()):
            raise ValueError(f'{key}: independent Stroop exclusion reconstruction failed.')
    counts = ledger.exclusion_reason.value_counts().to_dict()
    if sum(counts.values()) != len(data):
        raise ValueError('Stroop exclusion categories do not reconcile.')
    print(f'Stroop mask: {len(thresholds)} participant/sessions; both runs and congruencies pooled; ddof=1; no minimum RT. {counts}', flush=True)
    return ledger.sort_values(TRIAL_KEYS).reset_index(drop=True), thresholds


def inspect_stroop_theta_run(gdf_path, behoutput_path, expected_labels, preload=False):
    """Validate paired behavioral fields, every Status event, montage and real padding."""
    import mne
    path = Path(gdf_path)
    if not path.is_file() or not path.name.endswith('_stroop.gdf') or 'practice' in path.name:
        raise ValueError(f'Expected an existing non-practice Stroop GDF: {path}')
    beh = load_stroop_behoutput_file(behoutput_path)
    if (beh.trial_type.value_counts().to_dict() != {'congruent': 30, 'incongruent': 30}
            or not np.array_equal(np.where(beh.stimulus.str.upper()==beh.ink_color.str.upper(),
                                           'congruent', 'incongruent'), beh.trial_type)):
        raise ValueError(f'{behoutput_path}: expected 30/30 congruency labels consistent with word/ink colors.')
    observed = beh.loc[beh.response.ne(3), 'reaction_time_ms']
    if not np.isfinite(observed).all() or not observed.between(0, 2000, inclusive='right').all():
        raise ValueError(f'{behoutput_path}: invalid RT within the two-second vocal recording window.')
    raw = mne.io.read_raw_gdf(path, preload=preload, verbose='ERROR')
    try:
        omit = ['M1', 'M2', 'EOG', 'sens7', 'sens8', 'Status', 'CPZ']
        labels = [c for c in raw.ch_names if c not in omit]
        if (raw.info['sfreq'] != 512 or len(raw.ch_names) != 67 or len(set(raw.ch_names)) != 67
                or labels != list(expected_labels) or len(labels) != 61 or 'FCZ' not in labels or 'CPZ' in raw.ch_names):
            raise ValueError(f'{path}: sampling/channel order differs from the 61-channel distractor theta pipeline.')
        events = mne.find_events(raw, stim_channel='Status', shortest_event=1, verbose='ERROR')
        # Observed acquisition markers (code 1) bracket, and never interrupt, the task.
        if (events.shape != (122, 3) or not np.array_equal(events[[0, -1], 2], [1, 1])
                or np.any(events[1:-1, 2] == 1)):
            raise ValueError(f'{path}: expected two bracketing code-1 markers and 120 ordered task events.')
        task = events[1:-1]
        expected = beh.trial_type.map({'congruent': 8, 'incongruent': 32}).to_numpy()
        if (not np.array_equal(task[::2, 2], np.full(60, 4))
                or not np.array_equal(task[1::2, 2], expected)):
            raise ValueError(f'{path}: Stroop fixation/stimulus order or behavioral congruency alignment failed.')
        stimuli = task[1::2, 0]
        fixation_s = (stimuli-task[::2, 0])/512
        if (fixation_s < .3).any():
            raise ValueError(f'{path}: requested -300 ms baseline begins before fixation.')
        if stimuli[0]-raw.first_samp < 512 or stimuli[-1]-raw.first_samp+768 >= raw.n_times:
            raise ValueError(f'{path}: insufficient real EEG for -1 to +1.5 s epochs; do not pad or drop silently.')
        if np.any(np.diff(stimuli) <= 768):
            raise ValueError(f'{path}: next Stroop stimulus enters a preceding epoch.')
        qc = dict(n_events=len(events), n_trials=60, fixation_min_s=float(fixation_s.min()),
                  fixation_max_s=float(fixation_s.max()), n_samples=int(raw.n_times), source_mapping=SOURCE_URL)
        print(f'{path.name}: 60 aligned Stroop trials, 30/30 congruency, 61 scalp channels; fixation {fixation_s.min():.3f}–{fixation_s.max():.3f} s.', flush=True)
        return raw, beh, stimuli, qc
    except Exception:
        raw.close()
        raise


def prepare_stroop_theta_inputs(study_root, theta_cache_dir, output_dir, target_s=(.1, .5)):
    """Validate all 128 Stroop runs and create a separate session-pooled trial ledger."""
    target = np.asarray(target_s, dtype=float)
    if (target.shape != (2,) or not np.isfinite(target).all()
            or not 0 <= target[0] < target[1] <= 1.5):
        raise ValueError("Stroop target_s must contain two finite increasing poststimulus bounds within 0–1.5 s.")
    study_root, theta_root, out = map(Path, (study_root, theta_cache_dir, output_dir))
    out.mkdir(parents=True, exist_ok=True)
    source_settings = theta_root/'settings.json'
    manifest = json.loads((theta_root/'final_manifest.json').read_text())
    digest = hashlib.sha256(source_settings.read_bytes()).hexdigest()
    expected = [v for k, v in manifest['files_sha256'].items() if Path(k).name == 'settings.json']
    if expected != [digest]:
        raise ValueError(f'{source_settings}: distractor theta settings checksum mismatch.')
    settings = json.loads(source_settings.read_text())
    if (settings['frequencies_hz'] != np.arange(4, 8.01, .5).tolist()
            or settings['baseline_s'] != [-.3, -.1] or settings['target_s'] != [.1, .5]
            or settings['epoch_s'] != [-1., 1.5] or settings['filter_hz'] != [.1, 20.]
            or not settings.get('surface_laplacian') or settings['voltage_baseline'] is not None
            or settings['amplitude_rejection'] is not None):
        raise ValueError('Distractor theta settings differ from the requested Stroop pipeline.')
    settings['target_s'] = target.tolist()
    sample_times = np.arange(round(settings['epoch_s'][0]*settings['sfreq']),
                             round(settings['epoch_s'][1]*settings['sfreq'])+1)/settings['sfreq']
    selected = sample_times[(sample_times >= target[0]) & (sample_times <= target[1])]
    if selected.size == 0:
        raise ValueError('Stroop target window contains no samples at the configured sampling rate.')
    print(f'Stroop target: {1000*target[0]:g}–{1000*target[1]:g} ms; {len(selected)} samples, '
          f'actual centers {1000*selected[0]:.5f}–{1000*selected[-1]:.5f} ms.', flush=True)
    audit = validate_all_stroop_files(root_path=study_root)
    if any(audit['issues'].values()) or audit['summary']['behoutput_files_valid'] != 128:
        raise ValueError(f'Comprehensive Stroop behavioral validation failed: {audit}')
    (out/'comprehensive_stroop_validation.json').write_text(json.dumps(audit, indent=2, default=str)+'\n')
    runs, issues, found = collect_all_stroop_runs(root_path=study_root)
    if len(runs) != 128 or any(issues.values()) or set(found) != set(EXPECTED_SUBJECTS):
        raise ValueError(f'Stroop requires all 32 participants, two runs in each Session 1/5: {issues}')
    rows, inventory, failures = [], [], []
    for run in runs:
        source = Path(run['run_folder'])/(run['run_id']+'.gdf')
        behpath = Path(run['behoutput_file'])
        try:
            raw, beh, stimuli, qc = inspect_stroop_theta_run(source, behpath, settings['surface_laplacian']['channels'])
            beh = beh.assign(subject_id=run['subject_id'], group=get_subject_group(run['subject_id']),
                             session_id=run['session_number'], run_id=run['run_number'],
                             stimulus_sample=stimuli, stimulus_code=beh.trial_type.map({'congruent':8,'incongruent':32}))
            stat = source.stat()
            inventory.append(dict(subject_id=run['subject_id'], group=get_subject_group(run['subject_id']),
                session_id=run['session_number'], run_id=run['run_number'], source=str(source.resolve()),
                source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns, behoutput_source=str(behpath.resolve()),
                behoutput_sha256=hashlib.sha256(behpath.read_bytes()).hexdigest(), **qc))
            rows.append(beh); raw.close()
        except Exception as exc:
            failures.append(dict(source=str(source), error=str(exc)))
            print(f'Stroop validation failure: {source}: {exc}', flush=True)
    (out/'source_failures.json').write_text(json.dumps(failures, indent=2)+'\n')
    if failures:
        raise ValueError(f'{len(failures)} Stroop runs failed; see {out / "source_failures.json"}. No incomplete-cohort maps will be generated.')
    data = pd.concat(rows, ignore_index=True)
    inv = pd.DataFrame(inventory).sort_values(['subject_id','session_id','run_id']).reset_index(drop=True)
    if len(data) != 7680 or not inv.groupby(['subject_id','session_id']).size().eq(2).all():
        raise ValueError('Stroop cohort/run/trial totals do not reconcile.')
    ledger, thresholds = select_stroop_theta_trials(data)
    retained_counts = ledger.loc[ledger.retained].groupby(['subject_id','session_id','trial_type']).size()
    if len(retained_counts) != 128 or retained_counts.le(0).any():
        raise ValueError('At least one participant/session/congruency has no retained trials.')
    settings.update(task='stroop', conditions=list(CONDITIONS), rt_cleaned=True,
        trial_selection='Correct only; one-pass participant/session mean +/-3 sample SD across both runs/congruencies; no minimum RT cutoff',
        baseline_pool='All retained congruent and incongruent trials within participant/session/channel/frequency',
        expected_excluded_trials=int((~ledger.retained).sum()), n_retained=int(ledger.retained.sum()),
        source_task_mapping=SOURCE_URL, source_theta_settings_sha256=digest)
    ledger.to_csv(out/'trial_mask.csv', index=False); inv.to_csv(out/'source_inventory.csv', index=False)
    thresholds.to_csv(out/'rt_thresholds.csv', index=False)
    exclusions = ledger.groupby(['group','session_id','exclusion_reason']).size().reset_index(name='n_trials')
    exclusions.to_csv(out/'exclusions.csv', index=False)
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    mask_qc = dict(n_trials=len(ledger), n_retained=int(ledger.retained.sum()), n_excluded=int((~ledger.retained).sum()),
        n_subjects=32, n_runs=128, counts=ledger.exclusion_reason.value_counts().to_dict(),
        mask_sha256=hashlib.sha256((out/'trial_mask.csv').read_bytes()).hexdigest(),
        timing_note='Stroop fixation is approximately 500 ms; Morlet temporal support extends beyond the nominal baseline window. No timing correction or additional exclusion.',
        scope='Descriptive topographies only; no ROI inference, hypothesis tests or RT correlations.')
    (out/'mask_validation.json').write_text(json.dumps(mask_qc, indent=2)+'\n')
    seal_theta_paper_outputs(out)
    print(f'Prepared Stroop: 128 runs, 7,680 trials, {mask_qc["n_retained"]:,} retained. Independent NumPy mask verified.', flush=True)
    return inv, ledger, settings


def extract_stroop_theta_run(record, cell, settings):
    """Compute identical distractor-pipeline power sums using verified Stroop onsets."""
    import mne
    cell = cell.sort_values('trial_number')
    keep = cell.retained.to_numpy()
    if (cell.trial_number.tolist() != list(range(1,61)) or keep.dtype != bool or not keep.any()
            or settings.get('task') != 'stroop'):
        raise ValueError('Expected the ordered 60-trial Stroop ledger and a nonempty boolean mask.')
    path = Path(record['source'])
    labels = settings['surface_laplacian']['channels']
    raw, beh, stimuli, event_qc = inspect_stroop_theta_run(path, record['behoutput_source'], labels, preload=True)
    try:
        for column in ['trial_number','trial_type','response']:
            if not np.array_equal(beh[column],cell[column]):
                raise ValueError(f'{path}: changed behavioral {column}.')
        if (not np.allclose(beh.reaction_time_ms,cell.reaction_time_ms,rtol=0,atol=1e-10,equal_nan=True)
                or not np.array_equal(stimuli,cell.stimulus_sample)):
            raise ValueError(f'{path}: EEG/behavioral ledger alignment changed.')
        fs = settings['sfreq']; n_channels = settings['expected_scalp_channels']
        data = raw.get_data(picks=labels)
        if not np.isfinite(data).all():
            raise ValueError(f'{path}: nonfinite EEG samples.')
        filtered = mne.filter.filter_data(data, fs, *settings['filter_hz'],
            method='fir', phase='zero', fir_design='firwin', verbose='ERROR')
        spatial_qc=None
        if settings.get('surface_laplacian'):
            from .theta_laplacian import apply_theta_laplacian
            filtered,spatial_qc=apply_theta_laplacian(filtered,labels,settings['surface_laplacian'])
        offsets = np.arange(round(settings['epoch_s'][0]*fs), round(settings['epoch_s'][1]*fs)+1)
        indices = stimuli[:,None] - raw.first_samp + offsets[None,:]
        if indices.min() < 0 or indices.max() >= raw.n_times:
            raise ValueError(f'{path}: insufficient real-data epoch padding.')
        indices=indices[keep]
        tasks=beh.trial_type.map({'congruent':0,'incongruent':1}).to_numpy()[keep]
        counts=np.array([(tasks==c).sum() for c in (0,1)])
        times = offsets / fs
        bmask = (times >= settings['baseline_s'][0]) & (times <= settings['baseline_s'][1])
        pmask = (times >= settings['target_s'][0]) & (times <= settings['target_s'][1])
        if fs != 512 or bmask.sum() != 102 or not pmask.any():
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
            n_correct=int(beh.response.eq(1).sum()),n_other=int(beh.response.ne(1).sum()))
        if spatial_qc is not None:qc.update(spatial_qc)
        print(f'{path.name}: {keep.sum()}/60 trials, {n_channels} channels; padded wavelets; no voltage baseline/amplitude rejection.', flush=True)
        return arrays, qc
    finally:
        raw.close()


def _cache_stroop_theta_run(record, cell, settings, output_dir):
    """Reuse only a source-, mask-, settings- and code-matched intact run checkpoint."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    source=Path(record['source']);stat=source.stat()
    if (stat.st_size!=record['source_size'] or stat.st_mtime_ns!=record['source_mtime_ns']
            or hashlib.sha256(Path(record['behoutput_source']).read_bytes()).hexdigest()!=record['behoutput_sha256']):
        raise ValueError(f'{source}: source changed after preflight.')
    key=f"{record['subject_id']}_s{record['session_id']}_r{record['run_id']:02d}"
    cell=cell.sort_values('trial_number')
    signature=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()
        +cell[['trial_number','retained','stimulus_sample','reaction_time_ms']].to_csv(index=False).encode()
        +inspect.getsource(extract_stroop_theta_run).encode()+inspect.getsource(inspect_stroop_theta_run).encode()
        +inspect.getsource(apply_theta_laplacian).encode()+inspect.getsource(_laplacian_matrix).encode()).hexdigest()
    npzpath=out/f'{key}.npz';metapath=out/f'{key}.json'
    if npzpath.is_file() and metapath.is_file():
        old=json.loads(metapath.read_text())
        if (old.get('signature')==signature
                and all(old.get(k)==record[k] for k in ['source','source_size','source_mtime_ns','behoutput_sha256'])
                and old.get('npz_sha256')==hashlib.sha256(npzpath.read_bytes()).hexdigest()):
            old['npz_path']=str(npzpath.resolve())
            print(f'{key}: validated Stroop checkpoint reused.',flush=True)
            return old
    start=time.monotonic()
    arrays,qc=extract_stroop_theta_run(record,cell,settings)
    expected=np.array([np.sum(cell.retained & cell.trial_type.eq(c)) for c in CONDITIONS[:2]])
    if not np.array_equal(arrays['condition_counts'],expected):
        raise ValueError(f'{key}: power trial counts differ from the behavioral mask.')
    arrays['retained_trial_numbers']=cell.loc[cell.retained,'trial_number'].to_numpy(dtype=int)
    temp=out/f'{key}.tmp.npz';np.savez_compressed(temp,**arrays);temp.replace(npzpath)
    meta={**record,**qc,'signature':signature,'n_retained':int(cell.retained.sum()),
          'npz_path':str(npzpath.resolve()),'npz_sha256':hashlib.sha256(npzpath.read_bytes()).hexdigest(),
          'elapsed_s':time.monotonic()-start}
    metapath.write_text(json.dumps(meta,indent=2)+'\n')
    print(f'{key}: Stroop theta cached, {meta["n_retained"]} trials, {meta["elapsed_s"]:.1f}s.',flush=True)
    return meta


def build_stroop_theta_cache(inventory, ledger, settings, output_dir, n_jobs=2):
    """Compute or resume 128 Stroop runs, with one or two bounded-memory workers."""
    out=Path(output_dir)
    if n_jobs not in (1,2) or len(inventory)!=128 or len(ledger)!=7680 or settings.get('task')!='stroop':
        raise ValueError('Expected 128 Stroop runs, 7,680 trial rows and one or two workers.')
    mask_qc=json.loads((out/'mask_validation.json').read_text())
    if (hashlib.sha256((out/'trial_mask.csv').read_bytes()).hexdigest()!=mask_qc['mask_sha256']
            or json.loads((out/'settings.json').read_text())!=settings):
        raise ValueError('Saved Stroop mask/settings changed; rerun preparation.')
    pd.testing.assert_frame_equal(ledger.reset_index(drop=True),pd.read_csv(out/'trial_mask.csv'),check_dtype=False)
    pd.testing.assert_frame_equal(inventory.reset_index(drop=True),pd.read_csv(out/'source_inventory.csv'),check_dtype=False)
    grouped={k:v for k,v in ledger.groupby(['subject_id','session_id','run_id'])}
    logs=[]
    def work(record):
        key=(record['subject_id'],record['session_id'],record['run_id'])
        return _cache_stroop_theta_run(record,grouped[key],settings,out/'runs')
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        for result in pool.map(work,inventory.to_dict('records')):
            logs.append(result)
            pd.DataFrame(logs).to_csv(out/'run_log_progress.csv',index=False)
    log=pd.DataFrame(logs)
    if len(log)!=128 or log.n_retained.sum()!=mask_qc['n_retained'] or log.n_excluded.sum()!=mask_qc['n_excluded']:
        raise ValueError('Stroop EEG exclusions do not match the behavioral mask.')
    log.to_csv(out/'run_log.csv',index=False)
    print(f'Stroop theta complete: {len(log)} runs, {log.n_retained.sum():,} retained trials; no extra EEG exclusions.',flush=True)
    return log


def rewindow_stroop_theta_cache(inventory, ledger, settings, output_dir, source_cache_dir):
    """Select a contained time window from validated per-run linear-power checkpoints."""
    out, source = Path(output_dir), Path(source_cache_dir)
    if out.resolve()==source.resolve() or settings.get('task')!='stroop' or len(inventory)!=128 or len(ledger)!=7680:
        raise ValueError('Rewindowing requires separate Stroop output/source caches and the complete cohort.')
    # Validate every retained source artifact before using its settings or power.
    manifest=json.loads((source/'final_manifest.json').read_text())['files_sha256']
    for path,digest in manifest.items():
        if hashlib.sha256((source/Path(path).name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Stroop source cache integrity mismatch: {path}')
    previous=json.loads((source/'settings.json').read_text())
    if ({k:v for k,v in previous.items() if k!='target_s'} !=
            {k:v for k,v in settings.items() if k!='target_s'}):
        raise ValueError('Cached preprocessing, baseline or trial-selection settings differ.')
    target=np.asarray(settings['target_s'],dtype=float)
    if (target.shape!=(2,) or not np.isfinite(target).all()
            or not previous['target_s'][0]<=target[0]<target[1]<=previous['target_s'][1]):
        raise ValueError('Requested target must be wholly contained in the source window.')
    for root in (source,out):
        qc=json.loads((root/'mask_validation.json').read_text())
        if hashlib.sha256((root/'trial_mask.csv').read_bytes()).hexdigest()!=qc['mask_sha256']:
            raise ValueError(f'{root}: trial mask hash mismatch.')
        pd.testing.assert_frame_equal(ledger.reset_index(drop=True),pd.read_csv(root/'trial_mask.csv'),check_dtype=False)
        pd.testing.assert_frame_equal(inventory.reset_index(drop=True),pd.read_csv(root/'source_inventory.csv'),check_dtype=False)
    if json.loads((out/'settings.json').read_text())!=settings:
        raise ValueError('Saved output settings differ from requested settings.')
    log=pd.read_csv(source/'run_log.csv');run_keys=['subject_id','session_id','run_id']
    if len(log)!=128 or log.duplicated(run_keys).any():
        raise ValueError('Source cache must contain 128 unique run records.')
    pd.testing.assert_frame_equal(log[run_keys].sort_values(run_keys).reset_index(drop=True),
                                  inventory[run_keys].sort_values(run_keys).reset_index(drop=True))
    times=np.arange(round(settings['epoch_s'][0]*settings['sfreq']),
                    round(settings['epoch_s'][1]*settings['sfreq'])+1)/settings['sfreq']
    old_times=times[(times>=previous['target_s'][0]) & (times<=previous['target_s'][1])]
    new_times=times[(times>=target[0]) & (times<=target[1])]
    if settings['sfreq']!=512 or not new_times.size:
        raise ValueError('Expected 512 Hz and a nonempty target window.')
    select=(old_times>=target[0]) & (old_times<=target[1])
    runs=out/'runs';runs.mkdir(parents=True,exist_ok=True);rows=[]
    for record in inventory.to_dict('records'):
        subject,session,run=(record[k] for k in run_keys)
        key=f'{subject}_s{session}_r{run:02d}';path=source/'runs'/f'{key}.npz'
        row=log[(log.subject_id==subject)&(log.session_id==session)&(log.run_id==run)].iloc[0]
        meta=json.loads(path.with_suffix('.json').read_text())
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if (digest!=row.npz_sha256 or digest!=meta['npz_sha256']
                or any(meta[k]!=record[k] for k in ['source','source_size','source_mtime_ns','behoutput_sha256'])
                or meta['maximum_outside_epoch_kernel_energy']>settings['maximum_boundary_kernel_energy']):
            raise ValueError(f'{key}: checkpoint/source metadata or wavelet-boundary validation failed.')
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        part=ledger[(ledger.subject_id==subject)&(ledger.session_id==session)&(ledger.run_id==run)].sort_values('trial_number')
        expected=np.array([(part.retained & part.trial_type.eq(c)).sum() for c in CONDITIONS[:2]])
        b,p=arrays['baseline_sum'],arrays['post_sum']
        if (arrays['channel_labels'].tolist()!=settings['surface_laplacian']['channels']
                or not np.array_equal(arrays['frequencies'],settings['frequencies_hz'])
                or not np.array_equal(arrays['post_times'],old_times)
                or not np.array_equal(arrays['condition_counts'],expected)
                or not np.array_equal(arrays['retained_trial_numbers'],part.loc[part.retained,'trial_number'])
                or b.shape!=(61,9) or p.shape!=(2,61,9,len(old_times))
                or not np.isfinite(b).all() or not np.isfinite(p).all() or (b<=0).any()
                or any((p[c]<=0).any() for c in range(2) if expected[c]>0)
                or int(arrays['n_baseline_samples'])!=102):
            raise ValueError(f'{key}: invalid cached axes, power, trial IDs or counts.')
        arrays['post_times']=new_times.copy();arrays['post_sum']=p[...,select]
        dest=runs/path.name;temp=runs/f'{key}.tmp.npz'
        np.savez_compressed(temp,**arrays);temp.replace(dest)
        meta.update(n_target_samples=len(new_times),npz_path=str(dest.resolve()),
            npz_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            source_checkpoint=str(path.resolve()),source_checkpoint_sha256=digest,
            source_target_s=previous['target_s'],target_s=target.tolist(),
            cache_method='Exact time-sample subset of validated linear-power checkpoint',
            source_elapsed_s=meta['elapsed_s'],elapsed_s=0.,
            signature=hashlib.sha256(digest.encode()+json.dumps(settings,sort_keys=True).encode()
                                     +inspect.getsource(rewindow_stroop_theta_cache).encode()).hexdigest())
        dest.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');rows.append(meta)
    result=pd.DataFrame(rows)
    if result.n_retained.sum()!=int(ledger.retained.sum()):
        raise ValueError('Rewindowed cache trial totals do not match the current ledger.')
    result.to_csv(out/'run_log.csv',index=False)
    seal_theta_paper_outputs(out)
    print(f'Rewindowed {len(result)} validated Stroop runs to {1000*target[0]:g}–{1000*target[1]:g} ms: '
          f'{len(new_times)} samples, {result.n_retained.sum():,} retained trials; baseline and linear power unchanged.',flush=True)
    return result


def summarize_stroop_theta(cache_dir):
    """Pool run/trial linear power, use one shared baseline, then average dB over time/frequency."""
    root=Path(cache_dir)
    settings=json.loads((root/'settings.json').read_text())
    mask_qc=json.loads((root/'mask_validation.json').read_text())
    if settings.get('task')!='stroop' or hashlib.sha256((root/'trial_mask.csv').read_bytes()).hexdigest()!=mask_qc['mask_sha256']:
        raise ValueError('Invalid Stroop settings/mask integrity.')
    ledger=pd.read_csv(root/'trial_mask.csv');log=pd.read_csv(root/'run_log.csv')
    inventory=pd.read_csv(root/'source_inventory.csv')
    run_keys=['subject_id','session_id','run_id']
    if len(log)!=128 or log.duplicated(run_keys).any() or log.n_retained.sum()!=mask_qc['n_retained']:
        raise ValueError('Incomplete or duplicate Stroop run cache.')
    pd.testing.assert_frame_equal(log[run_keys].sort_values(run_keys).reset_index(drop=True),
                                  inventory[run_keys].sort_values(run_keys).reset_index(drop=True))
    epoch_times=np.arange(round(settings['epoch_s'][0]*settings['sfreq']),
                          round(settings['epoch_s'][1]*settings['sfreq'])+1)/settings['sfreq']
    expected_post_times=epoch_times[(epoch_times>=settings['target_s'][0]) & (epoch_times<=settings['target_s'][1])]
    if settings['sfreq']!=512 or expected_post_times.size==0:
        raise ValueError('Invalid sampling rate or empty target time axis.')
    print(f'Validating cached Stroop target {settings["target_s"]} s: {len(expected_post_times)} samples.',flush=True)
    rows=[];baselines=[];trial_counts=[];canonical=None
    for (subject,group,session),cell in log.groupby(['subject_id','group','session_id']):
        if len(cell)!=2 or set(cell.run_id)!={1,2} or group!=get_subject_group(subject):
            raise ValueError(f'{subject}/S{session}: need two distinct Stroop runs and correct group.')
        bs=None;ps=None;counts=np.zeros(2,dtype=int)
        for run in cell.itertuples():
            path=root/'runs'/f'{subject}_s{session}_r{run.run_id:02d}.npz'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=run.npz_sha256:
                raise ValueError(f'{path}: checkpoint hash mismatch.')
            with np.load(path,allow_pickle=False) as z:
                axes=(z['channel_labels'],z['frequencies'],z['post_times'])
                if canonical is None:canonical=tuple(a.copy() for a in axes)
                if (any(not np.array_equal(a,b) for a,b in zip(axes,canonical))
                        or axes[0].tolist()!=settings['surface_laplacian']['channels']
                        or not np.array_equal(axes[1],settings['frequencies_hz'])
                        or not np.array_equal(axes[2],expected_post_times)):
                    raise ValueError(f'{path}: channel/frequency/time axes mismatch.')
                b,p,n=z['baseline_sum'],z['post_sum'],z['condition_counts']
                part=ledger[(ledger.subject_id==subject)&(ledger.session_id==session)&(ledger.run_id==run.run_id)]
                kept=part.loc[part.retained].sort_values('trial_number')
                expected=np.array([kept.trial_type.eq(c).sum() for c in CONDITIONS[:2]])
                if (b.shape!=(61,9) or p.shape!=(2,61,9,len(expected_post_times)) or not np.array_equal(n,expected)
                        or not np.array_equal(z['retained_trial_numbers'],kept.trial_number)
                        or not np.isfinite(b).all() or not np.isfinite(p).all() or (b<=0).any()
                        or any((p[c]<=0).any() for c in range(2) if n[c]>0)
                        or int(z['n_baseline_samples'])!=102):
                    raise ValueError(f'{path}: invalid power arrays, retained IDs, counts or baseline samples.')
                bs=b.copy() if bs is None else bs+b
                ps=p.copy() if ps is None else ps+p
                counts+=n
        if (counts<=0).any():raise ValueError(f'{subject}/S{session}: missing retained congruency.')
        baseline=bs/counts.sum()
        for ch,values in zip(canonical[0],baseline):
            for freq,value in zip(canonical[1],values):
                baselines.append(dict(subject_id=subject,group=group,session_id=session,channel=ch,
                    frequency_hz=freq,baseline_power=float(value),n_trials=int(counts.sum())))
        for name,linear,n in [('congruent',ps[0]/counts[0],counts[0]),
                               ('incongruent',ps[1]/counts[1],counts[1]),
                               ('both',ps.sum(axis=0)/counts.sum(),counts.sum())]:
            values=(10*np.log10(linear/baseline[:,:,None])).mean(axis=(1,2))
            if not np.isfinite(values).all():raise ValueError('Nonfinite Stroop normalized theta.')
            trial_counts.append(dict(subject_id=subject,group=group,session_id=session,condition=name,n_trials=int(n)))
            for ch,value in zip(canonical[0],values):
                rows.append(dict(subject_id=subject,group=group,session_id=session,condition=name,
                    channel=ch,theta_db=float(value),n_trials=int(n)))
    table=pd.DataFrame(rows);counts=pd.DataFrame(trial_counts)
    if (len(table)!=32*2*3*61 or table.duplicated(['subject_id','session_id','condition','channel']).any()
            or set(table.subject_id)!=set(EXPECTED_SUBJECTS)
            or counts.loc[counts.condition.eq('both'),'n_trials'].sum()!=mask_qc['n_retained']):
        raise ValueError('Incomplete Stroop participant/session/condition/scalp summaries.')
    table.to_csv(root/'participant_session_topographies.csv',index=False)
    pd.DataFrame(baselines).to_csv(root/'pooled_baseline_power.csv',index=False)
    counts.to_csv(root/'participant_session_condition_counts.csv',index=False)
    seal_theta_paper_outputs(root)
    print(f'Stroop summaries validated: {len(table):,} values; 61 channels, 32 paired participants; conditions pooled in linear power before dB.',flush=True)
    return table,counts


def plot_stroop_theta_topographies(cache_dir, output_dir):
    """Export condition maps plus a paired incongruent-minus-congruent contrast page."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from .theta_session_overview import plot_pooled_laplacian_theta_sessions
    root=Path(cache_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    settings=json.loads((root/'settings.json').read_text())
    if settings.get('task')!='stroop' or settings.get('conditions')!=list(CONDITIONS):
        raise ValueError('Expected validated Stroop condition cache.')
    # One scale for pre/post and one for change, shared across all three conditions.
    figures={}
    for condition in CONDITIONS:
        figures[condition]=plot_pooled_laplacian_theta_sessions(root,out,condition=condition,
            color_limit_db=1.5,change_color_limit_db=1.5)
    checks=[x for v in figures.values() for x in v['panel_checks']]
    session_max=max(max(abs(x['rendered_min_db']),abs(x['rendered_max_db']),abs(x['electrode_min_db']),abs(x['electrode_max_db']))
                    for x in checks if x['comparison']!='change')
    change_max=max(max(abs(x['rendered_min_db']),abs(x['rendered_max_db']),abs(x['electrode_min_db']),abs(x['electrode_max_db']))
                   for x in checks if x['comparison']=='change')
    session_limit=max(1.5,float(np.ceil(session_max*2)/2))
    change_limit=max(1.5,float(np.ceil(change_max*2)/2))
    contrast='incongruent_minus_congruent'
    preview=plot_pooled_laplacian_theta_sessions(root,out,condition=contrast,color_limit_db=1.5)
    contrast_max=max(max(abs(x['rendered_min_db']),abs(x['rendered_max_db']),
                         abs(x['electrode_min_db']),abs(x['electrode_max_db'])) for x in preview['panel_checks'])
    contrast_limit=max(.25,float(np.ceil(contrast_max*4)/4))
    pdf_path=out/'stroop_theta_congruent_incongruent_pooled.pdf' 
    with plt.rc_context({'pdf.fonttype':42}),PdfPages(pdf_path) as pdf:
        for condition in CONDITIONS:
            figures[condition]=plot_pooled_laplacian_theta_sessions(root,out,condition=condition,
                color_limit_db=session_limit,change_color_limit_db=change_limit,pdf_pages=pdf)
        figures[contrast]=plot_pooled_laplacian_theta_sessions(root,out,condition=contrast,
            color_limit_db=contrast_limit,pdf_pages=pdf)
    if any(x['fraction_visible_pixels_outside_scale']>0 for v in figures.values() for x in v['panel_checks']):
        raise ValueError('Stroop map color scale clips visible pixels.')
    meta=dict(pdf=str(pdf_path.resolve()),conditions=list(CONDITIONS)+[contrast],figures=figures,contrast_limit_db=contrast_limit,
              pre_post_limit_db=session_limit,change_limit_db=change_limit,
              scope='Descriptive whole-scalp maps; no ROI statistics or correlations.',
              pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest())
    (out/'figure_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Saved 24 Stroop maps: congruent/incongruent/pooled plus condition contrast; shared condition pre/post ±{session_limit:g} dB, change ±{change_limit:g} dB; contrast ±{contrast_limit:g} dB across all six contrast panels; no clipping or statistics.',flush=True)
    return meta
