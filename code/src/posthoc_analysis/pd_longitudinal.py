"""Seven-point sensor-level Pd: filtered pre/post and all decoding distractors."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import contextlib
import hashlib
import io
import json
import multiprocessing

import numpy as np
import pandas as pd

from . import eeg
from .bci import _subject_session_folders, load_decoding_analysis_file
from .config import PROJECT_ROOT, get_subject_group
from .pd_filtered import _validate_filtered_pd_waveforms
from .triggers import load_training_trigger_file

TIME = np.arange(768) / 512 - .5
LABELS = ['Pre', 'Training 1', 'Training 2', 'Training 3', 'Training 4', 'Training 5', 'Post']
EXPECTED_RUNS = {1: 6, 2: 8, 3: 8, 4: 8, 5: 6}


def collect_pd_decoding_runs(subjects, project_root=PROJECT_ROOT, *, allow_available_runs=False):
    """Validate every decoding text file and inventory EEG; never infer practice.

    Exact *_decoding_practice folders are omitted. With explicit caller consent,
    retain all *_decoding runs (including e27 S1 and extra e22/e37 S5 runs),
    and audit missing GDFs. Malformed existing files always require resolution.
    """
    subjects = list(subjects)
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError('Provide unique, nonempty participant IDs.')
    rows = []
    for subject in sorted(subjects):
        group = get_subject_group(subject)
        for session, folder in enumerate(_subject_session_folders(project_root, subject), 1):
            folders = sorted(p for p in folder.glob('*_decoding') if p.is_dir())
            if not folders:
                raise ValueError(f'{subject} session {session}: no decoding runs in {folder}.')
            if len(folders) != EXPECTED_RUNS[session]:
                message = f'{subject} session {session}: {len(folders)} decoding folders; nominal {EXPECTED_RUNS[session]}.'
                if not allow_available_runs:
                    raise ValueError(message + ' Resolve run inclusion explicitly.')
                print('RUN-COUNT AUDIT:', message, 'Keeping every available run.')
            for run_id, run in enumerate(folders, 1):
                base = run / run.name
                trigger_path = base.with_suffix('.triggers.txt')
                analysis_path = base.with_suffix('.analysis.txt')
                gdf = base.with_suffix('.gdf')
                trigger = load_training_trigger_file(trigger_path, allow_zero_latency_response=True)
                # Check physical column count before the existing typed loader.
                values = np.loadtxt(analysis_path)
                if values.shape != (60, 8) or not np.isfinite(values).all() or not np.equal(values, values.round()).all():
                    raise ValueError(f'{analysis_path}: expected finite integer array (60, 8), got {values.shape}.')
                analysis = load_decoding_analysis_file(analysis_path)
                codes = trigger.trigger.to_numpy().reshape(60, 3)[:, 1]
                if (pd.Series(codes).value_counts().to_dict() != {8: 30, 32: 15, 44: 15}
                        or not np.array_equal(analysis.task, codes != 8)
                        or not np.array_equal(analysis.distractor_position.map({0: 8, 2: 32, 4: 44}), codes)):
                    raise ValueError(f'{run}: trigger/behavior condition or side mismatch, or invalid 30/15/15 counts.')
                available = gdf.is_file() and gdf.stat().st_size > 0
                if not available:
                    if not allow_available_runs:
                        raise FileNotFoundError(f'Missing/empty EEG: {gdf}')
                    print(f'MISSING EEG AUDIT: {gdf}; text is valid, EEG unavailable.')
                rows.append(dict(subject_id=subject, group=group, session_id=session,
                                 run_id=run_id, gdf_path=str(gdf), trigger_path=str(trigger_path),
                                 analysis_path=str(analysis_path), status='available' if available else 'missing_eeg',
                                 expected_n_runs=EXPECTED_RUNS[session]))
    inventory = pd.DataFrame(rows)
    if inventory.duplicated(['subject_id', 'session_id', 'run_id']).any():
        raise ValueError('Duplicate run keys in decoding inventory.')
    print(f'Validated decoding text pairs: {len(inventory)}; available EEG: '
          f'{inventory.status.eq("available").sum()}; missing EEG: {inventory.status.eq("missing_eeg").sum()}.')
    print('Practice-name folders omitted; no accuracy, RT, feedback, or artifact rejection applied.')
    return inventory


def _validate_decoding_wave(wave, run_rows):
    """Validate one pooled session and reconcile all 30 distractors per run."""
    wave = np.asarray(wave)
    if wave.shape != (768,) or not np.isfinite(wave).all():
        raise ValueError('Decoding Pd must be a finite 768-sample waveform.')
    if abs(wave[(TIME >= -.2) & (TIME < 0)].mean()) > 1e-8:
        raise ValueError('Decoding Pd baseline residual exceeds tolerance.')
    if not run_rows or any(r['n_distractor_trials'] != 30 or r['n_right'] != 15 or r['n_left'] != 15 for r in run_rows):
        raise ValueError('Each included run must contribute all 15 right + 15 left distractors.')
    print(f'Validated pooled decoding Pd: {len(run_rows)} runs, {30*len(run_rows)} distractors, 768 samples.')


def _build_decoding_pd_cell(job):
    """Process/cache one participant/session with unchanged canonical preprocessing."""
    subject, session, records, directory, force, code_hashes = job
    available = [r for r in records if r['status'] == 'available']
    if not available:
        raise ValueError(f'{subject} session {session}: no EEG available.')
    sources = []
    for row in records:
        for column in ['gdf_path', 'trigger_path', 'analysis_path']:
            path = Path(row[column])
            stat = path.stat() if path.exists() else None
            sources.append(dict(path=str(path), size=stat.st_size if stat else None,
                                mtime_ns=stat.st_mtime_ns if stat else None))
    inputs = dict(source_files=sources, code_sha256=code_hashes,
                  method='all available decoding distractors; equal trials; canonical sensor Pd')
    directory = Path(directory)
    npz_path = directory / f'{subject}_session{session}.npz'
    meta_path = directory / f'{subject}_session{session}.json'
    if not force and npz_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata.get('inputs') == inputs:
            if hashlib.sha256(npz_path.read_bytes()).hexdigest() != metadata['sha256']:
                raise ValueError(f'Cache checksum mismatch: {npz_path}')
            with np.load(npz_path, allow_pickle=False) as saved:
                wave = saved['pd_amplitude_uv']
                if not np.array_equal(saved['time_sec'], TIME):
                    raise ValueError(f'Cache time-grid mismatch: {npz_path}')
            run_rows = metadata['run_trial_counts']
            if [r['gdf_path'] for r in run_rows] != [r['gdf_path'] for r in available]:
                raise ValueError(f'Cache run coverage mismatch: {meta_path}')
            _validate_decoding_wave(wave, run_rows)
            return subject, session, wave, run_rows, 'cache'
    summed = np.zeros(768)
    run_rows = []
    for row in available:
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                epochs = eeg.load_filter_epoch_baseline_correct_training_run(
                    row['gdf_path'], validate_paired_files=False)
                trigger = load_training_trigger_file(row['trigger_path'], allow_zero_latency_response=True)
                codes = trigger.trigger.to_numpy().reshape(60, 3)[:, 1]
                if (not np.array_equal(codes, epochs['stimulus_events'][:, 2])
                        or not np.array_equal(epochs['kept_event_samples'], epochs['stimulus_events'][:, 0])
                        or not np.array_equal(epochs['time'], TIME)
                        or tuple(epochs['baseline_indices']) != (154, 256)):
                    raise ValueError('Status/text stimulus order, epoch alignment, time grid or baseline mismatch.')
                result = eeg.compute_pd_difference_wave(epochs['baseline_corrected_epochs'],
                                                       epochs['eeg_labels'], epochs['stimulus_events'])
            trials = result['pd_trials_samples_by_trials']
            if trials.shape != (768, 30):
                raise ValueError(f'Expected all 30 distractors, got {trials.shape}.')
            summed += trials.sum(axis=1)
            run_rows.append({**row, 'n_distractor_trials': 30, 'n_right': 15, 'n_left': 15,
                             'warnings': '\n'.join(x for x in log.getvalue().splitlines() if 'WARNING' in x)})
        except Exception as exc:
            raise ValueError(f'{row["gdf_path"]}: {exc}\n{log.getvalue()}') from exc
    wave = summed / (30 * len(run_rows))
    _validate_decoding_wave(wave, run_rows)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, time_sec=TIME, pd_amplitude_uv=wave)
    meta_path.write_text(json.dumps(dict(inputs=inputs, run_trial_counts=run_rows,
                                        sha256=hashlib.sha256(npz_path.read_bytes()).hexdigest()), indent=2))
    return subject, session, wave, run_rows, 'raw EEG'


def build_seven_point_pd(filtered_results, *, project_root=PROJECT_ROOT, cache_dir=None,
                         allow_available_runs=False, workers=1, force=False):
    """Reuse filtered pre/post; pool all decoding distractors at five training points.

    Session indices 0..6 represent Pre, Training 1..5, Post; original visit IDs
    remain in session_id. Cache entries bind raw-file metadata and analysis code.
    """
    prepost = filtered_results['subject_session_pd']
    _validate_filtered_pd_waveforms(prepost, filtered_results['trials'])
    if not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer.')
    subjects = sorted(prepost.subject_id.unique())
    inventory = collect_pd_decoding_runs(subjects, project_root,
                                        allow_available_runs=allow_available_runs)
    candidates = [Path(cache_dir)] if cache_dir is not None else [
        Path(project_root) / 'analyses/eeg_pd/seven_point', eeg.REPO_ROOT / 'analyses/eeg_pd/seven_point']
    directory = next((p for p in candidates if p.exists()), candidates[0])
    print(f'Decoding waveform cache: {directory}. Checked: {[str(p) for p in candidates]}. '
          'Only missing/stale participant/session entries require raw EEG preprocessing.', flush=True)
    code_dir = Path(__file__).parent
    hashes = {name: hashlib.sha256((code_dir / name).read_bytes()).hexdigest()
              for name in ['pd_longitudinal.py', 'eeg.py', 'triggers.py', 'bci.py', 'config.py']}
    jobs = [(subject, int(session), cell.to_dict('records'), str(directory), force, hashes)
            for (subject, session), cell in inventory.groupby(['subject_id', 'session_id'], sort=True)]
    frames, counts = [], []
    if workers == 1:
        results = map(_build_decoding_pd_cell, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
        futures = [executor.submit(_build_decoding_pd_cell, job) for job in jobs]
        results = (future.result() for future in as_completed(futures))
    try:
        for subject, session, wave, run_rows, origin in results:
            frames.append(pd.DataFrame(dict(subject_id=subject, group=get_subject_group(subject),
                session_id=session, point_index=session, sample_index=np.arange(768), time_sec=TIME,
                pd_amplitude_uv=wave, n_retained_distractor_trials=30*len(run_rows),
                n_runs=len(run_rows), expected_n_runs=EXPECTED_RUNS[session],
                trial_policy='all distractors')))
            counts.extend(run_rows)
            print(f'{subject} Training {session}: {30*len(run_rows)} distractors / {len(run_rows)} runs ({origin}).', flush=True)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    endpoints = prepost.copy()
    endpoints['point_index'] = endpoints.session_id.map({1: 0, 5: 6})
    endpoints['trial_policy'] = 'behavior filtered'
    waves = pd.concat([endpoints, *frames], ignore_index=True).sort_values(['subject_id', 'point_index', 'sample_index'])
    waves['point_label'] = waves.point_index.map(dict(enumerate(LABELS)))
    if waves.point_index.isna().any() or waves.groupby('subject_id').point_index.nunique().ne(7).any():
        raise ValueError('Every participant must have exactly seven time points.')
    counts = pd.concat([pd.DataFrame(counts), inventory[inventory.status.eq('missing_eeg')]], ignore_index=True)
    directory.mkdir(parents=True, exist_ok=True)
    counts.to_csv(directory / 'run_trial_counts.csv', index=False)
    print(f'Assembled seven points for {len(subjects)} participants; filtered pre/post copied unchanged.')
    return dict(subject_point_pd=waves, run_trial_counts=counts, cache_dir=directory)


def compute_seven_point_pd_auc(results, time_window=(.2, .5)):
    """Integrate max(mean contra−ipsi, 0), then summarize participants equally."""
    waves = results['subject_point_pd']
    required = {'subject_id', 'group', 'point_index', 'sample_index', 'time_sec', 'pd_amplitude_uv',
                'session_id', 'n_retained_distractor_trials', 'n_runs', 'trial_policy'}
    if waves.empty or required - set(waves) or waves.duplicated(['subject_id', 'point_index', 'sample_index']).any():
        raise ValueError('Empty/invalid seven-point waveform schema or duplicate samples.')
    window = np.asarray(time_window, dtype=float)
    if window.shape != (2,) or not np.isfinite(window).all() or not TIME[0] <= window[0] < window[1] <= TIME[-1]:
        raise ValueError('AUC window must be finite, increasing and inside recorded epochs.')
    mask = (TIME >= window[0]) & (TIME <= window[1])
    if mask.sum() < 2:
        raise ValueError('AUC needs at least two recorded samples.')
    rows = []
    for (subject, point), cell in waves.groupby(['subject_id', 'point_index']):
        cell = cell.sort_values('sample_index')
        if (not np.array_equal(cell.sample_index, np.arange(768))
                or not np.allclose(cell.time_sec, TIME, atol=1e-12, rtol=0)
                or not np.isfinite(cell.pd_amplitude_uv).all()
                or not cell.group.eq(get_subject_group(subject)).all()
                or point not in range(7)):
            raise ValueError(f'{subject}, point {point}: invalid samples/group/time grid.')
        rows.append(dict(subject_id=subject, group=get_subject_group(subject), point_index=int(point),
                         point_label=LABELS[int(point)], session_id=int(cell.session_id.iloc[0]),
                         positive_auc_uv_sec=float(np.trapezoid(np.maximum(cell.pd_amplitude_uv.to_numpy()[mask], 0), TIME[mask])),
                         n_distractor_trials=int(cell.n_retained_distractor_trials.iloc[0]),
                         n_runs=int(cell.n_runs.iloc[0]), trial_policy=cell.trial_policy.iloc[0],
                         time_window_start_sec=window[0], time_window_end_sec=window[1], n_samples=int(mask.sum())))
    auc = pd.DataFrame(rows)
    if any(set(cell.point_index) != set(range(7)) for _, cell in auc.groupby('subject_id')):
        raise ValueError('Each participant needs all seven points.')
    summary = auc.groupby(['group', 'point_index', 'point_label'], sort=True).positive_auc_uv_sec.agg(
        mean='mean', sd='std', n='count').reset_index()
    summary['sem'] = summary.sd / np.sqrt(summary.n)
    print(f'Positive AUC: {1000*window[0]:g}–{1000*window[1]:g} ms; {mask.sum()} recorded samples '
          f'({1000*TIME[mask][0]:g}–{1000*TIME[mask][-1]:g} ms), no boundary interpolation.')
    print('Trial difference waves → participant/point mean waveform → positive AUC → group mean ± SEM.')
    print(summary.to_string(index=False))
    return dict(participant_auc=auc, group_summary=summary, time_window=tuple(window))


def plot_seven_point_pd_auc(auc_results, output_dir=eeg.FIGURES_DIR, style='poster'):
    """Export BCI/mental-rehearsal trajectories using poster or paper typography."""
    if style not in {'poster', 'paper'}:
        raise ValueError("style must be 'poster' or 'paper'.")
    poster = style == 'poster'
    eeg._prepare_plot_environment()
    import matplotlib.pyplot as plt

    data = auc_results['participant_auc']
    required = {'subject_id', 'group', 'point_index', 'positive_auc_uv_sec'}
    if data.empty or required - set(data) or data.duplicated(['subject_id', 'point_index']).any():
        raise ValueError('Invalid participant AUC schema or duplicate participant/point rows.')
    if set(data.group) != {'bci', 'control'} or not np.isfinite(data.positive_auc_uv_sec).all() or data.positive_auc_uv_sec.lt(0).any():
        raise ValueError('Plot requires finite nonnegative AUCs in both configured groups.')
    for subject, cell in data.groupby('subject_id'):
        if set(cell.point_index) != set(range(7)) or not cell.group.eq(get_subject_group(subject)).all():
            raise ValueError(f'{subject}: incomplete points or invalid group.')
    summary = data.groupby(['group', 'point_index']).positive_auc_uv_sec.agg(['mean', 'sem', 'count']).reset_index()
    if summary['count'].lt(2).any():
        raise ValueError('SEM requires at least two participants per group/point.')
    rc = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
          'font.size': 10 if poster else 7, 'axes.labelsize': 12 if poster else 7,
          'xtick.labelsize': 10 if poster else 6.5, 'ytick.labelsize': 10 if poster else 6.5,
          'legend.fontsize': 10 if poster else 6.5, 'axes.linewidth': .8 if poster else .6,
          'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(7.0, 3.8 if poster else 2.9))
        for group, label, color in [('control', 'Mental rehearsal', '#3182BD'), ('bci', 'BCI', '#C65D28')]:
            cell = summary[summary.group.eq(group)].sort_values('point_index')
            n = cell['count'].unique()
            if len(n) != 1:
                raise ValueError('Participant counts must be constant across seven points.')
            ax.errorbar(cell.point_index, cell['mean'], yerr=cell['sem'], color=color,
                        marker='o', markersize=5 if poster else 3.5,
                        linewidth=1.5 if poster else 1.1, elinewidth=.9 if poster else .7,
                        capsize=3 if poster else 2,
                        label=f'{label} (n = {n[0]})')
        ax.set_xticks(range(7), LABELS)
        ax.set_ylabel('Pd positive AUC (µV·s)')
        ax.set_xlim(-.25, 6.25)
        ax.set_ylim(bottom=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', width=.8 if poster else .6, length=4 if poster else 3)
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.15), ncols=2)
        start, stop = np.asarray(auc_results['time_window']) * 1000
        separator = '\n' if poster else ' '
        fig.text(.5, .015, f'{start:g}–{stop:g} ms; mean ± SEM.' + separator
                 + 'Pre/post: behavioral exclusions; training: all distractor trials.',
                 ha='center', fontsize=9.5 if poster else 6.5)
        fig.subplots_adjust(left=.13 if poster else .1, right=.99,
                            bottom=.23 if poster else .19, top=.84)
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for extension in ['pdf', 'png']:
            path = directory / f'pd_positive_auc_seven_points.{extension}'
            fig.savefig(path, dpi=350, bbox_inches='tight')
            paths.append(path)
    print(f'Saved {style} Pd trajectories (equal participant weights; SEM): {paths}')
    return dict(figure=fig, axes=ax, saved_paths=paths)
