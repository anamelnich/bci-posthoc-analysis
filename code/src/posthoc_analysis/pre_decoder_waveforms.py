"""All-trial pre-task sensor comparisons for the two decoder class contrasts."""
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
from .analysis import load_training_analysis_file
from .config import PROJECT_ROOT, EXPECTED_SUBJECTS, get_subject_group
from .triggers import load_training_trigger_file

TIME = np.arange(768) / 512 - .5
CONDITIONS = ('no_distractor', 'distractor_right', 'distractor_left')
CODES = (8, 32, 44)


def compute_pre_condition_waveforms(epoch_result):
    """Return PO8−PO7 sums/counts for all three conditions, before sign display.

    Shape is conditions × 768 samples. Pool sums across runs before dividing
    by counts; the right-comparison display negates BOTH of its class means.
    """
    required = {'baseline_corrected_epochs', 'eeg_labels', 'stimulus_events',
                'kept_event_samples', 'time', 'baseline_indices', 'sample_rate'}
    if required.difference(epoch_result):
        raise ValueError(f'Missing epoch fields: {sorted(required.difference(epoch_result))}')
    labels = list(epoch_result['eeg_labels'])
    epochs = np.asarray(epoch_result['baseline_corrected_epochs'])
    events = np.asarray(epoch_result['stimulus_events'])
    if (epochs.shape != (len(labels), 768, 60) or not np.isfinite(epochs).all()
            or events.shape != (60, 3) or len(set(labels)) != len(labels)):
        raise ValueError('Expected finite channels × 768 × 60 epochs and 60 aligned events with unique labels.')
    if not {'PO7', 'PO8'}.issubset(labels):
        raise ValueError('PO7 and PO8 must both be present.')
    if (epoch_result['sample_rate'] != 512 or not np.array_equal(epoch_result['time'], TIME)
            or tuple(epoch_result['baseline_indices']) != (154, 256)
            or not np.array_equal(epoch_result['kept_event_samples'], events[:, 0])):
        raise ValueError('Unexpected sampling, time grid, baseline, or event/epoch alignment.')
    difference = epochs[labels.index('PO8')] - epochs[labels.index('PO7')]
    if not np.allclose(difference[154:256].mean(axis=0), 0, atol=1e-8, rtol=0):
        raise ValueError('PO8−PO7 trial baseline residual exceeds tolerance.')
    counts = np.array([(events[:, 2] == code).sum() for code in CODES])
    if not np.array_equal(counts, [30, 15, 15]):
        raise ValueError(f'Expected no/right/left counts [30, 15, 15], found {counts}.')
    sums = np.stack([difference[:, events[:, 2] == code].sum(axis=1) for code in CODES])
    print('Validated all 60 trials: no/right/left = 30/15/15; canonical PO8−PO7 sums.')
    return sums, counts


def _validate_subject_cache(waves, counts, n_runs):
    """Check complete condition waveforms, expected counts, and baseline."""
    if (waves.shape != (3, 768) or not np.isfinite(waves).all()
            or not np.array_equal(counts, np.array([30, 15, 15]) * n_runs)
            or not np.allclose(waves[:, 154:256].mean(axis=1), 0, atol=1e-8, rtol=0)):
        raise ValueError('Invalid cached shape, trial counts, finite values, or baseline.')
    print(f'Validated {n_runs} pre runs; no/right/left counts = {counts.tolist()}.', flush=True)


def _build_pre_subject(job):
    """Validate or rebuild one compact participant cache with source provenance."""
    subject, paths, directories, force, code_hashes = job
    sources = []
    for gdf in map(Path, paths):
        for path in (gdf, gdf.with_suffix('.analysis.txt'), gdf.with_suffix('.triggers.txt')):
            stat = path.stat()
            if not stat.st_size:
                raise ValueError(f'Empty source file: {path}')
            sources.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    inputs = dict(source_files=sources, code_sha256=code_hashes,
                  method='Session 1 training; all trials; native sensor scaling; equal trials within participant')
    for directory in map(Path, directories):
        npz, meta = directory / f'{subject}.npz', directory / f'{subject}.json'
        if not force and npz.is_file() and meta.is_file():
            metadata = json.loads(meta.read_text())
            if metadata.get('inputs') == inputs:
                if hashlib.sha256(npz.read_bytes()).hexdigest() != metadata['sha256']:
                    raise ValueError(f'Cache checksum mismatch: {npz}')
                with np.load(npz, allow_pickle=False) as saved:
                    if set(saved.files) != {'time_sec', 'po8_minus_po7', 'counts'} or not np.array_equal(saved['time_sec'], TIME):
                        raise ValueError(f'Invalid cache fields/time: {npz}')
                    waves, counts = saved['po8_minus_po7'], saved['counts']
                _validate_subject_cache(waves, counts, len(paths))
                if [r['gdf_path'] for r in metadata['run_audit']] != paths:
                    raise ValueError(f'Cache audit/run coverage differs: {meta}')
                print(f'{subject}: reusing validated pre-task cache.', flush=True)
                return subject, waves, counts, metadata['run_audit']
    print(f'{subject}: no matching cache; rebuilding {len(paths)} raw pre runs.', flush=True)
    summed, counts, audit = np.zeros((3, 768)), np.zeros(3, dtype=int), []
    for run_id, path in enumerate(map(Path, paths), 1):
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                # Check physical file width/integer values before typed loaders.
                raw_analysis = np.loadtxt(path.with_suffix('.analysis.txt'))
                if (raw_analysis.shape != (60, 8) or not np.isfinite(raw_analysis).all()
                        or not np.equal(raw_analysis, raw_analysis.round()).all()):
                    raise ValueError('Analysis file must be a finite integer array (60, 8).')
                analysis = load_training_analysis_file(path.with_suffix('.analysis.txt'))
                trigger = load_training_trigger_file(path.with_suffix('.triggers.txt'))
                codes = trigger.trigger.to_numpy().reshape(60, 3)[:, 1]
                if (not np.array_equal(analysis.task, codes != 8)
                        or not np.array_equal(analysis.distractor_position.map({0: 8, 2: 32, 4: 44}), codes)):
                    raise ValueError('Behavior and text stimulus conditions disagree.')
                epochs = eeg.load_filter_epoch_baseline_correct_training_run(path)
                if not np.array_equal(codes, epochs['stimulus_events'][:, 2]):
                    raise ValueError('Text and GDF stimulus ordering disagree.')
                run_sums, run_counts = compute_pre_condition_waveforms(epochs)
            summed += run_sums
            counts += run_counts
            audit.append(dict(subject_id=subject, session_id=1, run_id=run_id, gdf_path=str(path),
                              n_no_distractor=30, n_right=15, n_left=15,
                              warnings='\n'.join(line for line in log.getvalue().splitlines() if 'WARNING' in line)))
        except Exception as exc:
            raise ValueError(f'{path}: {exc}\n{log.getvalue()}') from exc
    waves = summed / counts[:, None]
    _validate_subject_cache(waves, counts, len(paths))
    directory = Path(directories[0])
    directory.mkdir(parents=True, exist_ok=True)
    npz, meta = directory / f'{subject}.npz', directory / f'{subject}.json'
    np.savez_compressed(npz, time_sec=TIME, po8_minus_po7=waves, counts=counts)
    meta.write_text(json.dumps(dict(inputs=inputs, run_audit=audit,
                                   sha256=hashlib.sha256(npz.read_bytes()).hexdigest()), indent=2))
    print(f'{subject}: saved all-trial pre-task waveforms.', flush=True)
    return subject, waves, counts, audit


def build_pre_decoder_waveforms(subjects=None, *, project_root=PROJECT_ROOT, cache_dir=None,
                                workers=1, force=False):
    """Pool every pre-task trial, preserving both groups and all three conditions.

    Reuse only source/code/checksum-validated caches. Historical condition means
    lack this provenance; filtered Pd caches omit no-distractor trials. No
    behavioral/artifact exclusion, class balancing, pruning, or xDAWN is applied.
    """
    subjects = sorted(EXPECTED_SUBJECTS if subjects is None else list(subjects))
    if not subjects or len(set(subjects)) != len(subjects) or set(subjects).difference(EXPECTED_SUBJECTS):
        raise ValueError('Provide unique configured subject IDs.')
    if not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer.')
    directories = [Path(cache_dir)] if cache_dir is not None else [
        Path(project_root) / 'analyses/eeg_pd/pre_decoder_conditions',
        eeg.REPO_ROOT / 'analyses/eeg_pd/pre_decoder_conditions']
    print(f'Cache lookup: {[str(p) for p in directories]}. Historical unfiltered averages lack '
          'source-bound provenance; filtered Pd caches do not contain the requested classes.')
    code_dir = Path(__file__).parent
    hashes = {name: hashlib.sha256((code_dir / name).read_bytes()).hexdigest()
              for name in ['pre_decoder_waveforms.py', 'eeg.py', 'analysis.py', 'triggers.py', 'config.py']}
    jobs = []
    for subject in subjects:
        paths = eeg._get_training_run_gdf_files_for_session(subject, 1, project_root, allow_incomplete=subject == 'e30')
        if len(paths) != (7 if subject == 'e30' else 8):
            raise ValueError(f'{subject}: unexpected pre-run count {len(paths)}.')
        jobs.append((subject, list(map(str, paths)), list(map(str, directories)), force, hashes))
    executor = None
    if workers == 1:
        results = map(_build_pre_subject, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
        futures = [executor.submit(_build_pre_subject, job) for job in jobs]
        results = (future.result() for future in as_completed(futures))
    frames, audits = [], []
    try:
        for subject, waves, counts, audit in results:
            for i, condition in enumerate(CONDITIONS):
                frames.append(pd.DataFrame(dict(subject_id=subject, group=get_subject_group(subject),
                    session_id=1, condition=condition, sample_index=np.arange(768), time_sec=TIME,
                    po8_minus_po7_uv=waves[i], n_trials=int(counts[i]), n_runs=len(audit))))
            audits.extend(audit)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    waves = pd.concat(frames, ignore_index=True).sort_values(['subject_id', 'condition', 'sample_index'])
    audit = pd.DataFrame(audits).sort_values(['subject_id', 'run_id'])
    print(f'Complete: {len(subjects)} participants / {len(audit)} pre runs / {len(audit)*60:,} trials.')
    print('All trials retained; equal trial weights within participant, equal participant weights in plots.')
    return dict(subject_waveforms=waves, run_audit=audit)


def plot_pre_decoder_waveforms(results, *, output_dir=eeg.FIGURES_DIR, style='poster'):
    """Plot two sensor class comparisons; right is PO7−PO8, left is PO8−PO7."""
    if style not in {'paper', 'poster'}:
        raise ValueError('style must be paper or poster.')
    data = results['subject_waveforms']
    required = {'subject_id', 'group', 'condition', 'sample_index', 'time_sec', 'po8_minus_po7_uv', 'n_trials'}
    if required.difference(data) or data.empty or data[list(required)].isna().any().any():
        raise ValueError('Missing or empty waveform fields.')
    subjects = sorted(data.subject_id.unique())
    if len(subjects) < 2 or set(data.condition) != set(CONDITIONS):
        raise ValueError('SEM requires at least two participants and all three conditions.')
    arrays = {}
    for condition in CONDITIONS:
        cell = data[data.condition.eq(condition)]
        if cell.duplicated(['subject_id', 'sample_index']).any():
            raise ValueError('Duplicate waveform samples.')
        for subject, wave in cell.groupby('subject_id'):
            wave = wave.sort_values('sample_index')
            if (not np.array_equal(wave.sample_index, np.arange(768))
                    or not np.array_equal(wave.time_sec, TIME) or wave.n_trials.nunique() != 1
                    or wave.n_trials.iloc[0] <= 0):
                raise ValueError(f'{subject}/{condition}: invalid time grid or trial count.')
        arrays[condition] = cell.pivot(index='subject_id', columns='sample_index', values='po8_minus_po7_uv').reindex(subjects).to_numpy()
        if arrays[condition].shape != (len(subjects), 768) or not np.isfinite(arrays[condition]).all():
            raise ValueError(f'Incomplete/non-finite condition: {condition}')
    eeg._prepare_plot_environment()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    sizes = dict(label=12, tick=10, panel=12) if style == 'poster' else dict(label=7, tick=7, panel=9)
    rows = []
    for side, condition, sign in [('right', 'distractor_right', -1), ('left', 'distractor_left', 1)]:
        for name in [condition, 'no_distractor']:
            values = sign * arrays[name]
            rows.append(pd.DataFrame(dict(side=side, condition=name, time_sec=TIME,
                mean_uv=values.mean(axis=0), sem_uv=values.std(axis=0, ddof=1)/np.sqrt(len(subjects)))))
    summary = pd.concat(rows, ignore_index=True)
    shown = summary[summary.time_sec.between(-.2, .7)]
    extent = max(abs(shown.mean_uv - shown.sem_uv).max(), abs(shown.mean_uv + shown.sem_uv).max())
    ylim = (-1.12 * max(extent, .1), 1.12 * max(extent, .1))
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'font.size': sizes['tick'], 'axes.labelsize': sizes['label'], 'axes.linewidth': .7,
                         'xtick.direction': 'out', 'ytick.direction': 'out', 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.65), sharex=True, sharey=True)
        for ax, side, panel, formula in zip(axes, ['right', 'left'], ['a', 'b'], ['PO7 − PO8', 'PO8 − PO7']):
            ax.axvspan(.2, .5, color='.94', zorder=0)
            ax.axhline(0, color='.65', lw=.6, zorder=1)
            ax.axvline(0, color='.65', lw=.6, zorder=1)
            for condition in ['distractor_' + side, 'no_distractor']:
                cell = summary[(summary.side == side) & (summary.condition == condition)]
                color = '#C65D28' if condition != 'no_distractor' else '#526D82'
                linestyle = '-' if condition != 'no_distractor' else '--'
                ax.fill_between(TIME, cell.mean_uv-cell.sem_uv, cell.mean_uv+cell.sem_uv, color=color, alpha=.17, lw=0)
                ax.plot(TIME, cell.mean_uv, color=color, lw=1.4, ls=linestyle)
            ax.text(-.14, 1.05, panel, transform=ax.transAxes, weight='bold', fontsize=sizes['panel'])
            ax.set_title(f'{side.capitalize()} distractor comparison\n{formula}', fontsize=sizes['label'], pad=8)
            ax.set(xlim=(-.2, .7), ylim=ylim, xlabel='Time from stimulus (s)')
            ax.set_xticks([-.2, 0, .2, .4, .6])
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(width=.7, length=3, labelsize=sizes['tick'])
        axes[0].set_ylabel('Difference amplitude (µV)')
        fig.legend(handles=[Line2D([], [], color='#C65D28', lw=1.4, label='Distractor on indicated side'),
                            Line2D([], [], color='#526D82', lw=1.4, ls='--', label='No distractor')],
                   loc='lower center', bbox_to_anchor=(.52, .075), ncol=2, frameon=False, fontsize=sizes['tick'])
        fig.text(.52, .025, f'Pre · n = {len(subjects)} participants · mean ± SEM · all trials', ha='center', fontsize=sizes['tick'])
        fig.subplots_adjust(left=.12, right=.985, top=.79, bottom=.32, wspace=.19)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for extension in ('png', 'pdf'):
            path = output_dir / f'pre_decoder_condition_waveforms.{extension}'
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            paths.append(path)
    counts = data.drop_duplicates(['subject_id', 'condition']).groupby('condition').n_trials.sum()
    print(f'Plot validated: {len(subjects)} equally weighted participants; total trial counts:\n{counts}')
    print(f'Shared y limits: {ylim}; no smoothing; native amplitude scaling matches the existing EEG pipeline.')
    print('The same no-distractor trials are sign-reversed between panels; all target positions included.')
    print(f'Saved: {paths}')
    return dict(figure=fig, summary=summary, trial_counts=counts, saved_paths=paths)
