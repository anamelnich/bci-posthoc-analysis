"""GDF-only decoding RT, conservative behavioral alignment, and next-trial QC.

No text-trigger timing or training RT fallback is used. Public functions return
inspectable tables; the notebook owns orchestration and output destinations.
"""
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

from . import config
from .bci import (
    REPO_ROOT, DECODING_ANALYSIS_OUTPUT_COLUMNS, _subject_session_folders,
    _expected_decoding_analysis_runs, load_decoding_analysis_file,
    validate_bci_consolidated_dataframe,
)

KEYS = ['subject_id', 'session_id', 'run_id', 'trial_id']
RUN_KEYS = KEYS[:-1]
FEEDBACK = {1: 'positive', 0: 'negative', 3: 'ambivalent'}
TYPES = {1: 'distractor_present', 0: 'distractor_absent'}
EXCLUSIONS = ['incorrect', 'timeout', 'missing_invalid_rt', 'below_150_ms', 'outlier_3sd']
CACHE_VERSION = 1


def load_feedback_rt_input(csv_path=None, root_path=None):
    """Resolve consolidated input in documented order; audit fields and keys."""
    root = Path(root_path) if root_path is not None else config.PROJECT_ROOT
    paths = ([Path(csv_path)] if csv_path is not None else
             [root / 'analyses/all_subjects_bci.csv', REPO_ROOT / 'analyses/all_subjects_bci.csv'])
    source = next((p for p in paths if p.is_file()), None)
    if source is None:
        raise FileNotFoundError(f'BCI CSV not found; checked: {paths}')
    df = pd.read_csv(source)
    required = set(KEYS + ['group', 'decoding_task', 'decoding_feedback', 'decoding_bci_output'])
    if df.empty or required - set(df):
        raise ValueError(f'{source}: empty table or missing columns {required - set(df)}')
    if df[KEYS].isna().any().any() or df.duplicated(KEYS).any():
        raise ValueError(f'{source}: missing or duplicate subject/session/run/trial keys.')
    for col, valid in [('session_id', range(1, 6)), ('trial_id', range(1, 61))]:
        if not df[col].isin(valid).all():
            raise ValueError(f'{source}: invalid {col}.')
    if not (np.isfinite(df.run_id) & df.run_id.gt(0) & df.run_id.eq(df.run_id.round())).all():
        raise ValueError(f'{source}: run_id must be a positive integer.')
    expected = {s: 'experimental' for s in config.BCI_GROUP_SUBJECTS}
    expected.update({s: 'control' for s in config.CONTROL_GROUP_SUBJECTS})
    if not df.subject_id.map(expected).eq(df.group).all():
        raise ValueError(f'{source}: group assignment differs from config or unknown subject.')
    audit = validate_bci_consolidated_dataframe(df)
    ranges = {'decoding_trial_index': range(1, 61), 'decoding_task': [0, 1],
              'decoding_feedback': [1, 2, 3], 'decoding_bci_output': [0, 1, 3],
              'decoding_target_position': [1, 2, 3, 4],
              'decoding_distractor_position': [0, 2, 4], 'decoding_dot_side': [0, 1]}
    for column in DECODING_ANALYSIS_OUTPUT_COLUMNS:
        if column not in df:
            continue
        original = df[column]
        numeric = pd.to_numeric(original, errors='coerce')
        invalid = original.notna() & (~np.isfinite(numeric) | numeric.ne(numeric.round()))
        invalid |= original.notna() & (~numeric.isin(ranges[column]) if column in ranges else numeric.le(0))
        if invalid.any():
            message = f'{column}: {int(invalid.sum())} invalid cells; retained as missing for technical-loss logging.'
            warnings.warn(message)
            audit['issues'].append(message)
        df[column] = numeric.mask(invalid)
    if 'decoding_trial_index' in df:
        mismatch = df.decoding_trial_index.notna() & df.decoding_trial_index.ne(df.trial_id)
        if mismatch.any():
            warnings.warn(f'{int(mismatch.sum())} decoding_trial_index mismatches; raw matching will reject these runs.')
    print(f'Feedback RT input: {source}; {len(df):,} rows. Missing/invalid behavioral '
          'cells will be logged as technical losses, never imputed.')
    return df, source, audit


def reconstruct_decoding_events(events, sfreq, recording_end, first_sample=0):
    """Anchor 60 trials to stimuli and bound responses by the next fixation.

    A missing or duplicate intervening fixation makes the affected trial's
    response interval uncertain. RT remains NaN. Missing stimuli invalidate
    the run because ordinal alignment cannot be recovered safely.
    """
    events = np.asarray(events)
    if (events.ndim != 2 or events.shape[1] != 3 or not len(events)
            or not np.issubdtype(events.dtype, np.integer)
            or np.any(np.diff(events[:, 0]) <= 0)):
        raise ValueError('Expected nonempty chronological integer N x 3 events.')
    if sfreq != 512 or recording_end <= first_sample:
        raise ValueError(f'Expected 512 Hz and nonempty recording; got {sfreq} Hz.')
    if (events[:, 0] < first_sample).any() or (events[:, 0] >= recording_end).any():
        raise ValueError('Event sample outside recording bounds.')
    stimuli = events[np.isin(events[:, 2], [8, 32, 44])]
    if len(stimuli) != 60:
        raise ValueError(f'Expected 60 stimulus anchors; found {len(stimuli)}. No trial shifting.')
    fixations = events[events[:, 2] == 4, 0]
    responses = events[events[:, 2] == 64, 0]
    rows = []
    used = set()
    for i, (sample, _, code) in enumerate(stimuli):
        next_stim = stimuli[i + 1, 0] if i < 59 else recording_end
        next_fix = fixations[(fixations > sample) & (fixations < next_stim)]
        prior_stim = stimuli[i - 1, 0] if i else first_sample - 1
        prior_fix = fixations[(fixations > prior_stim) & (fixations < sample)]
        reason = 'ok'
        # The last trial is explicitly bounded by the recording end.
        stop = int(next_fix[0]) if i < 59 and len(next_fix) == 1 else int(next_stim)
        if len(prior_fix) != 1 or (i < 59 and len(next_fix) != 1):
            reason = 'ambiguous_fixation_boundary'
        candidates = responses[(responses > sample) & (responses < stop)]
        used.update(int(x) for x in candidates)
        if reason == 'ok':
            reason = 'missing_response' if not len(candidates) else ('multiple_responses' if len(candidates) > 1 else 'ok')
        response = float(candidates[0]) if reason == 'ok' else np.nan
        rows.append(dict(trial_id=i + 1, stimulus_trigger=int(code), stimulus_sample=int(sample),
                         response_sample=response, response_count=len(candidates), response_stop_sample=stop,
                         rt_ms=(response - sample) / sfreq * 1000, rt_reason=reason,
                         sfreq_hz=sfreq, rt_source='gdf_status'))
    trials = pd.DataFrame(rows)
    counts = {str(int(k)): int(v) for k, v in zip(*np.unique(events[:, 2], return_counts=True))}
    qc = dict(n_events=len(events), n_stimuli=60, n_fixations=len(fixations),
              n_responses=len(responses), n_valid_rt=int(trials.rt_ms.notna().sum()),
              n_missing_response=int(trials.rt_reason.eq('missing_response').sum()),
              n_ambiguous_rt=int(trials.rt_reason.isin(['multiple_responses', 'ambiguous_fixation_boundary']).sum()),
              n_orphan_responses=len(set(map(int, responses)) - used), event_counts=json.dumps(counts))
    print(f'Status reconstruction: 60 stimuli; {qc["n_valid_rt"]} RTs; '
          f'{qc["n_missing_response"]} missing; {qc["n_ambiguous_rt"]} ambiguous.')
    return trials, qc


def load_decoding_gdf_rt(gdf_path):
    """Read raw Status only using the established MNE event-onset convention."""
    import mne
    path = Path(gdf_path)
    if not path.is_file() or not path.name.endswith('_decoding.gdf') or path.stat().st_size == 0:
        raise ValueError(f'Expected nonempty, non-practice decoding GDF: {path}')
    raw = mne.io.read_raw_gdf(path, preload=False, verbose='ERROR')
    try:
        names = [n for n in raw.ch_names if n.strip().lower() == 'status']
        if len(names) != 1:
            raise ValueError(f'{path}: no unique labeled Status channel; refusing unlabeled fallback.')
        channel = names[0]
        values = raw.get_data(picks=[channel])[0]
        if (values.shape != (raw.n_times,) or not np.isfinite(values).all()
                or not np.equal(values, np.round(values)).all() or (values < 0).any()):
            raise ValueError(f'{path}: malformed/noninteger/negative Status values.')
        events = mne.find_events(raw, stim_channel=channel, shortest_event=1, verbose='ERROR')
        # Verify event samples against actual onsets, not sustained-value samples.
        indices = events[:, 0] - raw.first_samp
        if (indices <= 0).any() or not np.array_equal(values[indices], events[:, 2]) or np.any(values[indices] == values[indices - 1]):
            raise ValueError(f'{path}: extracted event does not match raw Status transition.')
        trials, qc = reconstruct_decoding_events(events, float(raw.info['sfreq']),
                                                raw.first_samp + raw.n_times, raw.first_samp)
        qc.update(status_channel=channel, status_index=raw.ch_names.index(channel),
                  n_channels=len(raw.ch_names), n_samples=raw.n_times, sfreq_hz=raw.info['sfreq'])
        if len(raw.ch_names) != 67 or raw.ch_names.index(channel) != 66:
            warnings.warn(f'{path}: nonstandard channel layout; used labeled Status: {qc}.')
        event_table = pd.DataFrame(events, columns=['sample', 'previous_code', 'code'])
        print(f'Validated GDF: {path.name}; {raw.info["sfreq"]:g} Hz, Status index {qc["status_index"]}.')
        return trials, event_table, qc
    finally:
        raw.close()


def reconstruct_and_align_feedback_rt(bci, root_path=None, cache_dir=None, force=False):
    """Match raw analysis fingerprints to consolidated runs, then align GDF.

    Exact agreement on every available behavioral field, not guessed offsets,
    identifies a unique same-subject/session run. This handles the documented
    practice exceptions without duplicating subject IDs. Unmatched files and
    consolidated runs remain visible. The cache is invalidated by input content,
    source-file size/mtime, or reconstruction version changes.
    """
    root = Path(root_path) if root_path is not None else config.PROJECT_ROOT
    if bci.empty or set(KEYS + ['group']) - set(bci) or bci.duplicated(KEYS).any():
        raise ValueError('Expected nonempty uniquely keyed consolidated input.')
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(bci, index=True).values.tobytes()).hexdigest()
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache and not force and (cache / 'reconstruction_provenance.json').is_file():
        meta = json.loads((cache / 'reconstruction_provenance.json').read_text())
        valid = meta.get('version') == CACHE_VERSION and meta.get('input_sha256') == fingerprint and meta.get('root') == str(root.resolve())
        for item in meta.get('sources', []):
            p = Path(item['path'])
            valid = valid and p.exists() and [p.stat().st_size, p.stat().st_mtime_ns] == item['signature']
        valid = valid and meta.get('inventory') == sorted(str(p) for s in bci.subject_id.unique() for p in (root / s).glob('*/*_decoding/*') if p.suffix == '.gdf' or p.name.endswith('.analysis.txt'))
        if valid:
            tables = {}
            for name, digest in meta['tables'].items():
                path = cache / f'{name}.csv'
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    valid = False
                    break
                tables[name] = pd.read_csv(path)
            if valid:
                aligned = tables['aligned_trials']
                if aligned.duplicated(KEYS).any() or not aligned.rt_source.eq('gdf_status').all():
                    raise ValueError('Cached alignment invariants failed.')
                print(f'Reusing verified GDF reconstruction: {len(aligned):,} aligned trials from {cache}.')
                return tables
    print('No current validated decoding-RT cache; reconstructing raw GDF Status (no text timing).')
    logs, all_rt, all_events, aligned_runs, sources = [], [], [], [], []
    match_cols = [c for c in DECODING_ANALYSIS_OUTPUT_COLUMNS if c in bci]
    for subject in sorted(bci.subject_id.unique()):
        try:
            sessions = _subject_session_folders(root, subject)
        except (ValueError, FileNotFoundError) as exc:
            warnings.warn(str(exc))
            logs.append(dict(subject_id=subject, status='session_mapping_failed', issue=str(exc)))
            continue
        for session_id, folder in enumerate(sessions, 1):
            blocks = [(int(k), v.sort_values('trial_id')) for k, v in bci.loc[
                bci.subject_id.eq(subject) & bci.session_id.eq(session_id)].groupby('run_id')]
            folders = sorted(folder.glob('*_decoding'))
            expected = _expected_decoding_analysis_runs(subject, session_id)
            if len(folders) != expected:
                logs.append(dict(subject_id=subject, session_id=session_id, status='folder_count_warning',
                                 issue=f'{len(folders)} decoding folders; expected {expected}. Fingerprint matching used.'))
            for run_folder in folders:
                info = dict(subject_id=subject, session_id=session_id, run_folder=run_folder.name,
                            group='experimental' if subject in config.BCI_GROUP_SUBJECTS else 'control')
                analysis_path = run_folder / f'{run_folder.name}.analysis.txt'
                gdfs = list(run_folder.glob('*.gdf'))
                info['gdf_path'] = str(gdfs[0]) if len(gdfs) == 1 else ''
                for p in [analysis_path, *gdfs]:
                    if p.is_file():
                        sources.append(dict(path=str(p), signature=[p.stat().st_size, p.stat().st_mtime_ns]))
                matched = []
                try:
                    behavior = load_decoding_analysis_file(analysis_path).add_prefix('decoding_')
                    for rid, block in blocks:
                        if len(block) != 60 or block.trial_id.tolist() != list(range(1, 61)):
                            continue
                        lhs = block[match_cols].reset_index(drop=True)
                        # Trial indices and the observed behavioral fingerprint establish identity.
                        if not lhs['decoding_trial_index'].notna().all() or not lhs['decoding_task'].notna().any():
                            continue
                        observed = lhs.notna()
                        if ((lhs == behavior[match_cols]) | ~observed).all().all():
                            matched.append((rid, block))
                    if len(matched) != 1:
                        info.update(status='unmatched_behavioral_run', issue=f'Found {len(matched)} exact consolidated matches; no run numbering inferred.')
                    else:
                        info['run_id'] = matched[0][0]
                except (ValueError, OSError, pd.errors.ParserError) as exc:
                    info.update(status='missing_invalid_analysis', issue=str(exc))
                try:
                    if len(gdfs) != 1:
                        raise FileNotFoundError(f'Expected one GDF in {run_folder}; found {len(gdfs)}.')
                    rt, events, qc = load_decoding_gdf_rt(gdfs[0])
                    info.update(qc)
                    for table in (rt, events):
                        for k in ['subject_id', 'session_id', 'run_folder', 'gdf_path']:
                            table[k] = info[k]
                    all_rt.append(rt)
                    all_events.append(events)
                    if len(matched) == 1:
                        block = matched[0][1]
                        observed_task = block.decoding_task.notna().to_numpy()
                        if not np.array_equal(rt.stimulus_trigger.ne(8).to_numpy()[observed_task].astype(int), block.decoding_task.to_numpy()[observed_task]):
                            raise ValueError('GDF stimulus condition order differs from consolidated decoding_task.')
                        if 'decoding_distractor_position' in block:
                            expected_codes = block.decoding_distractor_position.map({0: 8, 2: 32, 4: 44})
                            observed = expected_codes.notna().to_numpy()
                            if not np.array_equal(rt.stimulus_trigger.to_numpy()[observed], expected_codes.to_numpy()[observed]):
                                raise ValueError('GDF stimulus side differs from behavioral distractor position.')
                        merged = block.merge(rt.drop(columns=['subject_id', 'session_id']), on='trial_id', validate='one_to_one')
                        aligned_runs.append(merged)
                        info.update(status='aligned', issue='')
                except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
                    info.update(status='missing_gdf' if isinstance(exc, FileNotFoundError) else 'invalid_gdf_alignment', issue=str(exc))
                logs.append(info)
                if info['status'] != 'aligned':
                    warnings.warn(f'{subject} session {session_id} {run_folder.name}: {info["status"]}: {info["issue"]}')
        print(f'Finished {subject}: {sum(x.get("status") == "aligned" for x in logs if x["subject_id"] == subject)} aligned runs.', flush=True)
    aligned = pd.concat(aligned_runs, ignore_index=True) if aligned_runs else pd.DataFrame(columns=[*bci.columns, 'rt_ms', 'rt_reason', 'rt_source'])
    # More than one raw folder matching the same consolidated run is ambiguous.
    duplicates = aligned.duplicated(KEYS, keep=False)
    duplicate_keys = aligned.loc[duplicates, RUN_KEYS].drop_duplicates()
    for row in duplicate_keys.to_dict('records'):
        logs.append(dict(**row, status='duplicate_raw_match', issue='Excluded every duplicate match.'))
    aligned = aligned.loc[~duplicates].copy()
    found = set(map(tuple, aligned[RUN_KEYS].drop_duplicates().to_numpy()))
    for key, block in bci.groupby(RUN_KEYS):
        if key not in found:
            logs.append(dict(zip(RUN_KEYS, key), group=block.group.iloc[0], status='unmatched_consolidated_run',
                             issue='No reliably aligned non-practice GDF/behavioral run.', n_trials=len(block),
                             n_potential_pairs=int(block.trial_id.sort_values().diff().eq(1).sum())))
    tables = dict(aligned_trials=aligned, reconstruction_trials=pd.concat(all_rt, ignore_index=True) if all_rt else pd.DataFrame(columns=['trial_id']),
                  status_events=pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(columns=['sample']),
                  reconstruction_log=pd.DataFrame(logs))
    print(f'GDF audit: {len(all_rt)} reconstructed runs; {len(aligned):,} aligned trials; '
          f'{len(found)} aligned runs. Statuses: {tables["reconstruction_log"].status.value_counts().to_dict()}')
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        digests = {}
        for name, table in tables.items():
            path = cache / f'{name}.csv'
            table.to_csv(path, index=False)
            digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory = sorted(str(p) for s in bci.subject_id.unique() for p in (root / s).glob('*/*_decoding/*') if p.suffix == '.gdf' or p.name.endswith('.analysis.txt'))
        (cache / 'reconstruction_provenance.json').write_text(json.dumps(dict(version=CACHE_VERSION,
            input_sha256=fingerprint, root=str(root.resolve()), sources=sources, inventory=inventory, tables=digests), indent=2))
    return tables


def build_feedback_rt_pairs(aligned):
    """Build within-run consecutive pairs; control ignores previous feedback."""
    required = set(KEYS + ['group', 'decoding_bci_output', 'decoding_feedback', 'decoding_task', 'rt_ms'])
    if required - set(aligned) or aligned.duplicated(KEYS).any():
        raise ValueError(f'Invalid aligned trial table; missing {required - set(aligned)} or duplicated keys.')
    data = aligned.sort_values(KEYS).copy()
    grouped = data.groupby(RUN_KEYS, sort=False)
    data['previous_trial_id'] = grouped.trial_id.shift()
    data['previous_bci_feedback'] = grouped.decoding_bci_output.shift()
    data = data.rename(columns=dict(trial_id='current_trial_id', decoding_task='current_trial_type',
                                    decoding_feedback='current_behavioral_outcome', rt_ms='current_rt_ms'))
    data = data.loc[data.current_trial_id.ne(1)].copy()
    data['technical_reason'] = 'eligible'
    data.loc[~data.current_trial_id.sub(data.previous_trial_id).eq(1), 'technical_reason'] = 'nonconsecutive_pair'
    eligible = data.technical_reason.eq('eligible')
    data.loc[eligible & (~data.current_trial_type.isin([0, 1]) | ~data.current_behavioral_outcome.isin([1, 2, 3])), 'technical_reason'] = 'missing_invalid_behavior'
    eligible = data.technical_reason.eq('eligible')
    data.loc[eligible & data.group.eq('experimental') & ~data.previous_bci_feedback.isin(FEEDBACK), 'technical_reason'] = 'missing_invalid_previous_feedback'
    technical = data.loc[~data.technical_reason.eq('eligible')].copy()
    pairs = data.loc[data.technical_reason.eq('eligible')].copy()
    if (pairs.groupby(RUN_KEYS).size() > 59).any():
        raise ValueError('More than 59 within-run pairs.')
    # No control category is constructed or used; retain NaN as unconditional.
    pairs.loc[pairs.group.eq('control'), 'previous_bci_feedback'] = np.nan
    print(f'Validly aligned candidates: {pairs.groupby("group").size().to_dict()}; '
          f'technical pair losses: {technical.technical_reason.value_counts().to_dict()}.')
    return pairs, technical


def clean_feedback_rt_pairs(pairs, roster=None):
    """Sequential RT exclusions, one-pass 3-SD trimming, and >50% subject loss."""
    required = {'subject_id', 'group', 'session_id', 'current_trial_type', 'current_behavioral_outcome', 'current_rt_ms'}
    if required - set(pairs) or not pairs.current_behavioral_outcome.isin([1, 2, 3]).all():
        raise ValueError('Candidate pairs must have valid behavioral metadata.')
    data = pairs.copy()
    data['exclusion_reason'] = 'retained'
    data.loc[data.current_behavioral_outcome.eq(2), 'exclusion_reason'] = 'incorrect'
    data.loc[data.current_behavioral_outcome.eq(3), 'exclusion_reason'] = 'timeout'
    valid_rt = np.isfinite(data.current_rt_ms) & data.current_rt_ms.gt(0)
    data.loc[data.exclusion_reason.eq('retained') & ~valid_rt, 'exclusion_reason'] = 'missing_invalid_rt'
    data.loc[data.exclusion_reason.eq('retained') & data.current_rt_ms.lt(150), 'exclusion_reason'] = 'below_150_ms'
    keys = ['subject_id', 'session_id', 'current_trial_type']
    strata = data.loc[data.exclusion_reason.eq('retained')].groupby(keys).current_rt_ms.agg(['count', 'mean', 'std']).reset_index()
    # Include strata with zero correct/valid observations in the skip audit.
    strata = data[keys].drop_duplicates().merge(strata, on=keys, how='left', validate='one_to_one')
    strata['count'] = strata['count'].fillna(0).astype(int)
    strata['trim_status'] = np.where(strata['count'].lt(2), 'skipped_n_lt_2', np.where(strata['std'].eq(0), 'skipped_zero_sd', 'applied'))
    strata['lower_ms'] = strata['mean'] - 3 * strata['std']
    strata['upper_ms'] = strata['mean'] + 3 * strata['std']
    data = data.merge(strata, on=keys, how='left', validate='many_to_one')
    outlier = (data.exclusion_reason.eq('retained') & data.trim_status.eq('applied') &
               (data.current_rt_ms.lt(data.lower_ms) | data.current_rt_ms.gt(data.upper_ms)))
    data.loc[outlier, 'exclusion_reason'] = 'outlier_3sd'
    if roster is None:
        roster = pairs[['subject_id', 'group']].drop_duplicates()
    if roster.subject_id.duplicated().any() or not roster.group.isin(['experimental', 'control']).all():
        raise ValueError('Roster must have unique subjects and valid group assignments.')
    reports = []
    for subject in roster.itertuples(index=False):
        block = data.loc[data.subject_id.eq(subject.subject_id)]
        counts = block.exclusion_reason.value_counts()
        n = len(block)
        kept = int(counts.get('retained', 0))
        loss = 100 * (n - kept) / n if n else np.nan
        reports.append(dict(subject_id=subject.subject_id, group=subject.group, candidate_pairs=n,
                            **{x: int(counts.get(x, 0)) for x in EXCLUSIONS}, total_excluded=n - kept,
                            exclusion_percentage=loss, retained_pairs=kept,
                            subject_excluded=n == 0 or loss > 50,
                            subject_exclusion_reason='zero_candidates' if n == 0 else ('over_50_percent' if loss > 50 else 'included')))
    report = pd.DataFrame(reports)
    data = data.merge(report[['subject_id', 'subject_excluded']], on='subject_id', validate='many_to_one')
    retained = data.loc[data.exclusion_reason.eq('retained') & ~data.subject_excluded].copy()
    summaries = []
    for group in ['experimental', 'control']:
        for stage in ['before_subject_exclusion', 'after_subject_exclusion']:
            block = data.loc[data.group.eq(group) & (True if stage.startswith('before') else ~data.subject_excluded)]
            counts = block.exclusion_reason.value_counts()
            for reason in EXCLUSIONS + ['retained']:
                n = int(counts.get(reason, 0))
                summaries.append(dict(group=group, stage=stage, reason=reason, n=n, denominator=len(block), percent=100*n/len(block) if len(block) else np.nan))
        group_report = report.loc[report.group.eq(group)]
        print(f'\n{group}: included {(~group_report.subject_excluded).sum()}; excluded {group_report.subject_excluded.sum()}. '
              f'>50% loss: {group_report.loc[group_report.exclusion_percentage.gt(50), "subject_id"].tolist()}; '
              f'zero candidates: {group_report.loc[group_report.candidate_pairs.eq(0), "subject_id"].tolist()}')
        print('Five highest exclusion percentages:\n' + group_report.sort_values('exclusion_percentage', ascending=False).head(5).to_string(index=False))
    if int(report.candidate_pairs.sum()) != len(data) or int(report.retained_pairs.sum()) != data.exclusion_reason.eq('retained').sum():
        raise ValueError('Exclusion counts fail conservation check.')
    summary = pd.DataFrame(summaries)
    print(summary.to_string(index=False))
    print(f'3-SD strata: {strata.trim_status.value_counts().to_dict()}; final retained trials: {len(retained):,}.')
    return dict(pairs_with_exclusions=data, retained_trials=retained, subject_exclusions=report,
                exclusion_summary=summary, outlier_strata=strata)


def summarize_feedback_rt_quality(reconstruction, technical_pairs):
    """Separate run-level technical audit from candidate-pair behavioral losses."""
    log = reconstruction['reconstruction_log']
    aligned = reconstruction['aligned_trials']
    if 'status' not in log or aligned.duplicated(KEYS).any():
        raise ValueError('Invalid reconstruction log or duplicate aligned trials.')
    summary = log.groupby('status', dropna=False).size().rename('n_records').reset_index()
    summary['note'] = 'Audit records can overlap; do not sum as lost runs or trials.'
    lost_runs = log.loc[log.status.eq('unmatched_consolidated_run')].copy()
    if lost_runs.duplicated(RUN_KEYS).any():
        raise ValueError('Technical loss ledger duplicates a consolidated run.')
    pair_summary = technical_pairs.groupby(['group', 'technical_reason']).size().rename('n_pairs').reset_index()
    raw = reconstruction['reconstruction_trials']
    n_runs = raw[['subject_id', 'session_id', 'run_folder']].drop_duplicates().shape[0] if len(raw) else 0
    print(f'Processed {n_runs} GDF runs / {len(raw):,} stimulus-anchored trials; '
          f'{len(aligned):,} aligned trials in {len(aligned[RUN_KEYS].drop_duplicates())} runs.')
    print('Technical run audit:\n' + summary.to_string(index=False))
    print('Unmatched consolidated runs (unique lost-run ledger):\n' + lost_runs.to_string(index=False))
    print('Technical pair losses after run alignment:\n' + pair_summary.to_string(index=False))
    return dict(technical_run_summary=summary, unmatched_consolidated_runs=lost_runs,
                technical_pair_summary=pair_summary, technical_pair_log=technical_pairs)
