"""Read-only investigation of training responses absent from GDF Status."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .consolidated import collect_all_training_runs
from .analysis import load_training_analysis_file
from .triggers import compute_training_reaction_times_gdf


def audit_missing_training_responses(training_trials, root_path=None, output_dir=None):
    """Inspect every missing RT against sample-level Status and alternative events.

    Uses the corrected trial table as an index, then independently revalidates
    each affected run against raw GDF, trigger text, and analysis files. Text
    RT locates an approximate response time only. Missingness is defined by
    response_count, so explicitly pooled text RTs do not hide absent GDF events.
    Saves durable trial/run/Status-segment QC when output_dir is provided.
    """
    import mne

    root_path = Path(root_path) if root_path is not None else PROJECT_ROOT
    keys = ['subject_id', 'session_id', 'run_id', 'trial']
    required = set(keys + ['rt_ms', 'rt_source', 'response_count', 'feedback',
                           'stimulus_sample', 'stimulus_trigger', 'text_rt_ms', 'group'])
    if not isinstance(training_trials, pd.DataFrame) or required - set(training_trials):
        raise ValueError(f'Training table must contain {sorted(required)}.')
    data = training_trials.copy()
    if data.empty or data[keys].isna().any().any() or data.duplicated(keys).any():
        raise ValueError('Training table is empty or contains missing/duplicate trial keys.')
    if not data.rt_source.isin(['gdf_status', 'trigger_text_fallback']).all() or not data.session_id.isin([1, 5]).all():
        raise ValueError('Audit requires corrected GDF Status training RTs from Sessions 1/5.')
    if 'gdf_rt_ms' not in data:
        if data.rt_source.eq('trigger_text_fallback').any():
            raise ValueError('Pooled RTs require original gdf_rt_ms for the raw audit.')
        data['gdf_rt_ms'] = data.rt_ms
    if not data.feedback.isin([1, 2, 3]).all() or not data.stimulus_trigger.isin([8, 32, 44]).all():
        raise ValueError('Invalid feedback or stimulus codes in training table.')
    for key, run in data.groupby(keys[:3]):
        if sorted(run.trial.tolist()) != list(range(1, 61)):
            raise ValueError(f'{key}: expected unique trials 1..60.')
    missing = data[data.response_count.eq(0)].copy()
    if data.response_count.gt(1).any():
        raise ValueError('Some missing RTs have ambiguous responses; this audit targets absent responses only.')
    if missing.empty:
        raise ValueError('No missing GDF responses to audit.')
    runs, collection_issues, _ = collect_all_training_runs(root_path)
    lookup = {(r['subject_id'], r['session_number'], r['run_number']): r for r in runs}
    affected_keys = missing[keys[:3]].drop_duplicates()
    print(f'Auditing {len(missing)} missing responses / {len(data)} trials in {len(affected_keys)} runs; raw data and analysis CSV are read-only.')
    rows, segments, run_rows = [], [], []
    for key, absent in missing.groupby(keys[:3], sort=True):
        if key not in lookup:
            raise ValueError(f'{key}: cannot locate matching non-practice training run; collection issues: {collection_issues}')
        info = lookup[key]
        gdf = info['trigger_file'].with_name(info['run_id'] + '.gdf')
        validated = compute_training_reaction_times_gdf(gdf, info['trigger_file'], info['analysis_file'])
        saved = data[(data.subject_id == key[0]) & (data.session_id == key[1]) & (data.run_id == key[2])].sort_values('trial')
        for column in ['gdf_rt_ms', 'stimulus_sample', 'stimulus_trigger', 'text_rt_ms', 'response_count']:
            raw_column = 'rt_ms' if column == 'gdf_rt_ms' else column
            if not np.allclose(saved[column].to_numpy(), validated[raw_column].to_numpy(), equal_nan=True, rtol=0, atol=1e-8):
                raise ValueError(f'{gdf}: saved {column} disagrees with revalidated raw run.')
        analysis = load_training_analysis_file(info['analysis_file']).sort_values('trial_index')
        if not np.array_equal(saved.feedback.to_numpy(), analysis.feedback.to_numpy()):
            raise ValueError(f'{gdf}: saved feedback disagrees with analysis file.')
        raw = mne.io.read_raw_gdf(gdf, preload=False, verbose='ERROR')
        try:
            sfreq = raw.info['sfreq']
            if sfreq != 512 or 'Status' not in raw.ch_names:
                raise ValueError(f'{gdf}: expected Status at 512 Hz.')
            values = raw.get_data(picks=['Status'])[0]
            if len(values) != raw.n_times or not np.isfinite(values).all() or not np.allclose(values, np.round(values)):
                raise ValueError(f'{gdf}: Status samples must be finite integers with recording-length alignment.')
            status = np.round(values).astype(np.int64)
            change = np.r_[0, np.flatnonzero(np.diff(status)) + 1]
            stop = np.r_[change[1:], len(status)]
            values_at_change = status[change]
            absolute = change + raw.first_samp
            alternate = mne.find_events(raw, stim_channel='Status', shortest_event=1,
                                        consecutive=True, initial_event=True, verbose='ERROR')
            # Recorded Status marker widths do not measure physical pulse duration
            # or the experiment driver's timer; acquisition can encode edges.
            marker_widths = []
            for sample in validated.stimulus_sample:
                pos = np.searchsorted(absolute, sample)
                if pos >= len(absolute) or absolute[pos] != sample:
                    raise ValueError(f'{gdf}: stimulus {sample} is not a raw Status transition.')
                marker_widths.append((stop[pos] - change[pos]) / sfreq * 1000)
            for trial in absent.itertuples(index=False):
                start = int(trial.stimulus_sample)
                end = int(validated.iloc[trial.trial].stimulus_sample) if trial.trial < 60 else raw.first_samp + raw.n_times
                window = status[start - raw.first_samp + 1:end - raw.first_samp]
                response64 = int(np.sum(window == 64))
                response_bit = int(np.sum((window & 64) != 0))
                alternative_count = int(np.sum((alternate[:, 2] == 64) & (alternate[:, 0] > start) & (alternate[:, 0] < end)))
                expected = start + float(trial.text_rt_ms) * sfreq / 1000
                expected_index = int(round(expected)) - raw.first_samp
                if not (0 <= expected_index < len(status)):
                    raise ValueError(f'{gdf}, trial {trial.trial}: text-based expected response outside recording.')
                ids = {'subject_id':key[0], 'session_id':key[1], 'run_id':key[2], 'trial':trial.trial}
                row = {**ids, 'group':trial.group, 'run_name':info['run_id'], 'gdf_path':str(gdf),
                       'feedback':trial.feedback, 'stimulus_trigger':trial.stimulus_trigger,
                       'text_rt_ms':trial.text_rt_ms, 'rt_ms':trial.rt_ms, 'rt_source':trial.rt_source,
                       'gdf_rt_ms':trial.gdf_rt_ms,
                       'stimulus_sample':start, 'next_stimulus_or_end_sample':end,
                       'recorded_stimulus_marker_ms':marker_widths[trial.trial - 1],
                       'status_at_text_expected_response':int(status[expected_index]),
                       'raw_64_sample_count':response64, 'raw_response_bit_sample_count':response_bit,
                       'alternative_extraction_response_count':alternative_count,
                       'interval_status_values':json.dumps(np.unique(window).tolist()),
                       'samples_until_recording_end':int(raw.first_samp + raw.n_times - start),
                       'raw_diagnosis':'absent_from_status' if response_bit == 0 else 'response_bit_present_needs_review'}
                rows.append(row)
                # Full trial plus 50 ms before stimulus; run-length encoding is lossless.
                visible_start = max(raw.first_samp, start - int(np.ceil(.05 * sfreq)))
                use = (absolute < end) & (stop + raw.first_samp > visible_start)
                for j in np.flatnonzero(use):
                    seg_start = max(int(absolute[j]), visible_start)
                    seg_stop = min(int(stop[j] + raw.first_samp), end)
                    segments.append({**ids, 'start_sample':seg_start,'stop_sample_exclusive':seg_stop,
                                     'start_ms_from_stimulus':(seg_start-start)/sfreq*1000,
                                     'stop_ms_from_stimulus':(seg_stop-start)/sfreq*1000,
                                     'status_code':int(values_at_change[j])})
            run_rows.append({'subject_id':key[0], 'session_id':key[1], 'run_id':key[2],
                             'run_name':info['run_id'], 'n_missing':len(absent),
                             'recorded_stimulus_marker_median_ms':float(np.median(marker_widths)),
                             'recorded_stimulus_marker_min_ms':min(marker_widths),
                             'recorded_stimulus_marker_max_ms':max(marker_widths),
                             'status_codes':json.dumps(np.unique(status).tolist())})
        finally:
            raw.close()
    trial_qc = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    run_qc = pd.DataFrame(run_rows)
    status_segments = pd.DataFrame(segments)
    if len(trial_qc) != len(missing) or trial_qc.duplicated(keys).any():
        raise ValueError('QC output did not retain exactly one record for every missing trial.')
    summaries = summarize_missing_training_responses(data)
    outputs = {'missing_trials':trial_qc,'affected_runs':run_qc,'status_segments':status_segments, **summaries}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, table in outputs.items():
            table.to_csv(output_dir / f'training_missing_response_{name}.csv',index=False)
        print(f'Saved missing-response QC tables to {output_dir}.')
    print(f'Raw Status diagnosis: {trial_qc.raw_diagnosis.value_counts().to_dict()}')
    print(f'Alternative extraction finds {trial_qc.alternative_extraction_response_count.sum()} responses. '
          'Recorded marker widths are not physical pulse/timer measurements.')
    return outputs


def summarize_missing_training_responses(training_trials):
    """Summarize missingness using all trials as denominators, without cleaning.

    Diagnostic bands use the supplied experiment driver's 50 ms timer and
    the analysis's 150 ms minimum RT; these bands do not modify primary RTs.
    """
    required = {'subject_id','session_id','group','trial','feedback','stimulus_trigger',
                'rt_ms','text_rt_ms','response_count','rt_source'}
    if not isinstance(training_trials, pd.DataFrame) or required - set(training_trials) or training_trials.empty:
        raise ValueError(f'Missingness summary requires nonempty data with {sorted(required)}.')
    data = training_trials.assign(missing_gdf_response=training_trials.response_count.eq(0))
    if not np.isfinite(data.text_rt_ms).all() or (data.text_rt_ms < 0).any():
        raise ValueError('Diagnostic text RT must be finite and nonnegative.')
    if (not data.rt_source.isin(['gdf_status', 'trigger_text_fallback']).all()
            or not data.response_count.isin([0, 1]).all()):
        raise ValueError('Summary requires corrected RT sources; ambiguous responses need a separate audit.')
    pooled = data.rt_source.eq('trigger_text_fallback')
    gdf_rt = data.get('gdf_rt_ms', data.rt_ms)
    if (not gdf_rt.isna().equals(data.response_count.eq(0))
            or (pooled & (~data.missing_gdf_response | data.text_rt_ms.ge(50))).any()
            or not np.allclose(data.rt_ms, gdf_rt.where(~pooled, data.text_rt_ms),
                               equal_nan=True, rtol=0, atol=1e-8)):
        raise ValueError('RT provenance or original GDF RT missingness disagrees with response counts.')
    data['text_rt_band'] = pd.cut(data.text_rt_ms, [-np.inf, 50, 150, np.inf], right=False,
                                 labels=['<50 ms','50–149 ms','≥150 ms'])
    tables = {}
    for name, keys in [('by_subject',['subject_id','group']),('by_session',['session_id']),
                       ('by_group_session',['group','session_id']),('by_condition',['stimulus_trigger']),
                       ('by_feedback',['feedback']),('by_trial_number',['trial']),('by_text_rt',['text_rt_band'])]:
        table = data.groupby(keys, observed=True).agg(n_trials=('rt_ms','size'), n_missing=('missing_gdf_response','sum')).reset_index()
        table['missing_percent'] = table.n_missing / table.n_trials * 100
        if table.n_trials.sum() != len(data) or table.n_missing.sum() != data.missing_gdf_response.sum():
            raise ValueError(f'{name}: missingness denominator reconciliation failed.')
        tables[name] = table
    print(f'Missing responses: {data.missing_gdf_response.sum()} / {len(data)} ({100*data.missing_gdf_response.mean():.3f}%).')
    print(tables['by_text_rt'].to_string(index=False))
    return tables
