"""Trigger parsing utilities for training and decoding runs."""

from pathlib import Path

import warnings

import numpy as np
import pandas as pd

TRAINING_TRIGGER_COUNT = 180
TRAINING_TRIGGERS_PER_TRIAL = 3
TRAINING_TRIALS = TRAINING_TRIGGER_COUNT // TRAINING_TRIGGERS_PER_TRIAL
FS = 512
FIXATION_CODE = 4
RESPONSE_CODE = 64
STIMULUS_CODES = {
    8: "no_distractor",
    32: "distractor_right",
    44: "distractor_left",
}

TRAINING_COLUMNS = ["trial", "trigger", "time"]


def load_training_trigger_file(filepath, allow_zero_latency_response=False):
    """Load and validate a training trigger text file.

    Parameters
    ----------
    filepath : str or Path
        Path to a `triggers.txt` file for a training or decoding run.
    allow_zero_latency_response : bool
        Permit a response timestamp equal to its stimulus timestamp. This is
        appropriate when text RT is not an analysis input (e.g. GDF-derived RT).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['trial', 'trigger', 'time']; training time is milliseconds.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Trigger file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        dtype=float,
    )

    if df.shape[1] != len(TRAINING_COLUMNS):
        raise ValueError(f"{filepath}: expected 3 columns, got {df.shape[1]}.")
    df.columns = TRAINING_COLUMNS
    if not np.isfinite(df.to_numpy()).all() or (df % 1 != 0).any().any():
        raise ValueError(f"{filepath}: trigger fields must be finite integers.")
    df = df.astype("int64")

    validate_training_triggers(
        df,
        filepath,
        allow_zero_latency_response=allow_zero_latency_response,
    )
    return df


def validate_training_triggers(df, filepath=None, allow_zero_latency_response=False):
    """Validate that a training trigger DataFrame matches expected structure."""
    if df.shape[0] != TRAINING_TRIGGER_COUNT:
        raise ValueError(
            f"Expected {TRAINING_TRIGGER_COUNT} rows in training trigger file"
            f"{' ' + str(filepath) if filepath else ''}, got {df.shape[0]} rows."
        )

    if list(df.columns) != TRAINING_COLUMNS:
        raise ValueError(
            f"Unexpected columns in trigger file{' ' + str(filepath) if filepath else ''}. "
            f"Expected {TRAINING_COLUMNS}, got {list(df.columns)}."
        )

    if not all(pd.api.types.is_integer_dtype(df[c]) for c in TRAINING_COLUMNS):
        raise ValueError(f"{filepath or 'Trigger table'}: all fields must be integers.")
    if df.isna().any().any() or (df["time"] < 0).any():
        raise ValueError(f"{filepath or 'Trigger table'}: missing fields or negative millisecond timestamps.")

    trial_counts = df["trial"].value_counts().sort_index()
    if len(trial_counts) != TRAINING_TRIALS:
        raise ValueError(
            f"Expected {TRAINING_TRIALS} unique trials in training trigger file"
            f"{' ' + str(filepath) if filepath else ''}, got {len(trial_counts)}."
        )

    invalid_counts = trial_counts[trial_counts != TRAINING_TRIGGERS_PER_TRIAL]
    if not invalid_counts.empty:
        raise ValueError(
            "Each trial must have exactly 3 triggers. "
            f"Found invalid trial row counts: {invalid_counts.to_dict()}."
        )

    trial_numbers = trial_counts.index.tolist()
    expected_trials = list(range(1, TRAINING_TRIALS + 1))
    if trial_numbers != expected_trials:
        raise ValueError(
            "Training trigger trial numbers must be consecutive 1..60. "
            f"Found: {trial_numbers[:5]}...{trial_numbers[-5:]}"
        )

    for trial, group in df.groupby("trial", sort=True):
        if len(group) != TRAINING_TRIGGERS_PER_TRIAL:
            raise ValueError(f"Trial {trial} does not contain exactly 3 trigger rows.")

        triggers = group["trigger"].tolist()
        if triggers[0] != FIXATION_CODE:
            raise ValueError(
                f"Trial {trial} expected fixation trigger {FIXATION_CODE} first, got {triggers[0]}.")
        if triggers[1] not in STIMULUS_CODES:
            raise ValueError(
                f"Trial {trial} expected stimulus trigger in {list(STIMULUS_CODES)} second, got {triggers[1]}.")
        if triggers[2] != RESPONSE_CODE:
            raise ValueError(
                f"Trial {trial} expected response trigger {RESPONSE_CODE} third, got {triggers[2]}.")

        times = group["time"].tolist()
        valid_time_order = times[0] < times[1] < times[2]
        tolerated_zero_latency_response = (
            allow_zero_latency_response and times[0] < times[1] == times[2]
        )
        if not (valid_time_order or tolerated_zero_latency_response):
            raise ValueError(
                f"Trial {trial} trigger times are not strictly increasing: {times}."
            )


def compute_training_reaction_times_gdf(gdf_filepath, trigger_file=None, analysis_file=None):
    """Compute training RT from Status, validating same-run text/analysis alignment.

    Require 512 Hz, 60 stimuli (30/15/15 codes 8/32/44), and identical stimulus
    order in text. Search strictly between successive stimuli, or recording end.
    Missing/multiple responses produce NaN RT; multiple responses also flag an
    alignment error. This extractor never replaces a GDF RT with text;
    consolidation applies apply_training_text_rt_fallback() separately.
    Practice is excluded.
    """
    import mne
    from .analysis import load_training_analysis_file

    gdf_filepath = Path(gdf_filepath)
    if not gdf_filepath.name.endswith("_training.gdf"):
        raise ValueError(f"Expected a non-practice *_training.gdf: {gdf_filepath}")
    trigger_file = Path(trigger_file) if trigger_file is not None else gdf_filepath.with_suffix(".triggers.txt")
    analysis_file = Path(analysis_file) if analysis_file is not None else gdf_filepath.with_suffix(".analysis.txt")
    for path, suffix in ((gdf_filepath, ".gdf"), (trigger_file, ".triggers.txt"), (analysis_file, ".analysis.txt")):
        if path.resolve() != gdf_filepath.with_suffix(suffix).resolve():
            raise ValueError(f"Cross-run file alignment: {path} does not belong to {gdf_filepath}.")
        if not path.is_file():
            raise FileNotFoundError(f"Required training RT source missing: {path}; no text RT fallback.")
    # Text timing is diagnostic, not the RT source; zero text latency is allowed.
    triggers = load_training_trigger_file(trigger_file, allow_zero_latency_response=True)
    analysis = load_training_analysis_file(analysis_file)
    text_stim = triggers[triggers["trigger"].isin(STIMULUS_CODES)].reset_index(drop=True)
    text_response = triggers[triggers["trigger"] == RESPONSE_CODE].reset_index(drop=True)
    if not np.array_equal(text_stim["trial"], np.arange(1, 61)):
        raise ValueError(f"{trigger_file}: stimuli must occur in trial order 1..60.")
    if not np.array_equal(text_response["trial"], text_stim["trial"]):
        raise ValueError(f"{trigger_file}: response trial order does not match stimuli.")

    raw = mne.io.read_raw_gdf(gdf_filepath, preload=False, verbose="ERROR")
    try:
        sfreq = float(raw.info["sfreq"])
        if sfreq != FS:
            raise ValueError(f"{gdf_filepath}: expected 512 Hz, got {sfreq}.")
        if "Status" not in raw.ch_names or raw.n_times <= 0:
            raise ValueError(f"{gdf_filepath}: missing Status channel or empty recording.")
        events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose="ERROR")
        if events.ndim != 2 or events.shape[1] != 3 or not len(events):
            raise ValueError(f"{gdf_filepath}: expected a nonempty N x 3 Status event array.")
        if not np.issubdtype(events.dtype, np.integer) or np.any(np.diff(events[:, 0]) <= 0):
            raise ValueError(f"{gdf_filepath}: Status samples must be ordered, unique integers.")
        recording_end = raw.first_samp + raw.n_times
        if np.any(events[:, 0] < raw.first_samp) or np.any(events[:, 0] >= recording_end):
            raise ValueError(f"{gdf_filepath}: Status samples outside recording bounds.")
        unexpected = sorted(int(code) for code in set(events[:, 2]) - {4, 8, 32, 44, 64})
        if unexpected:
            warnings.warn(f"{gdf_filepath}: ignoring unrelated Status codes {unexpected}.", RuntimeWarning)
        stimuli = events[np.isin(events[:, 2], list(STIMULUS_CODES))]
        counts = dict(zip(*np.unique(stimuli[:, 2], return_counts=True)))
        if counts != {8: 30, 32: 15, 44: 15}:
            raise ValueError(f"{gdf_filepath}: expected 60 stimuli (8:30, 32:15, 44:15), got {counts}.")
        if not np.array_equal(stimuli[:, 2], text_stim["trigger"]):
            bad = (np.flatnonzero(stimuli[:, 2] != text_stim["trigger"].to_numpy()) + 1).tolist()
            raise ValueError(f"{gdf_filepath}: GDF/text stimulus order mismatch at trials {bad}.")
        expected_codes = analysis["distractor_position"].map({0: 8, 2: 32, 4: 44})
        if not np.array_equal(stimuli[:, 2], expected_codes) or not np.array_equal(
            (stimuli[:, 2] != 8).astype(int), analysis["task"]
        ):
            raise ValueError(f"{analysis_file}: task/distractor side does not match Status stimulus order.")

        response_samples = events[events[:, 2] == RESPONSE_CODE, 0]
        rows = []
        for index, (sample, _, code) in enumerate(stimuli):
            end = stimuli[index + 1, 0] if index + 1 < len(stimuli) else recording_end
            responses = response_samples[(response_samples > sample) & (response_samples < end)]
            count = len(responses)
            response_sample = int(responses[0]) if count == 1 else np.nan
            rt_samples = response_sample - int(sample)
            rows.append({
                "trial": index + 1,
                "stimulus_trigger": int(code),
                "condition": STIMULUS_CODES[int(code)],
                "stimulus_sample": int(sample),
                "response_sample": response_sample,
                "rt_samples": rt_samples,
                "rt_ms": rt_samples / sfreq * 1000.0,
                "rt_source": "gdf_status",
                "sfreq_hz": sfreq,
                "response_count": count,
                "alignment_error": count > 1,
                "text_rt_ms": int(text_response.iloc[index]["time"]) - int(text_stim.iloc[index]["time"]),
            })
        result = pd.DataFrame(rows)
        for count_label, mask in (("missing", result["response_count"] == 0), ("multiple", result["alignment_error"])):
            if mask.any():
                warnings.warn(f"{gdf_filepath}: {count_label} responses at trials {result.loc[mask, 'trial'].tolist()}; GDF-only RT set to NaN.", RuntimeWarning)
        orphan_responses = response_samples[response_samples <= stimuli[0, 0]]
        if len(orphan_responses):
            warnings.warn(f"{gdf_filepath}: responses outside trial intervals at samples {orphan_responses.tolist()}.", RuntimeWarning)
        print(f"{gdf_filepath.name}: validated 60 aligned trials, 30/15/15 conditions, {sfreq:g} Hz; "
              f"{result['rt_ms'].notna().sum()} valid RTs, {(result['response_count'] == 0).sum()} missing, "
              f"{result['alignment_error'].sum()} alignment errors.")
        return result
    finally:
        raw.close()


def apply_training_text_rt_fallback(training_trials):
    """Pool validated sub-50 ms text RTs when the GDF response is absent.

    This explicit exception covers the audited 50 ms trigger-lockout cases.
    Preserve GDF-only RT in gdf_rt_ms and retain missing response/sample fields.
    Never fill ambiguous responses or missing responses outside this pattern.
    Accepts a reconstructed run or consolidated table; safe to call repeatedly.
    """
    required = {'rt_ms', 'rt_source', 'rt_samples', 'response_sample',
                'response_count', 'alignment_error', 'text_rt_ms', 'sfreq_hz'}
    if not isinstance(training_trials, pd.DataFrame) or training_trials.empty or required - set(training_trials):
        raise ValueError(f'Text fallback requires nonempty, GDF-validated training trials with {sorted(required)}.')
    data = training_trials.copy()
    if not data.rt_source.isin(['gdf_status', 'trigger_text_fallback']).all():
        raise ValueError('Text fallback requires corrected GDF provenance, not legacy text-derived RTs.')
    if 'gdf_rt_ms' not in data:
        if not data.rt_source.eq('gdf_status').all():
            raise ValueError('Previously pooled RTs require the original gdf_rt_ms column.')
        data['gdf_rt_ms'] = data.rt_ms
    if (not data.sfreq_hz.eq(512).all()
            or not np.isfinite(data.response_count).all()
            or (data.response_count < 0).any()
            or not data.response_count.eq(data.response_count.round()).all()
            or not data.alignment_error.isin([True, False]).all()
            or not data.alignment_error.eq(data.response_count.gt(1)).all()):
        raise ValueError('Expected 512 Hz, nonnegative integer response counts, and matching alignment-error flags.')
    unique = data.response_count.eq(1)
    if (not data.gdf_rt_ms.notna().equals(unique)
            or not data.rt_samples.notna().equals(unique)
            or not data.response_sample.notna().equals(unique)
            or not np.isfinite(data.loc[unique, 'gdf_rt_ms']).all()
            or (data.loc[unique, 'gdf_rt_ms'] <= 0).any()
            or not np.allclose(data.gdf_rt_ms, data.rt_samples / data.sfreq_hz * 1000,
                               equal_nan=True, rtol=0, atol=1e-8)):
        raise ValueError('GDF RT/sample fields must agree and be missing unless exactly one response exists.')
    if not np.isfinite(data.text_rt_ms).all() or (data.text_rt_ms < 0).any():
        raise ValueError('Aligned text RTs must be finite, nonnegative milliseconds.')
    fallback = data.response_count.eq(0) & data.text_rt_ms.lt(50)
    already_pooled = data.rt_source.eq('trigger_text_fallback')
    if (already_pooled & ~fallback).any() or not np.allclose(
            data.rt_ms, data.gdf_rt_ms.where(~already_pooled, data.text_rt_ms),
            equal_nan=True, rtol=0, atol=1e-8):
        raise ValueError('Existing RT values/provenance disagree with GDF or the permitted text fallback.')
    data['rt_ms'] = data.gdf_rt_ms.where(~fallback, data.text_rt_ms)
    data['rt_source'] = np.where(fallback, 'trigger_text_fallback', 'gdf_status')
    if fallback.any():
        print(f'Text RT fallback: {fallback.sum()} absent GDF responses filled with aligned text milliseconds <50; '
              'gdf_rt_ms/response_sample/rt_samples remain NaN. These fail the existing 150 ms minimum.')
    else:
        print('Text RT fallback: no eligible absent responses; GDF RTs unchanged.')
    return data
