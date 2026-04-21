"""Trigger parsing utilities for training and decoding runs."""

from pathlib import Path

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


def load_training_trigger_file(filepath):
    """Load and validate a training trigger text file.

    Parameters
    ----------
    filepath : str or Path
        Path to a `triggers.txt` file for a training or decoding run.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['trial', 'trigger', 'time'].
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Trigger file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=TRAINING_COLUMNS,
        dtype={"trial": int, "trigger": int, "time": int},
    )

    validate_training_triggers(df, filepath)
    return df


def validate_training_triggers(df, filepath=None):
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
        if not (times[0] < times[1] < times[2]):
            raise ValueError(
                f"Trial {trial} trigger times are not strictly increasing: {times}."
            )


def compute_training_reaction_times(triggers, fs=FS):
    """Compute reaction time for each trial from a validated trigger table."""
    if isinstance(triggers, (str, Path)):
        triggers = load_training_trigger_file(triggers)
    elif not isinstance(triggers, pd.DataFrame):
        raise TypeError("triggers must be a file path or pandas DataFrame.")

    validate_training_triggers(triggers)

    rows = []
    for trial, group in triggers.groupby("trial", sort=True):
        fixation, stimulus, response = group.iloc[0], group.iloc[1], group.iloc[2]
        stimulus_code = int(stimulus["trigger"])
        rt_samples = int(response["time"]) - int(stimulus["time"])
        if rt_samples < 0:
            raise ValueError(
                f"Negative RT for trial {trial}: response time {response['time']} < stimulus time {stimulus['time']}.")

        rows.append({
            "trial": int(trial),
            "stimulus_trigger": stimulus_code,
            "condition": STIMULUS_CODES[stimulus_code],
            "response_time": int(response["time"]),
            "stimulus_time": int(stimulus["time"]),
            "rt_samples": rt_samples,
            "rt_ms": (rt_samples / fs) * 1000.0,
        })

    return pd.DataFrame(rows)


def rt_outlier_summary(rt_df, n_std=2.0):
    """Compute RT outlier summary for training reaction times.

    Parameters
    ----------
    rt_df : pandas.DataFrame
        DataFrame returned by compute_training_reaction_times.
    n_std : float, optional
        Number of standard deviations to use for the threshold, by default 2.0.

    Returns
    -------
    dict
        Summary including threshold, outlier count, total trials, and percent outlier.
    """
    if not isinstance(rt_df, pd.DataFrame):
        raise TypeError("rt_df must be a pandas DataFrame.")
    if "rt_ms" not in rt_df.columns:
        raise ValueError("rt_df must contain an 'rt_ms' column.")

    mean_rt = rt_df["rt_ms"].mean()
    std_rt = rt_df["rt_ms"].std(ddof=0)
    threshold_low = mean_rt - n_std * std_rt
    threshold_high = mean_rt + n_std * std_rt

    outliers = rt_df[(rt_df["rt_ms"] < threshold_low) | (rt_df["rt_ms"] > threshold_high)]
    outlier_count = len(outliers)
    total_trials = len(rt_df)
    percent_outlier = (outlier_count / total_trials) * 100 if total_trials else 0.0

    return {
        "mean_rt_ms": mean_rt,
        "std_rt_ms": std_rt,
        "threshold_low_ms": threshold_low,
        "threshold_high_ms": threshold_high,
        "outlier_count": outlier_count,
        "total_trials": total_trials,
        "percent_outlier": percent_outlier,
        "outlier_trials": outliers[["trial", "condition", "rt_ms"]].copy(),
    }
