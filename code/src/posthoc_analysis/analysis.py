"""Analysis file parsing utilities for training and decoding runs."""

from pathlib import Path

import pandas as pd

TRAINING_ANALYSIS_ROWS = 60
TRAINING_ANALYSIS_COLUMNS = [
    "trial_index",
    "task",
    "feedback",
    "target_position",
    "distractor_position",
    "dot_side",
    "intertrial_interval_ms",
    "bci_output",
]

# Valid values
TASK_VALUES = {0, 1}  # 0=no distractor, 1=distractor
FEEDBACK_VALUES = {1, 2, 3}  # 1=correct, 2=incorrect, 3=timeout
POSITION_VALUES = {1, 2, 3, 4}
DISTRACTOR_POSITION_VALUES = {0, 2, 4}  # 0=no distractor, 2/4=position (right/left)
DOT_SIDE_VALUES = {0, 1}  # 0=left, 1=right
BCI_OUTPUT_TRAINING = {99}


def load_training_analysis_file(filepath):
    """Load and validate a training analysis text file.

    Parameters
    ----------
    filepath : str or Path
        Path to an `analysis.txt` file for a training run.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns as specified in TRAINING_ANALYSIS_COLUMNS.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Analysis file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=TRAINING_ANALYSIS_COLUMNS,
        dtype={
            "trial_index": int,
            "task": int,
            "feedback": int,
            "target_position": int,
            "distractor_position": int,
            "dot_side": int,
            "intertrial_interval_ms": int,
            "bci_output": int,
        },
    )

    validate_training_analysis(df, filepath)
    return df


def validate_training_analysis(df, filepath=None):
    """Validate that a training analysis DataFrame matches expected structure."""
    if df.shape[0] != TRAINING_ANALYSIS_ROWS:
        raise ValueError(
            f"Expected {TRAINING_ANALYSIS_ROWS} rows in training analysis file"
            f"{' ' + str(filepath) if filepath else ''}, got {df.shape[0]} rows."
        )

    if list(df.columns) != TRAINING_ANALYSIS_COLUMNS:
        raise ValueError(
            f"Unexpected columns in analysis file{' ' + str(filepath) if filepath else ''}. "
            f"Expected {TRAINING_ANALYSIS_COLUMNS}, got {list(df.columns)}."
        )

    # Check trial indices are 1 to 60
    expected_trials = list(range(1, TRAINING_ANALYSIS_ROWS + 1))
    if df["trial_index"].tolist() != expected_trials:
        raise ValueError(
            f"Trial indices must be consecutive 1..{TRAINING_ANALYSIS_ROWS}. "
            f"Found: {df['trial_index'].tolist()[:5]}...{df['trial_index'].tolist()[-5:]}"
        )

    # Validate value ranges
    if not df["task"].isin(TASK_VALUES).all():
        invalid = df[~df["task"].isin(TASK_VALUES)]["task"].unique()
        raise ValueError(f"Invalid task values: {invalid}. Expected {TASK_VALUES}.")

    if not df["feedback"].isin(FEEDBACK_VALUES).all():
        invalid = df[~df["feedback"].isin(FEEDBACK_VALUES)]["feedback"].unique()
        raise ValueError(f"Invalid feedback values: {invalid}. Expected {FEEDBACK_VALUES}.")

    if not df["target_position"].isin(POSITION_VALUES).all():
        invalid = df[~df["target_position"].isin(POSITION_VALUES)]["target_position"].unique()
        raise ValueError(f"Invalid target_position values: {invalid}. Expected {POSITION_VALUES}.")

    if not df["distractor_position"].isin(DISTRACTOR_POSITION_VALUES).all():
        invalid = df[~df["distractor_position"].isin(DISTRACTOR_POSITION_VALUES)]["distractor_position"].unique()
        raise ValueError(f"Invalid distractor_position values: {invalid}. Expected {DISTRACTOR_POSITION_VALUES}.")

    if not df["dot_side"].isin(DOT_SIDE_VALUES).all():
        invalid = df[~df["dot_side"].isin(DOT_SIDE_VALUES)]["dot_side"].unique()
        raise ValueError(f"Invalid dot_side values: {invalid}. Expected {DOT_SIDE_VALUES}.")

    # For training, BCI output should always be 99
    if not df["bci_output"].isin(BCI_OUTPUT_TRAINING).all():
        invalid = df[~df["bci_output"].isin(BCI_OUTPUT_TRAINING)]["bci_output"].unique()
        raise ValueError(f"Invalid BCI output values for training: {invalid}. Expected {BCI_OUTPUT_TRAINING}.")

    # Intertrial interval should be positive
    if (df["intertrial_interval_ms"] <= 0).any():
        invalid = df[df["intertrial_interval_ms"] <= 0]["intertrial_interval_ms"].tolist()
        raise ValueError(f"Intertrial intervals must be positive. Found: {invalid}.")