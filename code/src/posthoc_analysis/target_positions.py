"""Target-location counts in unfiltered, non-practice training trials."""

import pandas as pd

from .analysis import TRAINING_ANALYSIS_COLUMNS, validate_training_analysis


def summarize_no_distractor_target_positions(training_trials):
    """Validate full training runs and count lateral versus midline targets.

    Position codes 2/4 are right/left (the shared spatial code used by
    distractor positions and stimulus triggers); 1/3 are the midline pair.
    No accuracy, RT, or EEG exclusions are applied. Return overall, session,
    position-code, and run summaries without modifying the input table.
    """
    required = {
        "subject_id", "session_id", "run_id", "task_type", "trial",
        "stimulus_trigger", *TRAINING_ANALYSIS_COLUMNS[1:],
    }
    missing = required.difference(training_trials.columns)
    if missing:
        raise ValueError(f"Training table lacks required columns: {sorted(missing)}")
    if training_trials.empty or training_trials[list(required)].isna().any().any():
        raise ValueError("Training table is empty or contains missing trial metadata.")
    data = training_trials.copy()
    if not data.task_type.eq("training").all():
        raise ValueError("Expected only non-practice task_type='training' rows.")
    if not data.session_id.isin([1, 5]).all():
        raise ValueError("Training sessions must be 1 or 5.")
    keys = ["subject_id", "session_id", "run_id"]
    if data.duplicated(keys + ["trial"]).any():
        raise ValueError("Duplicate subject/session/run/trial keys in training table.")
    numeric = ["session_id", "run_id", "trial", "stimulus_trigger",
               *TRAINING_ANALYSIS_COLUMNS[1:]]
    for column in numeric:
        values = pd.to_numeric(data[column], errors="raise")
        if not (values % 1 == 0).all():
            raise ValueError(f"Expected finite integer values in {column}.")
        data[column] = values.astype(int)
    if not data.task.eq(data.distractor_position.ne(0).astype(int)).all():
        raise ValueError("Task and distractor presence disagree.")
    expected_triggers = data.distractor_position.map({0: 8, 2: 32, 4: 44})
    if not data.stimulus_trigger.eq(expected_triggers).all():
        raise ValueError("Stimulus triggers and distractor position codes disagree.")
    for key, run in data.groupby(keys, sort=True):
        analysis = run.sort_values("trial").rename(columns={"trial": "trial_index"})
        validate_training_analysis(analysis[TRAINING_ANALYSIS_COLUMNS], filepath=str(key))
        if run.stimulus_trigger.value_counts().to_dict() != {8: 30, 32: 15, 44: 15}:
            raise ValueError(f"{key}: expected 30 no-distractor and 15 trials per distractor side.")
    for key, session in data.groupby(keys[:2], sort=True):
        expected = 8 if key[1] == 1 else 4
        actual = session.run_id.nunique()
        if actual != expected:
            print(f"WARNING: {key}: {actual}/{expected} training runs available; "
                  "counting available trials only (e30 Session 1 has a documented missing run).")
        if sorted(session.run_id.unique()) != list(range(1, actual + 1)):
            raise ValueError(f"{key}: run IDs must be consecutive starting at 1.")

    nd = data.loc[data.task.eq(0)].copy()
    nd["target_axis"] = nd.target_position.map({1: "midline", 2: "lateral",
                                               3: "midline", 4: "lateral"})
    overall = nd.target_axis.value_counts().reindex(["lateral", "midline"], fill_value=0)
    overall = overall.rename_axis("target_axis").rename("n_trials").to_frame()
    overall["percent"] = 100 * overall.n_trials / len(nd)
    by_session = nd.groupby(["session_id", "target_axis"]).size().unstack(fill_value=0)
    by_run = nd.groupby(keys + ["target_axis"]).size().unstack(fill_value=0)
    by_position = nd.target_position.value_counts().sort_index().rename("n_trials")
    print(f"Validated {len(data):,} trials / {len(by_run):,} runs / "
          f"{data.subject_id.nunique()} participants; practice excluded.")
    print(f"No-distractor trials: {len(nd):,}; no behavioral or EEG filtering.")
    print(overall.to_string(float_format=lambda value: f"{value:.2f}"))
    return {"overall": overall, "by_session": by_session, "by_run": by_run,
            "by_position": by_position}
