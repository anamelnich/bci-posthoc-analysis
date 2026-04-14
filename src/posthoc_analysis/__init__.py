"""Python tools for CNBI attention-distraction posthoc EEG analysis."""

from .data_io import (
    RunPaths,
    discover_runs,
    read_analysis_txt,
    read_triggers_txt,
    read_stroop_behoutput,
    drop_practice_trials,
)
from .preprocessing import EpochConfig, make_epochs_from_gdf, posterior_and_lateralized_features, read_gdf_array
from .modeling import (
    DecoderModel,
    train_side_decoder,
    predict_decoder,
    classify_three_way,
    cross_validated_side_decoding,
)
from .stats import permutation_test_multiclass_accuracy, reaction_time_from_triggers
from .pipeline import run_project_pipeline

__all__ = [
    "RunPaths",
    "discover_runs",
    "read_analysis_txt",
    "read_triggers_txt",
    "read_stroop_behoutput",
    "drop_practice_trials",
    "EpochConfig",
    "make_epochs_from_gdf",
    "posterior_and_lateralized_features",
    "read_gdf_array",
    "DecoderModel",
    "train_side_decoder",
    "predict_decoder",
    "classify_three_way",
    "cross_validated_side_decoding",
    "permutation_test_multiclass_accuracy",
    "reaction_time_from_triggers",
    "run_project_pipeline",
]
