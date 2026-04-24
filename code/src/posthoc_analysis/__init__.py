"""Post-hoc analysis package for the BCI study."""

from .config import (
    get_subject_group,
    BCI_GROUP_SUBJECTS,
    CONTROL_GROUP_SUBJECTS,
    EXPECTED_SUBJECTS,
    PROJECT_ROOT,
)
from .triggers import (
    load_training_trigger_file,
    compute_training_reaction_times,
    rt_outlier_summary,
)
from .analysis import load_training_analysis_file
from .consolidated import (
    generate_consolidated_training_csv,
    validate_all_files_comprehensive,
    generate_consolidated_stroop_csv,
    validate_all_stroop_files_comprehensive,
)
try:
    from .behavioral import (
        analyze_stroop_accuracy,
        analyze_stroop_effect,
        analyze_stroop_reaction_time,
        analyze_stroop_timeout_exclusions,
        analyze_reaction_time,
        analyze_accuracy_and_timeout,
        analyze_distractor_cost,
        create_behavioral_summary_table,
        create_stroop_accuracy_summary_table,
        load_and_analyze_stroop_effect_data,
        load_and_analyze_stroop_reaction_time_data,
        validate_behavioral_summary,
        validate_stroop_accuracy_summary,
        print_behavioral_summary_checks,
        load_and_analyze_stroop_accuracy_data,
        load_and_summarize_behavioral_data,
        load_and_analyze_stroop_timeout_data,
    )
    _BEHAVIORAL_IMPORT_ERROR = None
except ImportError as exc:
    _BEHAVIORAL_IMPORT_ERROR = exc
    print(
        "WARNING: posthoc_analysis.behavioral could not be imported. "
        f"Reason: {exc}. Consolidation utilities remain available."
    )

__all__ = [
    "get_subject_group",
    "BCI_GROUP_SUBJECTS",
    "CONTROL_GROUP_SUBJECTS",
    "EXPECTED_SUBJECTS",
    "PROJECT_ROOT",
    "load_training_trigger_file",
    "compute_training_reaction_times",
    "rt_outlier_summary",
    "load_training_analysis_file",
    "generate_consolidated_training_csv",
    "validate_all_files_comprehensive",
    "generate_consolidated_stroop_csv",
    "validate_all_stroop_files_comprehensive",
]

if _BEHAVIORAL_IMPORT_ERROR is None:
    __all__.extend([
        "analyze_stroop_accuracy",
        "analyze_stroop_effect",
        "analyze_stroop_reaction_time",
        "analyze_stroop_timeout_exclusions",
        "analyze_reaction_time",
        "analyze_accuracy_and_timeout",
        "analyze_distractor_cost",
        "create_behavioral_summary_table",
        "create_stroop_accuracy_summary_table",
        "load_and_analyze_stroop_effect_data",
        "load_and_analyze_stroop_reaction_time_data",
        "validate_behavioral_summary",
        "validate_stroop_accuracy_summary",
        "print_behavioral_summary_checks",
        "load_and_analyze_stroop_accuracy_data",
        "load_and_summarize_behavioral_data",
        "load_and_analyze_stroop_timeout_data",
    ])
