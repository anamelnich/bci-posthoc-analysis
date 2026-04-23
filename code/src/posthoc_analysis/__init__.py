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
from .behavioral import (
    create_behavioral_summary_table,
    validate_behavioral_summary,
    print_behavioral_summary_checks,
    load_and_summarize_behavioral_data,
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
    "create_behavioral_summary_table",
    "validate_behavioral_summary",
    "print_behavioral_summary_checks",
    "load_and_summarize_behavioral_data",
]
