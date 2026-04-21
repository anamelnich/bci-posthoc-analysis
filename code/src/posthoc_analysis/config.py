"""Study configuration for subject group assignments."""

from pathlib import Path

BCI_GROUP_SUBJECTS = [
    "e21", "e22", "e25", "e26", "e29", "e30", "e31", "e32",
    "e38", "e39", "e41", "e44", "e46", "e49", "e51", "e55",
]

CONTROL_GROUP_SUBJECTS = [
    "e23", "e24", "e27", "e33", "e36", "e37", "e40", "e42",
    "e43", "e45", "e47", "e48", "e50", "e52", "e53", "e54",
]

# Complete list of all subjects that should be in the study.
# Update this if subjects are added or removed.
EXPECTED_SUBJECTS = sorted(BCI_GROUP_SUBJECTS + CONTROL_GROUP_SUBJECTS)

def _validate_subject_groups():
    """Validate that group assignment lists are complete and non-overlapping."""
    expected_count = 16

    if len(BCI_GROUP_SUBJECTS) != expected_count:
        raise ValueError(
            f"Expected {expected_count} BCI subjects, got {len(BCI_GROUP_SUBJECTS)}."
        )
    if len(CONTROL_GROUP_SUBJECTS) != expected_count:
        raise ValueError(
            f"Expected {expected_count} control subjects, got {len(CONTROL_GROUP_SUBJECTS)}."
        )

    bci_duplicates = [subj for subj in BCI_GROUP_SUBJECTS if BCI_GROUP_SUBJECTS.count(subj) > 1]
    control_duplicates = [subj for subj in CONTROL_GROUP_SUBJECTS if CONTROL_GROUP_SUBJECTS.count(subj) > 1]
    overlapping = set(BCI_GROUP_SUBJECTS).intersection(CONTROL_GROUP_SUBJECTS)

    if bci_duplicates:
        raise ValueError(
            f"Duplicate entries found in BCI_GROUP_SUBJECTS: {sorted(set(bci_duplicates))}."
        )
    if control_duplicates:
        raise ValueError(
            f"Duplicate entries found in CONTROL_GROUP_SUBJECTS: {sorted(set(control_duplicates))}."
        )
    if overlapping:
        raise ValueError(
            f"Subjects listed in both groups: {sorted(overlapping)}."
        )

    # Check for missing and unexpected subjects
    assigned_subjects = set(BCI_GROUP_SUBJECTS + CONTROL_GROUP_SUBJECTS)
    expected_set = set(EXPECTED_SUBJECTS)
    missing_subjects = sorted(expected_set - assigned_subjects)
    unexpected_subjects = sorted(assigned_subjects - expected_set)

    if missing_subjects:
        raise ValueError(
            f"Missing subjects (not assigned to any group): {missing_subjects}."
        )
    if unexpected_subjects:
        raise ValueError(
            f"Unexpected subjects (not in EXPECTED_SUBJECTS): {unexpected_subjects}."
        )

    print(f"Loaded {expected_count} BCI subjects and {expected_count} control subjects.")
    print(f"Total assigned subjects: {len(assigned_subjects)}.")
    print("Subject group assignment validation passed: "
          "no duplicates, no overlap, all expected subjects assigned.")


_validate_subject_groups()

# Base directory for all project data
PROJECT_ROOT = Path('/Users/hililbby/Library/CloudStorage/Box-Box/CNBI/Attention_distraction/project_healthy')


def _check_subject_directories():
    """Check which expected subject directories exist on disk."""
    if not PROJECT_ROOT.exists():
        print(f"WARNING: Project root directory does not exist: {PROJECT_ROOT}")
        return

    # List existing subject directories
    existing_subjects = set()
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir() and item.name.startswith('e') and item.name[1:].isdigit():
            existing_subjects.add(item.name)

    expected_set = set(EXPECTED_SUBJECTS)
    missing_subjects = sorted(expected_set - existing_subjects)
    unexpected_subjects = sorted(existing_subjects - expected_set)

    if missing_subjects:
        print(f"Missing subject directories on disk: {missing_subjects}")
    else:
        print("All expected subject directories found on disk.")

    if unexpected_subjects:
        print(f"Unexpected subject directories on disk: {unexpected_subjects}")


_check_subject_directories()


def _check_subject_data_files():
    """Validate that expected files and directory structure exist for each subject."""
    if not PROJECT_ROOT.exists():
        print("Cannot validate data files: PROJECT_ROOT does not exist.")
        return

    expected_set = set(EXPECTED_SUBJECTS)
    existing_subjects = set()
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir() and item.name.startswith('e') and item.name[1:].isdigit():
            existing_subjects.add(item.name)

    # Only check subjects that actually exist on disk
    subjects_to_check = sorted(expected_set & existing_subjects)

    if not subjects_to_check:
        print("No subjects found to validate.")
        return

    print(f"\nValidating file structure for {len(subjects_to_check)} subjects...")

    issues = []
    for subject_id in subjects_to_check:
        subject_dir = PROJECT_ROOT / subject_id

        # Check for 5 session folders
        session_folders = sorted([
            d for d in subject_dir.iterdir()
            if d.is_dir() and d.name.startswith(subject_id + '_') and len(d.name.split('_')[1]) == 8
        ])

        if len(session_folders) != 5:
            issues.append(f"  {subject_id}: Expected 5 session folders, found {len(session_folders)}.")
        else:
            # Check for run folders and files in each session
            for session_dir in session_folders:
                run_folders = [d for d in session_dir.iterdir() if d.is_dir()]
                if not run_folders:
                    issues.append(f"  {subject_id} / {session_dir.name}: No run folders found.")
                else:
                    # Check for .gdf or other expected files in run folders
                    for run_dir in run_folders:
                        files = list(run_dir.glob(run_dir.name + '.*'))
                        if not files:
                            issues.append(f"  {subject_id}: {run_dir.name}/ missing expected files (*.gdf, *.txt, etc.).")

    # Check for decoder files
    decoders_dir = PROJECT_ROOT / 'decoders'
    if decoders_dir.exists():
        for subject_id in subjects_to_check:
            decoder_r = decoders_dir / f"{subject_id}_decoderR.mat"
            decoder_l = decoders_dir / f"{subject_id}_decoderL.mat"
            decoder_n = decoders_dir / f"{subject_id}_decoderN.mat"
            if not (decoder_r.exists() and decoder_l.exists() and decoder_n.exists()):
                issues.append(f"  {subject_id}: Missing decoder files (R, L, or N).")

    # Check online_info files
    online_info_dir = PROJECT_ROOT / 'online_info'
    if online_info_dir.exists():
        for subject_id in subjects_to_check:
            matching_files = list(online_info_dir.glob(f"*{subject_id}*"))
            if not matching_files:
                issues.append(f"  {subject_id}: No online_info files found (threshold logs or posteriors).")

    if issues:
        print(f"Found {len(issues)} data structure issues:")
        for issue in issues:
            print(issue)
    else:
        print(f"✓ All {len(subjects_to_check)} subjects have expected directory structure and files.")


_check_subject_data_files()

SUBJECT_GROUP = {
    subject_id: "bci" for subject_id in BCI_GROUP_SUBJECTS
}
SUBJECT_GROUP.update({
    subject_id: "control" for subject_id in CONTROL_GROUP_SUBJECTS
})


def get_subject_group(subject_id):
    """Return the group label for a subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier such as 'e21'.

    Returns
    -------
    str
        'bci' or 'control'. Raises KeyError if the subject is not defined.
    """
    normalized = subject_id.lower().strip()
    if normalized not in SUBJECT_GROUP:
        raise KeyError(
            f"Subject group not defined for {subject_id}. "
            "Check code/src/posthoc_analysis/config.py for the subject list."
        )
    return SUBJECT_GROUP[normalized]
