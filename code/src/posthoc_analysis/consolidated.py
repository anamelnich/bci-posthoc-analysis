"""Consolidated trial-wise data aggregation across all subjects and runs."""

from pathlib import Path
import pandas as pd

from .config import PROJECT_ROOT, BCI_GROUP_SUBJECTS, CONTROL_GROUP_SUBJECTS
from .triggers import load_training_trigger_file, validate_training_triggers
from .analysis import load_training_analysis_file, validate_training_analysis


def parse_run_folder_name(folder_name):
    """Parse subject_datetime_tasktype format into components.
    
    Parameters
    ----------
    folder_name : str
        Run folder name like 'e39_20260303092652_training'
    
    Returns
    -------
    dict
        Keys: subject_id, datetime_str, task_type
    """
    parts = folder_name.split('_')
    if len(parts) < 3:
        return None
    
    subject_id = parts[0]
    datetime_str = parts[1]
    task_type = '_'.join(parts[2:])  # Handle multi-part task names like "training_practice"
    
    return {
        'subject_id': subject_id,
        'datetime_str': datetime_str,
        'task_type': task_type
    }


def parse_session_folder_name(folder_name):
    """Parse subject_date format into components.
    
    Parameters
    ----------
    folder_name : str
        Session folder name like 'e39_20260303'
    
    Returns
    -------
    dict
        Keys: subject_id, session_date
    """
    parts = folder_name.split('_')
    if len(parts) != 2:
        return None
    
    return {
        'subject_id': parts[0],
        'session_date': parts[1]
    }


def get_session_number(session_index):
    """Convert 0-indexed session position to 1-based session number.
    
    Parameters
    ----------
    session_index : int
        Position in sorted session list (0-4)
    
    Returns
    -------
    int
        Session number 1-5
    """
    return session_index + 1


def collect_all_training_runs(root_path=None):
    """Collect all training runs (excluding training_practice) across all subjects and sessions.
    
    Parameters
    ----------
    root_path : str or Path, optional
        Root project path. Defaults to PROJECT_ROOT from config.
    
    Returns
    -------
    tuple
        (runs, issues, subjects_found) where:
        - runs: list of dict, each containing subject_id, session_id, run_id, etc.
        - issues: dict of issues encountered
        - subjects_found: list of subject IDs that have directories in the filesystem
    """
    if root_path is None:
        root_path = PROJECT_ROOT
    
    root_path = Path(root_path)
    
    all_subjects = sorted(set(BCI_GROUP_SUBJECTS + CONTROL_GROUP_SUBJECTS))
    
    # Count actual subject directories found
    subjects_found = []
    for subject_id in all_subjects:
        subject_folder = root_path / subject_id
        if subject_folder.exists():
            subjects_found.append(subject_id)
    
    runs = []
    issues = {
        'missing_trigger_files': [],
        'missing_analysis_files': [],
        'mismatched_file_counts': [],
    }
    
    for subject_id in subjects_found:  # Only process subjects that exist
        subject_folder = root_path / subject_id
        
        # List all session folders for this subject
        session_folders = sorted([
            f for f in subject_folder.iterdir() 
            if f.is_dir() and '_' in f.name
        ])
        
        for session_index, session_folder in enumerate(session_folders):
            session_info = parse_session_folder_name(session_folder.name)
            if session_info is None:
                continue
            
            session_number = get_session_number(session_index)
            session_date = session_info['session_date']
            session_id = f"{subject_id}_{session_date}"
            
            # Find all training-related runs in this session
            run_folders = sorted([
                f for f in session_folder.iterdir()
                if f.is_dir() and ('training' in f.name.lower()) and ('training_practice' not in f.name.lower())
            ])
            
            for run_index, run_folder in enumerate(run_folders, 1):  # Start numbering from 1
                run_info = parse_run_folder_name(run_folder.name)
                if run_info is None:
                    continue
                
                run_id = run_folder.name  # Keep original folder name for file paths
                run_number = run_index   # Sequential number within session (1-8 for session 1, 1-4 for session 5)
                task_type = run_info['task_type']
                
                # Look for trigger and analysis files
                trigger_file = run_folder / f"{run_id}.triggers.txt"
                analysis_file = run_folder / f"{run_id}.analysis.txt"
                
                # Track missing files
                if not trigger_file.exists():
                    issues['missing_trigger_files'].append(str(trigger_file))
                if not analysis_file.exists():
                    issues['missing_analysis_files'].append(str(analysis_file))
                
                # Only add if both files exist
                if trigger_file.exists() and analysis_file.exists():
                    runs.append({
                        'subject_id': subject_id,
                        'session_id': session_id,
                        'session_number': session_number,
                        'run_id': run_id,  # Original folder name
                        'run_number': run_number,  # Sequential number within session
                        'task_type': task_type,
                        'run_folder': run_folder,
                        'trigger_file': trigger_file,
                        'analysis_file': analysis_file,
                    })
    
    return runs, issues, subjects_found


def generate_expected_runs_summary(subjects_found):
    """Generate a summary of expected runs based on experiment structure.
    
    Parameters
    ----------
    subjects_found : list
        List of subject IDs that have directories in filesystem
        
    Returns
    -------
    dict
        Summary with expected counts and detailed run expectations
    """
    expected_runs = []
    
    for subject_id in subjects_found:
        # Each subject should have sessions 1 and 5
        for session_num in [1, 5]:
            if session_num == 1:
                # Session 1: 8 training runs
                num_runs = 8
            else:
                # Session 5: 4 training runs  
                num_runs = 4
                
            for run_num in range(1, num_runs + 1):
                expected_runs.append({
                    'subject_id': subject_id,
                    'session_number': session_num,
                    'run_number': run_num,
                    'expected_key': f"{subject_id}_session{session_num}_run{run_num}"
                })
    
    return {
        'total_expected': len(expected_runs),
        'expected_runs': expected_runs,
        'runs_per_subject': 12,  # 8 + 4
        'subjects_count': len(subjects_found)
    }


def validate_all_training_files(root_path=None):
    """Comprehensive validation of all training trigger and analysis files against documentation.
    
    Parameters
    ----------
    root_path : str or Path, optional
        Root project path. Defaults to PROJECT_ROOT from config.
    
    Returns
    -------
    dict
        Validation results with issues found and summary statistics
    """
    if root_path is None:
        root_path = PROJECT_ROOT
    
    root_path = Path(root_path)
    
    validation_results = {
        'total_files_checked': 0,
        'issues': {
            'analysis_files': [],
            'trigger_files': [],
            'file_count_mismatches': [],
            'structure_violations': []
        },
        'summary': {
            'analysis_files_valid': 0,
            'trigger_files_valid': 0,
            'runs_with_both_files': 0
        }
    }
    
    # Get all subjects found
    all_subjects = sorted(set(BCI_GROUP_SUBJECTS + CONTROL_GROUP_SUBJECTS))
    subjects_found = []
    for subject_id in all_subjects:
        subject_folder = root_path / subject_id
        if subject_folder.exists():
            subjects_found.append(subject_id)
    
    # Check each subject
    for subject_id in subjects_found:
        subject_folder = root_path / subject_id
        
        # Get all session folders
        session_folders = sorted([
            f for f in subject_folder.iterdir() 
            if f.is_dir() and '_' in f.name
        ])
        
        for session_folder in session_folders:
            session_info = parse_session_folder_name(session_folder.name)
            if session_info is None:
                continue
            
            session_number = get_session_number(session_folders.index(session_folder))
            
            # Find all training runs in this session
            training_run_folders = sorted([
                f for f in session_folder.iterdir()
                if f.is_dir() and ('training' in f.name.lower()) and ('training_practice' not in f.name.lower())
            ])
            
            # Validate file counts for this session
            expected_runs = 8 if session_number == 1 else 4
            if len(training_run_folders) != expected_runs:
                validation_results['issues']['file_count_mismatches'].append({
                    'subject_id': subject_id,
                    'session_number': session_number,
                    'expected_runs': expected_runs,
                    'found_runs': len(training_run_folders),
                    'run_folders': [f.name for f in training_run_folders]
                })
            
            # Validate each training run
            for run_folder in training_run_folders:
                run_id = run_folder.name
                trigger_file = run_folder / f"{run_id}.triggers.txt"
                analysis_file = run_folder / f"{run_id}.analysis.txt"
                
                validation_results['total_files_checked'] += 1
                
                # Check if both files exist
                has_trigger = trigger_file.exists()
                has_analysis = analysis_file.exists()
                
                if has_trigger and has_analysis:
                    validation_results['summary']['runs_with_both_files'] += 1
                
                # Validate trigger file
                if has_trigger:
                    trigger_issues = _validate_trigger_file(trigger_file, run_id)
                    if not trigger_issues:
                        validation_results['summary']['trigger_files_valid'] += 1
                    else:
                        validation_results['issues']['trigger_files'].extend(trigger_issues)
                else:
                    validation_results['issues']['trigger_files'].append({
                        'file': str(trigger_file),
                        'issue': 'File does not exist'
                    })
                
                # Validate analysis file
                if has_analysis:
                    analysis_issues = _validate_analysis_file(analysis_file, run_id)
                    if not analysis_issues:
                        validation_results['summary']['analysis_files_valid'] += 1
                    else:
                        validation_results['issues']['analysis_files'].extend(analysis_issues)
                else:
                    validation_results['issues']['analysis_files'].append({
                        'file': str(analysis_file),
                        'issue': 'File does not exist'
                    })
    
    return validation_results


def _validate_trigger_file(trigger_file_path, run_id):
    """Validate a single trigger file against documentation specifications."""
    issues = []
    
    try:
        # Load the file
        df = pd.read_csv(trigger_file_path, sep=r"\s+", header=None, names=['trial', 'trigger_type', 'time'])
        
        # Check row count: should be 180 (3 triggers × 60 trials)
        if len(df) != 180:
            issues.append({
                'file': str(trigger_file_path),
                'issue': f'Expected 180 rows (3 triggers × 60 trials), found {len(df)} rows'
            })
            return issues  # Can't validate further if wrong row count
        
        # Check column count
        if df.shape[1] != 3:
            issues.append({
                'file': str(trigger_file_path),
                'issue': f'Expected 3 columns, found {df.shape[1]} columns'
            })
        
        # Check trial numbers: should be 1-60, each appearing exactly 3 times
        trial_counts = df['trial'].value_counts().sort_index()
        expected_trials = list(range(1, 61))
        actual_trials = sorted(trial_counts.index.tolist())
        
        if actual_trials != expected_trials:
            issues.append({
                'file': str(trigger_file_path),
                'issue': f'Trial numbers should be 1-60, found {actual_trials}'
            })
        
        # Each trial should have exactly 3 triggers
        invalid_trial_counts = trial_counts[trial_counts != 3]
        if len(invalid_trial_counts) > 0:
            issues.append({
                'file': str(trigger_file_path),
                'issue': f'Trials with wrong trigger count: {invalid_trial_counts.to_dict()}'
            })
        
        # Validate trigger sequences for each trial
        for trial_num in range(1, 61):
            trial_data = df[df['trial'] == trial_num].sort_values('time')
            if len(trial_data) == 3:
                triggers = trial_data['trigger_type'].tolist()
                # Expected sequence: [4, (8|32|44), 64]
                if not (triggers[0] == 4 and triggers[2] == 64 and triggers[1] in [8, 32, 44]):
                    issues.append({
                        'file': str(trigger_file_path),
                        'issue': f'Trial {trial_num}: Invalid trigger sequence {triggers}, expected [4, (8|32|44), 64]'
                    })
        
        # Check timing values are reasonable (positive integers)
        if not df['time'].dtype.kind in 'iu':  # integer or unsigned int
            issues.append({
                'file': str(trigger_file_path),
                'issue': 'Time column should contain integers (samples)'
            })
        
        if (df['time'] <= 0).any():
            issues.append({
                'file': str(trigger_file_path),
                'issue': 'Time values should be positive (samples from start of recording)'
            })
        
    except Exception as e:
        issues.append({
            'file': str(trigger_file_path),
            'issue': f'Error reading/parsing file: {str(e)}'
        })
    
    return issues


def _validate_analysis_file(analysis_file_path, run_id):
    """Validate a single analysis file against documentation specifications."""
    issues = []
    
    try:
        # Load the file (no headers, whitespace-separated)
        df = pd.read_csv(analysis_file_path, sep=r"\s+", header=None, 
                        names=['trial_index', 'task', 'feedback', 'target_position', 
                               'distractor_position', 'dot_side', 'intertrial_interval_ms', 'bci_output'])
        
        # Check row count: should be exactly 60 trials
        if len(df) != 60:
            issues.append({
                'file': str(analysis_file_path),
                'issue': f'Expected 60 rows (60 trials), found {len(df)} rows'
            })
            return issues  # Can't validate further if wrong row count
        
        # Check column count
        if df.shape[1] != 8:
            issues.append({
                'file': str(analysis_file_path),
                'issue': f'Expected 8 columns, found {df.shape[1]} columns'
            })
        
        # Validate trial_index: should be 1-60
        expected_trials = list(range(1, 61))
        actual_trials = sorted(df['trial_index'].tolist())
        if actual_trials != expected_trials:
            issues.append({
                'file': str(analysis_file_path),
                'issue': f'Trial indices should be 1-60, found {actual_trials}'
            })
        
        # Validate task column: should be 0 or 1
        invalid_tasks = df[~df['task'].isin([0, 1])]
        if len(invalid_tasks) > 0:
            issues.append({
                'file': str(analysis_file_path),
                'issue': f'Task column should be 0 (no distractor) or 1 (distractor), found invalid values: {sorted(invalid_tasks["task"].unique())}'
            })
        
        # Validate feedback column: should be 1, 2, or 3
        invalid_feedback = df[~df['feedback'].isin([1, 2, 3])]
        if len(invalid_feedback) > 0:
            issues.append({
                'file': str(analysis_file_path),
                'issue': f'Feedback column should be 1 (correct), 2 (incorrect), or 3 (timeout), found invalid values: {sorted(invalid_feedback["feedback"].unique())}'
            })
        
        # Validate target_position: should be 1, 2, 3, 4
        invalid_targets = df[~df['target_position'].isin([1, 2, 3, 4])]
        if len(invalid_targets) > 0:
            issues.append({
                'file': str(analysis_file_path),
                'issue': f'Target position should be 1, 2, 3, or 4, found invalid values: {sorted(invalid_targets["target_position"].unique())}'
            })
        
        # Validate distractor_position: should be 0, 2, 4
        invalid_distractors = df[~df['distractor_position'].isin([0, 2, 4])]
        if len(invalid_distractors) > 0:
            issues.append({
                'file': str(analysis_file_path),
                'issue': f'Distractor position should be 0, 2, or 4, found invalid values: {sorted(invalid_distractors["distractor_position"].unique())}'
            })
        
        # Validate dot_side: should be 0 or 1
        invalid_sides = df[~df['dot_side'].isin([0, 1])]
        if len(invalid_sides) > 0:
            issues.append({
                'file': str(analysis_file_path),
                'issue': f'Dot side should be 0 (left) or 1 (right), found invalid values: {sorted(invalid_sides["dot_side"].unique())}'
            })
        
        # Validate intertrial_interval_ms: should be positive
        if (df['intertrial_interval_ms'] <= 0).any():
            issues.append({
                'file': str(analysis_file_path),
                'issue': 'Intertrial interval should be positive milliseconds'
            })
        
        # Validate bci_output: for training, should be 99
        invalid_bci = df[df['bci_output'] != 99]
        if len(invalid_bci) > 0:
            issues.append({
                'file': str(analysis_file_path),
                'issue': f'BCI output should be 99 for training runs, found other values: {sorted(invalid_bci["bci_output"].unique())}'
            })
        
    except Exception as e:
        issues.append({
            'file': str(analysis_file_path),
            'issue': f'Error reading/parsing file: {str(e)}'
        })
    
    return issues


def compare_found_vs_expected(found_runs, expected_summary):
    """Compare found runs against expected runs to identify missing ones.
    
    Parameters
    ----------
    found_runs : list
        List of run dicts from collect_all_training_runs
    expected_summary : dict
        Output from generate_expected_runs_summary
        
    Returns
    -------
    dict
        Comparison results with missing runs, found runs, etc.
    """
    # Create lookup of found runs by subject/session/run pattern
    found_lookup = {}
    for run in found_runs:
        # Extract run number from run_id (e.g., 'e39_20260303092652_training' -> look for pattern)
        # This is approximate - we'll match by subject/session and count
        key = f"{run['subject_id']}_session{run['session_number']}"
        if key not in found_lookup:
            found_lookup[key] = []
        found_lookup[key].append(run)
    
    missing_runs = []
    found_by_subject_session = {}
    
    # Check each expected run
    for expected in expected_summary['expected_runs']:
        subject_id = expected['subject_id']
        session_num = expected['session_number']
        run_num = expected['run_number']
        
        key = f"{subject_id}_session{session_num}"
        
        # Count how many runs we found for this subject/session
        found_count = len(found_lookup.get(key, []))
        
        # For session 1, expect 8 runs; session 5, expect 4 runs
        expected_count = 8 if session_num == 1 else 4
        
        if found_count < expected_count:
            # This subject/session is missing runs
            missing_runs.append({
                'subject_id': subject_id,
                'session_number': session_num,
                'expected_runs': expected_count,
                'found_runs': found_count,
                'missing_count': expected_count - found_count
            })
        
        # Track found runs by subject/session
        if key not in found_by_subject_session:
            found_by_subject_session[key] = found_count
    
    return {
        'missing_runs': missing_runs,
        'found_by_subject_session': found_by_subject_session,
        'total_found': len(found_runs),
        'total_expected': expected_summary['total_expected']
    }


def load_and_merge_training_run(run_info):
    """Load trigger and analysis files for a single run and merge them.
    
    Parameters
    ----------
    run_info : dict
        Dict with keys: subject_id, session_id, session_number, run_id,
        task_type, trigger_file, analysis_file
    
    Returns
    -------
    pandas.DataFrame
        Merged trial-wise data with all columns from both files, plus
        subject_id, session_id, run_id (sequential within session), task_type
    
    Raises
    ------
    ValueError
        If trial counts don't match or files are malformed
    """
    trigger_file = run_info['trigger_file']
    analysis_file = run_info['analysis_file']
    
    # Load files
    try:
        triggers_df = load_training_trigger_file(trigger_file)
    except Exception as e:
        raise ValueError(f"Failed to load trigger file {trigger_file}: {e}")
    
    try:
        analysis_df = load_training_analysis_file(analysis_file)
    except Exception as e:
        raise ValueError(f"Failed to load analysis file {analysis_file}: {e}")
    
    # Check trial counts match
    if len(triggers_df) // 3 != len(analysis_df):
        raise ValueError(
            f"Trial count mismatch for {run_info['run_id']}: "
            f"triggers has {len(triggers_df) // 3} trials, "
            f"analysis has {len(analysis_df)} trials"
        )
    
    # Compute RT from triggers
    rt_rows = []
    for trial, group in triggers_df.groupby("trial", sort=True):
        stimulus = group.iloc[1]
        response = group.iloc[2]
        rt_samples = int(response["time"]) - int(stimulus["time"])
        rt_ms = (rt_samples / 512.0) * 1000.0
        
        rt_rows.append({
            'trial': int(trial),
            'rt_samples': rt_samples,
            'rt_ms': rt_ms,
        })
    
    rt_df = pd.DataFrame(rt_rows)
    
    # Merge: analysis + RT columns
    # Rename trial_index to trial for consistency
    merged = analysis_df.copy()
    merged = merged.rename(columns={'trial_index': 'trial'})
    merged = merged.merge(rt_df, on='trial', how='left')
    
    # Add run-level metadata
    merged['subject_id'] = run_info['subject_id']
    merged['session_id'] = run_info['session_number']  # Session number as integer (1 or 5)
    merged['run_id'] = run_info['run_number']  # Use sequential run number (1-8 for session 1, 1-4 for session 5)
    merged['task_type'] = run_info['task_type']
    
    return merged


def generate_consolidated_training_csv(output_path=None, root_path=None):
    """Generate consolidated CSV with all training (non-practice) trial-wise data.
    
    The output DataFrame includes:
    - 'group' column: 'experimental' or 'control' group membership
    - 'session_id' column: session number as integer (1 or 5)
    - 'run_id' column: sequential run number within session (1-8 for session 1, 1-4 for session 5)
    
    Parameters
    ----------
    output_path : str or Path, optional
        Path to save CSV. Defaults to 
        `/Users/.../project_healthy/analyses/all_subjects_training.csv`
    root_path : str or Path, optional
        Root project path. Defaults to PROJECT_ROOT from config.
    
    Returns
    -------
    dict
        Summary with keys:
        - 'dataframe': consolidated pandas DataFrame
        - 'total_runs': number of runs loaded
        - 'total_trials': number of trials
        - 'issues': dict of issues encountered
        - 'subjects_present': list of subjects in final data
    """
    if output_path is None:
        output_path = Path(
            '/Users/hililbby/Library/CloudStorage/Box-Box/CNBI/Attention_distraction'
            '/project_healthy/analyses/all_subjects_training.csv'
        )
    else:
        output_path = Path(output_path)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all runs
    all_runs, collection_issues, subjects_found = collect_all_training_runs(root_path)
    
    # Generate expected runs summary
    expected_summary = generate_expected_runs_summary(subjects_found)
    
    # Compare found vs expected
    comparison = compare_found_vs_expected(all_runs, expected_summary)
    
    # Print detailed analysis
    print(f"Expected training runs based on experiment structure: {expected_summary['total_expected']} (12 per subject × {len(subjects_found)} subjects found)")
    print(f"Found complete training run(s) with both trigger and analysis files: {len(all_runs)}")
    
    if len(all_runs) != expected_summary['total_expected']:
        print(f"WARNING: Found {len(all_runs)} runs but expected {expected_summary['total_expected']} based on experiment structure.")
        print(f"This suggests missing data for {expected_summary['total_expected'] - len(all_runs)} training runs.")
        
        # Print detailed missing runs analysis
        print("\n" + "=" * 80)
        print("DETAILED MISSING RUNS ANALYSIS")
        print("=" * 80)
        
        # Group missing runs by subject/session to avoid duplicates
        missing_by_key = {}
        for missing in comparison['missing_runs']:
            key = f"{missing['subject_id']}_session{missing['session_number']}"
            if key not in missing_by_key:
                missing_by_key[key] = missing
        
        if missing_by_key:
            print(f"\nFound {len(missing_by_key)} subject/session combinations with missing runs:")
            for key, missing in missing_by_key.items():
                print(f"  - Subject {missing['subject_id']}, Session {missing['session_number']}: "
                      f"Expected {missing['expected_runs']} runs, found {missing['found_runs']} "
                      f"(missing {missing['missing_count']} runs)")
                
                # For the missing subject/session, list what runs were found vs expected
                if missing['missing_count'] > 0:
                    found_runs_for_session = []
                    for run in all_runs:
                        if (run['subject_id'] == missing['subject_id'] and 
                            run['session_number'] == missing['session_number']):
                            # Extract run number from run_id if possible
                            run_id = run['run_id']
                            found_runs_for_session.append(run_id)
                    
                    print(f"    Found runs: {sorted(found_runs_for_session)}")
                    print(f"    Expected: {missing['expected_runs']} training runs")
        else:
            print("\nNo specific subject/session combinations identified as missing runs.")
            print("This suggests the issue may be with file parsing or unexpected directory structure.")
        
        # Print summary by subject/session
        print(f"\nRuns found by subject/session:")
        for key, count in sorted(comparison['found_by_subject_session'].items()):
            subject_id, session_part = key.split('_session')
            session_num = session_part
            expected_for_session = 8 if session_num == '1' else 4
            status = "✓" if count == expected_for_session else "✗"
            print(f"  {status} {key}: {count}/{expected_for_session} runs")
        
        print("\n" + "=" * 80)
    else:
        print("✓ Found expected number of training runs.")
    print()
    
    # Load and merge each run
    all_data = []
    merge_issues = {
        'trial_count_mismatch': [],
        'load_errors': [],
    }
    
    for i, run_info in enumerate(all_runs, 1):
        try:
            merged = load_and_merge_training_run(run_info)
            all_data.append(merged)
            # Only print if there was an issue, not for successful loads
        except Exception as e:
            error_msg = f"{run_info['run_id']}: {str(e)}"
            merge_issues['load_errors'].append(error_msg)
            print(f"ERROR loading {run_info['run_id']}: {e}")
    
    # Concatenate all data
    if all_data:
        consolidated = pd.concat(all_data, ignore_index=True)
        
        # Add group column
        consolidated['group'] = consolidated['subject_id'].apply(
            lambda x: 'experimental' if x in BCI_GROUP_SUBJECTS else 'control'
        )
    else:
        consolidated = pd.DataFrame()
    
    # Only print summary if there were issues
    has_issues = (
        collection_issues['missing_trigger_files'] or
        collection_issues['missing_analysis_files'] or
        merge_issues['load_errors']
    )
    
    if has_issues:
        print()
        print("=" * 70)
        print("ISSUES FOUND DURING CONSOLIDATION")
        print("=" * 70)
        print(f"Total runs successfully loaded: {len(all_data)}")
        print(f"Total trials: {len(consolidated)}")
        print(f"Unique subjects: {consolidated['subject_id'].nunique() if len(consolidated) > 0 else 0}")
        print(f"Unique sessions: {consolidated['session_id'].nunique() if len(consolidated) > 0 else 0}")
        print()
        
        # Print issues
        if collection_issues['missing_trigger_files']:
            print(f"Missing trigger files: {len(collection_issues['missing_trigger_files'])}")
            for f in collection_issues['missing_trigger_files'][:5]:
                print(f"  - {f}")
            if len(collection_issues['missing_trigger_files']) > 5:
                print(f"  ... and {len(collection_issues['missing_trigger_files']) - 5} more")
            print()
        
        if collection_issues['missing_analysis_files']:
            print(f"Missing analysis files: {len(collection_issues['missing_analysis_files'])}")
            for f in collection_issues['missing_analysis_files'][:5]:
                print(f"  - {f}")
            if len(collection_issues['missing_analysis_files']) > 5:
                print(f"  ... and {len(collection_issues['missing_analysis_files']) - 5} more")
            print()
        
        if merge_issues['load_errors']:
            print(f"Load/merge errors: {len(merge_issues['load_errors'])}")
            for err in merge_issues['load_errors'][:5]:
                print(f"  - {err}")
            if len(merge_issues['load_errors']) > 5:
                print(f"  ... and {len(merge_issues['load_errors']) - 5} more")
            print()
        
        print("=" * 70)
        print()
    
    # Save to CSV
    if len(consolidated) > 0:
        # Reorder columns for readability
        id_cols = ['subject_id', 'group', 'session_id', 'run_id', 'task_type', 'trial']
        other_cols = [c for c in consolidated.columns if c not in id_cols]
        consolidated = consolidated[id_cols + other_cols]
        
        consolidated.to_csv(output_path, index=False)
        print(f"Saved consolidated data to: {output_path}")
    else:
        print("No data to save (all runs failed to load).")
    
    print("=" * 70)
    print()
    
    return {
        'dataframe': consolidated,
        'output_path': str(output_path),
        'total_runs': len(all_data),
        'total_trials': len(consolidated),
        'subjects_present': sorted(consolidated['subject_id'].unique().tolist()) if len(consolidated) > 0 else [],
        'collection_issues': collection_issues,
        'merge_issues': merge_issues,
    }


def validate_all_files_comprehensive():
    """Run comprehensive validation of all training files and report results.
    
    Returns
    -------
    bool
        True if all files are valid, False if any issues found
    """
    print("Running comprehensive validation of all training trigger and analysis files...")
    print("=" * 80)
    
    results = validate_all_training_files()
    
    # Print summary
    print(f"Files checked: {results['total_files_checked']}")
    print(f"Runs with both trigger and analysis files: {results['summary']['runs_with_both_files']}")
    print(f"Valid trigger files: {results['summary']['trigger_files_valid']}")
    print(f"Valid analysis files: {results['summary']['analysis_files_valid']}")
    print()
    
    # Check for issues
    total_issues = sum(len(issues) for issues in results['issues'].values())
    
    if total_issues == 0:
        print("✓ All files look good! No issues found.")
        print("All trigger and analysis files conform to expected structure and content.")
        return True
    else:
        print(f"✗ Found {total_issues} issues across all files:")
        print()
        
        # Report issues by category
        if results['issues']['file_count_mismatches']:
            print(f"File count mismatches: {len(results['issues']['file_count_mismatches'])}")
            for issue in results['issues']['file_count_mismatches']:
                print(f"  - Subject {issue['subject_id']}, Session {issue['session_number']}: "
                      f"Expected {issue['expected_runs']} runs, found {issue['found_runs']}")
            print()
        
        if results['issues']['trigger_files']:
            print(f"Trigger file issues: {len(results['issues']['trigger_files'])}")
            # Group by type of issue for cleaner output
            issue_counts = {}
            for issue in results['issues']['trigger_files']:
                issue_type = issue.get('issue', 'Unknown issue')
                if issue_type not in issue_counts:
                    issue_counts[issue_type] = 0
                issue_counts[issue_type] += 1
            
            for issue_type, count in issue_counts.items():
                print(f"  - {issue_type}: {count} files")
            print()
        
        if results['issues']['analysis_files']:
            print(f"Analysis file issues: {len(results['issues']['analysis_files'])}")
            # Group by type of issue for cleaner output
            issue_counts = {}
            for issue in results['issues']['analysis_files']:
                issue_type = issue.get('issue', 'Unknown issue')
                if issue_type not in issue_counts:
                    issue_counts[issue_type] = 0
                issue_counts[issue_type] += 1
            
            for issue_type, count in issue_counts.items():
                print(f"  - {issue_type}: {count} files")
            print()
        
        print("=" * 80)
        return False
