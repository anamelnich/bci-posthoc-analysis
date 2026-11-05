"""Training Pd after behavioral exclusions, with equal retained-trial weighting."""

from pathlib import Path
import contextlib
import hashlib
import io
import json

import numpy as np
import pandas as pd

from . import eeg
from .analysis import TRAINING_ANALYSIS_COLUMNS, validate_training_analysis
from .behavioral import _prepare_reaction_time_trial_data, _remove_rt_outliers
from .config import PROJECT_ROOT, get_subject_group
from .consolidated import collect_all_training_runs, load_and_merge_training_run
from .triggers import apply_training_text_rt_fallback

TRIAL_KEYS = ["subject_id", "session_id", "run_id", "trial"]
REASONS = ["incorrect", "timeout", "missing_rt", "rt_below_150", "rt_outlier_3sd", "retained"]


def prepare_pd_trial_exclusions(training_trials):
    """Validate corrected training trials and reuse the exact training-RT filter.

    Bounds use correct, finite RT >=150 ms, pooling every run and condition
    within participant/session. Exclusion reasons are sequential and exclusive.
    """
    required = set(TRIAL_KEYS + TRAINING_ANALYSIS_COLUMNS[1:] + [
        "group", "task_type", "stimulus_trigger", "stimulus_sample", "condition",
        "rt_ms", "gdf_rt_ms", "rt_source", "response_sample", "rt_samples",
        "response_count", "alignment_error", "text_rt_ms", "sfreq_hz",
    ])
    if not isinstance(training_trials, pd.DataFrame) or training_trials.empty:
        raise ValueError("Pd exclusions require a nonempty corrected training DataFrame.")
    if required - set(training_trials):
        raise ValueError(f"Pd exclusions missing columns: {sorted(required - set(training_trials))}")
    data = training_trials.copy().sort_values(TRIAL_KEYS).reset_index(drop=True)
    if data[TRIAL_KEYS].isna().any().any() or data.duplicated(TRIAL_KEYS).any():
        raise ValueError("Missing or duplicate participant/session/run/trial keys.")
    if not data.task_type.eq("training").all() or not data.session_id.isin([1, 5]).all():
        raise ValueError("Supply only non-practice training trials in Sessions 1 and 5.")
    for column in ["session_id", "run_id", "trial", "stimulus_sample", "stimulus_trigger"]:
        if not np.isfinite(data[column]).all() or not data[column].eq(data[column].round()).all():
            raise ValueError(f"{column} must contain finite integers.")
    if data.run_id.lt(1).any() or data.stimulus_sample.lt(0).any():
        raise ValueError("Run IDs must be positive and stimulus samples nonnegative.")
    expected_groups = data.subject_id.map(get_subject_group)
    if not data.group.replace({"experimental": "bci"}).equals(expected_groups):
        raise ValueError("Training group labels disagree with configured participant groups.")
    data = apply_training_text_rt_fallback(data)
    if data.alignment_error.any():
        raise ValueError("Resolve ambiguous GDF trial alignment before computing Pd.")
    for key, run in data.groupby(TRIAL_KEYS[:3], sort=True):
        analysis = run.rename(columns={"trial": "trial_index"})[TRAINING_ANALYSIS_COLUMNS]
        validate_training_analysis(analysis, filepath=str(key))
        if run.stimulus_trigger.value_counts().to_dict() != {8: 30, 32: 15, 44: 15}:
            raise ValueError(f"{key}: expected 30/15/15 stimulus conditions.")
        if (not run.condition.equals(run.stimulus_trigger.map(eeg.STIMULUS_CODES))
                or not run.task.eq(run.stimulus_trigger.ne(8).astype(int)).all()
                or not run.distractor_position.eq(run.stimulus_trigger.map({8: 0, 32: 2, 44: 4})).all()
                or not np.all(np.diff(run.stimulus_sample) > 0)):
            raise ValueError(f"{key}: stimulus order, task, condition, or side mismatch.")
        if not np.allclose(run.response_sample - run.stimulus_sample, run.rt_samples,
                           equal_nan=True, rtol=0, atol=1e-8):
            raise ValueError(f"{key}: response/stimulus samples disagree with RT samples.")
    # Avoid a pre-existing descriptive trial_type column overriding documented task codes.
    eligible, _, _ = _prepare_reaction_time_trial_data(data.drop(columns=["trial_type"], errors="ignore"))
    cleaned, bounds = _remove_rt_outliers(eligible)
    retained_keys = pd.MultiIndex.from_frame(cleaned[TRIAL_KEYS])
    data["exclusion_reason"] = np.select(
        [data.feedback.eq(2), data.feedback.eq(3), ~np.isfinite(data.rt_ms), data.rt_ms.lt(150)],
        ["incorrect", "timeout", "missing_rt", "rt_below_150"], default="retained",
    )
    is_retained = pd.MultiIndex.from_frame(data[TRIAL_KEYS]).isin(retained_keys)
    data.loc[data.exclusion_reason.eq("retained") & ~is_retained, "exclusion_reason"] = "rt_outlier_3sd"
    data["retained_for_pd"] = is_retained & data.stimulus_trigger.isin([32, 44])
    if not np.array_equal(data.exclusion_reason.eq("retained"), is_retained):
        raise ValueError("Exclusion audit does not reconcile with existing training-RT filter.")
    summary = (data.groupby(["subject_id", "group", "session_id", "exclusion_reason"])
               .size().unstack(fill_value=0).reindex(columns=REASONS, fill_value=0).reset_index())
    summary["total_trials"] = summary[REASONS].sum(axis=1)
    summary["excluded_trials"] = summary.total_trials - summary.retained
    summary["percent_excluded"] = 100 * summary.excluded_trials / summary.total_trials
    print("Pd behavioral exclusions (all training conditions; mutually exclusive):")
    print(summary[REASONS + ["total_trials"]].sum().to_string())
    print(f"Retained distractor trials for Pd: {int(data.retained_for_pd.sum())}; "
          "RT bounds = one-pass participant/session mean ±3 sample SD, pooled runs/conditions.")
    return {"trials": data, "exclusion_summary": summary, "rt_bounds": bounds}


def _validate_filtered_pd_waveforms(waves, trials):
    """Check complete time grids, participant coverage, and retained denominators."""
    required = {"subject_id", "group", "session_id", "sample_index", "time_sec",
                "pd_amplitude_uv", "n_retained_distractor_trials", "n_runs", "expected_n_runs"}
    if waves.empty or required - set(waves) or waves.duplicated(["subject_id", "session_id", "sample_index"]).any():
        raise ValueError("Invalid filtered Pd waveform schema, empty data, or duplicate samples.")
    expected = set(map(tuple, trials[["subject_id", "session_id"]].drop_duplicates().to_numpy()))
    observed = set(map(tuple, waves[["subject_id", "session_id"]].drop_duplicates().to_numpy()))
    if expected != observed:
        raise ValueError(f"Filtered Pd participant/session coverage mismatch: {expected ^ observed}")
    for (subject, session), cell in waves.groupby(["subject_id", "session_id"]):
        cell = cell.sort_values("sample_index")
        source = trials[trials.subject_id.eq(subject) & trials.session_id.eq(session)]
        count = int(source.retained_for_pd.sum())
        if (not np.array_equal(cell.sample_index, np.arange(768))
                or not np.allclose(cell.time_sec, np.arange(768) / 512 - .5, rtol=0, atol=1e-12)
                or not np.isfinite(cell.pd_amplitude_uv).all()
                or not cell.group.eq(get_subject_group(subject)).all()
                or count == 0 or not cell.n_retained_distractor_trials.eq(count).all()
                or not cell.n_runs.eq(source.run_id.nunique()).all()
                or not cell.expected_n_runs.eq(eeg.EXPECTED_TRAINING_RUNS_BY_SESSION[session]).all()):
            raise ValueError(f"{subject} session {session}: invalid time grid, waveform, group, or counts.")
        if abs(cell.loc[cell.time_sec.ge(-.2) & cell.time_sec.lt(0), "pd_amplitude_uv"].mean()) > 1e-8:
            raise ValueError(f"{subject} session {session}: baseline residual exceeds tolerance.")
    print(f"Validated {len(expected)} filtered Pd waveforms: 768 samples at 512 Hz, "
          "-0.5 to <1 s; -0.2 to <0 s baseline; exact retained-trial denominators.")


def rebuild_behavior_filtered_pd(training_trials, project_root=PROJECT_ROOT, cache_dir=None, force=False):
    """Rebuild pooled-trial Pd from validated GDF epochs, or reuse a matching cache.

    Reuses canonical 0.1–20 Hz, baseline-corrected PO7/PO8 processing and native
    amplitude scaling of the existing Pd pipeline. Each retained distractor
    trial has equal weight; participants have equal weight downstream.
    Cache provenance binds behavioral input, raw-file size/mtime, and source code.
    """
    result = prepare_pd_trial_exclusions(training_trials)
    trials = result["trials"]
    project_root = Path(project_root)
    runs, issues, _ = collect_all_training_runs(project_root)
    lookup = {(r["subject_id"], r["session_number"], r["run_number"]): r for r in runs}
    source_files = []
    for (subject, session), cell in trials.groupby(["subject_id", "session_id"]):
        gdfs = eeg._get_training_run_gdf_files_for_session(subject, int(session), project_root, allow_incomplete=True)
        if sorted(cell.run_id.unique()) != list(range(1, len(gdfs) + 1)):
            raise ValueError(f"{subject} session {session}: trial table/raw run coverage mismatch; {issues}")
        for run_id, gdf in enumerate(gdfs, 1):
            key = (subject, session, run_id)
            if key not in lookup or Path(lookup[key]["trigger_file"]) != gdf.with_suffix(".triggers.txt"):
                raise ValueError(f"{key}: inconsistent raw run mapping.")
            for path in [gdf, gdf.with_suffix(".triggers.txt"), gdf.with_suffix(".analysis.txt")]:
                stat = path.stat()
                source_files.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    code_dir = Path(__file__).parent
    provenance = {
        "method": "correct; RT>=150; pooled participant/session ±3 sample SD; equal retained-trial Pd",
        "trial_sha256": hashlib.sha256(trials.to_csv(index=False).encode()).hexdigest(),
        "source_files": source_files,
        "code_sha256": {name: hashlib.sha256((code_dir / name).read_bytes()).hexdigest()
                        for name in ["pd_filtered.py", "eeg.py", "behavioral.py", "triggers.py", "analysis.py", "consolidated.py", "config.py"]},
    }
    # The unfiltered condition averages cannot reconstruct excluded trials.
    candidates = [Path(cache_dir)] if cache_dir is not None else [
        project_root / "analyses" / "eeg_pd" / "behavior_filtered",
        eeg.REPO_ROOT / "analyses" / "eeg_pd" / "behavior_filtered",
    ]
    for directory in candidates:
        manifest = directory / "provenance.json"
        wave_path = directory / "subject_session_pd.csv"
        run_path = directory / "run_trial_counts.csv"
        if not force and manifest.is_file() and wave_path.is_file() and run_path.is_file():
            saved = json.loads(manifest.read_text())
            if saved.get("inputs") == provenance:
                for path in [wave_path, run_path]:
                    if hashlib.sha256(path.read_bytes()).hexdigest() != saved["output_sha256"][path.name]:
                        raise ValueError(f"Filtered Pd cache checksum mismatch: {path}")
                waves = pd.read_csv(wave_path)
                _validate_filtered_pd_waveforms(waves, trials)
                print(f"Using validated behavior-filtered Pd cache: {wave_path}")
                return {**result, "subject_session_pd": waves,
                        "run_trial_counts": pd.read_csv(run_path), "cache_dir": directory}
    print("Rebuilding raw EEG: no matching behavior-filtered Pd cache at "
          f"{[str(p) for p in candidates]}. Unfiltered condition averages cannot apply trial exclusions.")
    wave_frames, run_rows = [], []
    for (subject, session), cell in trials.groupby(["subject_id", "session_id"], sort=True):
        pooled_sum = np.zeros(768)
        n_retained = 0
        for run_id, run in cell.groupby("run_id", sort=True):
            run = run.sort_values("trial")
            info = lookup[(subject, session, run_id)]
            gdf = Path(info["trigger_file"]).with_suffix("").with_suffix(".gdf")
            # Suppress routine per-run output, but preserve it on an integrity failure.
            log = io.StringIO()
            try:
                with contextlib.redirect_stdout(log):
                    raw_trials = load_and_merge_training_run(info).sort_values("trial")
                    for column in TRAINING_ANALYSIS_COLUMNS[1:] + [
                        "trial", "stimulus_sample", "stimulus_trigger", "response_sample",
                        "rt_samples", "gdf_rt_ms", "rt_ms", "response_count", "text_rt_ms", "sfreq_hz",
                    ]:
                        if not np.allclose(run[column], raw_trials[column], equal_nan=True, rtol=0, atol=1e-8):
                            raise ValueError(f"{gdf}: consolidated {column} disagrees with raw reconstruction.")
                    epochs = eeg.load_filter_epoch_baseline_correct_training_run(gdf)
                    if (not np.array_equal(epochs["kept_event_samples"], run.stimulus_sample)
                            or not np.array_equal(epochs["stimulus_events"][:, 2], run.stimulus_trigger)
                            or tuple(epochs["baseline_indices"]) != (154, 256)
                            or not np.allclose(epochs["time"], np.arange(768) / 512 - .5, atol=1e-12, rtol=0)):
                        raise ValueError(f"{gdf}: EEG epochs do not align with trial table/time/baseline.")
                    pd_run = eeg.compute_pd_difference_wave(
                        epochs["baseline_corrected_epochs"], epochs["eeg_labels"], epochs["stimulus_events"])
                indices = pd_run["pd_trial_indices"]
                keep = run.retained_for_pd.to_numpy()[indices]
                retained = pd_run["pd_trials_samples_by_trials"][:, keep]
                pooled_sum += retained.sum(axis=1)
                n_retained += retained.shape[1]
                run_rows.append({"subject_id": subject, "group": get_subject_group(subject),
                                 "session_id": session, "run_id": run_id, "gdf_path": str(gdf),
                                 "n_original_distractor": len(indices), "n_retained_distractor": int(keep.sum()),
                                 "n_retained_right": int((run.retained_for_pd & run.stimulus_trigger.eq(32)).sum()),
                                 "n_retained_left": int((run.retained_for_pd & run.stimulus_trigger.eq(44)).sum())})
            except Exception:
                print(log.getvalue())
                raise
        if n_retained == 0:
            raise ValueError(f"{subject} session {session}: no retained distractor trials; cannot calculate Pd.")
        wave_frames.append(pd.DataFrame({
            "subject_id": subject, "group": get_subject_group(subject), "session_id": session,
            "sample_index": np.arange(768), "time_sec": epochs["time"],
            "pd_amplitude_uv": pooled_sum / n_retained,
            "n_retained_distractor_trials": n_retained, "n_runs": cell.run_id.nunique(),
            "expected_n_runs": eeg.EXPECTED_TRAINING_RUNS_BY_SESSION[session],
        }))
        print(f"{subject} session {session}: pooled {n_retained}/{int(cell.task.sum())} distractor trials "
              f"across {cell.run_id.nunique()} runs.", flush=True)
    waves, run_counts = pd.concat(wave_frames, ignore_index=True), pd.DataFrame(run_rows)
    _validate_filtered_pd_waveforms(waves, trials)
    directory = candidates[0]
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {"subject_session_pd.csv": waves, "run_trial_counts.csv": run_counts,
               "trial_exclusion_audit.csv": trials, "exclusion_summary.csv": result["exclusion_summary"],
               "rt_bounds.csv": result["rt_bounds"]}
    checksums = {}
    for name, table in outputs.items():
        path = directory / name
        table.to_csv(path, index=False)
        checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "provenance.json").write_text(json.dumps({"inputs": provenance, "output_sha256": checksums}, indent=2))
    print(f"Saved expensive filtered waveforms and exclusion/alignment QC: {directory}")
    return {**result, "subject_session_pd": waves, "run_trial_counts": run_counts, "cache_dir": directory}


def analyze_and_plot_behavior_filtered_pd(filtered_results, output_dir=eeg.FIGURES_DIR):
    """Run existing positive-AUC tests and orange/blue poster plots on identical data."""
    waves = filtered_results["subject_session_pd"]
    _validate_filtered_pd_waveforms(waves, filtered_results["trials"])
    auc = eeg.compute_pd_positive_auc(waves)
    cell_summary = eeg.summarize_pd_positive_auc(auc)
    anova = eeg.run_pd_positive_auc_mixed_anova(auc)
    contrasts = eeg.run_pd_positive_auc_planned_contrasts(auc)
    summary = eeg.summarize_group_pd_pre_post(waves)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    figures, paths = {}, []
    for group in ["bci", "control"]:
        fig, _ = eeg.plot_group_pd_pre_post(summary, group=group)
        for extension in ["pdf", "png"]:
            path = directory / f"training_pd_{group}_pre_post.{extension}"
            fig.savefig(path, dpi=300)
            paths.append(path)
        figures[group] = fig
    print("Saved orange/blue Pd poster plots (mean ± participant SEM); "
          "same pooled-trial waveforms used for 0.2–0.5 s positive AUC. "
          "Planned-contrast P values and 95% CIs are unadjusted, matching existing tests.")
    return {"pd_positive_auc": auc, "cell_summary": cell_summary,
            "anova_table": anova["anova_table"], "anova_diagnostics": anova["diagnostics"],
            "planned_contrasts": contrasts, "group_pd_summary": summary,
            "figures": figures, "saved_paths": paths}
