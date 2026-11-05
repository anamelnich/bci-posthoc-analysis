# Data Locations and Precomputed Outputs

## When to use this file

Read this file before locating, loading, regenerating, or overwriting an
analysis output. In particular, use it before starting a raw-data computation
because a validated precomputed result may already exist.

## Canonical locations

### Study data and precomputed analyses

The canonical study root is configured as `PROJECT_ROOT`:

`/Users/hililbby/Library/CloudStorage/Box-Box/CNBI/Attention_distraction/project_healthy`

Its `analyses/` directory is the primary location for shared, precomputed
study outputs, including the corrected `all_subjects_training.csv` trial table.

Current Pd analyses use the explicitly configured repository cache
`analyses/eeg_pd/behavior_filtered/subject_session_pd.csv` together with its
trial-exclusion audit and provenance. The study-root
`training_session_condition_eeg_averages.csv` is a retired unfiltered artifact,
not a source for current Pd figures or statistics.

### Analysis repository

This repository contains reusable code in `code/src/posthoc_analysis/`, the
continuous analysis notebook in `notebooks/posthoc_analysis.ipynb`, and
publication figures in `figures/`.

The repository's `analyses/` directory is a portable fallback for a
precomputed CSV when the canonical study-root copy is unavailable. It should
not silently replace the canonical study output when that output is present.

## Required lookup order

Before recomputing an analysis from raw files:

1. Use an explicitly supplied input path, if one was provided.
2. Check `PROJECT_ROOT/analyses/` for the expected precomputed file.
3. Check this repository's `analyses/` directory as a fallback.
4. Only then regenerate from raw recordings, and print the reason that
   regeneration was necessary.

For filtered Pd, `rebuild_behavior_filtered_pd()` checks an explicit cache
directory first; otherwise it checks study-root and then repository caches.
It validates provenance and outputs before reuse. The current notebook passes
the repository `analyses/eeg_pd/behavior_filtered/` directory explicitly.

## Output policy

- Save reusable shared tables to `PROJECT_ROOT/analyses/` unless the caller
  explicitly specifies another location.
- For repository-local `analyses/`, retain only expensive-to-recreate,
  downstream-required, or QC artifacts. Follow `doc/analyses.md`; do not save
  ordinary notebook-derived summary/statistical tables there.
- Save manuscript-ready figure exports to this repository's `figures/`
  directory.
- Avoid creating duplicate raw-data-derived CSVs in the repository solely to
  make a figure when a validated precomputed source is available.
- If a file is not found, report every location checked in the error message.

## Training RT clock correction

Legacy `all_subjects_training.csv` files with text-derived `rt_samples` and
`rt_ms = rt_samples / 512 * 1000` are not validated RT sources. Regeneration
from non-practice GDF Status is required even if such a CSV exists. Corrected
outputs preserve `gdf_rt_ms` and label `rt_source = gdf_status` or the explicit
`trigger_text_fallback` for absent responses with aligned text RT <50 ms.
Text milliseconds remain in `text_rt_ms` without sample-rate conversion.
Validated GDF-only trial tables may be upgraded with
`apply_training_text_rt_fallback()`; no rescaling or new EEG/Stroop computation
is required. See `training_response_qc.md` for the audited fallback pattern.

## Seven-point Pd trajectory cache

The notebook explicitly uses repository `analyses/eeg_pd/seven_point/` for
expensive decoding-session sensor Pd waveforms. There was no precomputed
five-session Pd waveform source at the canonical study-root or repository
locations. The existing filtered pre/post cache is reused unchanged. See
`analyses.md` for cache provenance and the approved raw-run exceptions.

## Consolidated theta reference

The posterior-theta reference now defaults to the repository-local
`exploratory_results/theta_paper_reproduction/`; no Codex worktree is required.
`PD_THETA_REFERENCE_ROOT` remains an optional explicit override. Historical
absolute paths in cache manifests identify original provenance; loaders resolve
run checkpoints locally and still validate the recorded checksums.
