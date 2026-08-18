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
study outputs. For example, the group Pd waveform source file is:

`PROJECT_ROOT/analyses/training_session_condition_eeg_averages.csv`

Use this location by default for downstream analysis and figure generation.

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

For `training_session_condition_eeg_averages.csv`,
`resolve_training_condition_eeg_averages_csv()` implements this order and
prints the selected path.

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
