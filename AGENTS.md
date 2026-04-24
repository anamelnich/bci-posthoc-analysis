# AGENTS.md

## Purpose

This repository is for post hoc analysis of the Pd-based closed-loop BCI study in healthy adults.

Primary goals:
- test whether Pd-based BCI training changes physiological, behavioral, and classifier outcomes relative to a mental-rehearsal control group
- generate publication-ready figures in a clean Nature Neuroscience-style
- run statistical analyses that directly answer the study hypotheses
- support exploratory follow-up analyses without breaking reproducibility

## Scientific Context

This project evaluates whether modulation of distractor positivity (Pd) via a closed-loop BCI improves attentional control in a healthy population relative to a mental-rehearsal control group.

Key expected effects:
- improved BCI modulation over runs and sessions in the BCI group
- improved online classifier performance over sessions
- increased Pd from Session 1 to Session 5 in the BCI group
- improved behavioral distractor suppression from Session 1 to Session 5
- transfer or generalization to Stroop performance

The control group is a mental-rehearsal control group.

## Priority Analyses

Unless explicitly told otherwise, prioritize:

### BCI training
- threshold trajectories across runs and sessions
- classification accuracy across runs and sessions
- session-wise AUPRC during training, especially Session 1 vs Session 5

### EEG / Pd
- Pd peak amplitude
- Pd positive area under the curve
- pre/post comparisons
- group comparisons
- ERP visualizations
- topoplots before vs after training

### Behavior
- attentional capture cost from the Additional Singleton task
- Stroop interference
- pre/post comparisons
- group comparisons

## Core Hypotheses

1. Training modulation:
   whether threshold values increase over runs and sessions in the BCI group more than in control
2. Classifier performance:
   whether online or post hoc AUPRC improves more in the BCI group than in control
3. EEG / Pd outcome:
   whether Pd measures increase from Session 1 to Session 5 in the BCI group more than in control
4. Behavioral distractor suppression:
   whether attentional capture cost changes from Session 1 to Session 5 as predicted
5. Generalization:
   whether Stroop interference changes from Session 1 to Session 5 as predicted

## Documentation Map

Use these files before implementing analyses:
- `experiment_structure.md`: sessions, tasks, runs, and trial counts
- `file_structure.md`: folder hierarchy, file locations, naming patterns, session-level extra files
- `file_contents.md`: contents of non-decoder files including `.analysis.txt`, `.triggers.txt`, `.behoutput.txt`, and `online_info` files
- `triggers.md`: trigger meanings, task-dependent trigger logic, trial reconstruction, and RT computation
- `decoder_mat.md`: decoder `.mat` structure, classifier interpretation, saved transforms, and performance fields
- `eeg_preprocessing.md`: preprocessing pipeline, feature extraction, data shapes, and decoder-matched transformations

## Non-Negotiable Analysis Rules

- Exclude practice runs by default.
- Treat trigger meanings as task-dependent.
- Do not confuse decoder feature conventions with ERP plotting conventions.
- For decoder-matched processing, use values stored in decoder files rather than re-deriving them.
- Do not recompute xDAWN filters, normalization, or feature selection for decoding-session inference; use saved transforms.
- Treat `baseline_idx` and `resample.time` as relative to epoch indexing with time 0 at `epochOnset`.
- Keep trial alignment explicit across preprocessing, feature extraction, and plotting.
- Always validate file integrity and data structure before analysis.

## Analysis and Statistics Preferences

- Default to the documented group and time contrasts unless explicitly asked otherwise.
- Keep dependent variables, factors, and contrasts explicit in code.
- Save intermediate summary tables when they help make the stats auditable.
- Report effect sizes when appropriate.
- Prefer `scipy` and `statsmodels` for inferential statistics when available.
- If the local environment has package or linkage issues, keep the code written for `scipy` and `statsmodels` and report the environment problem as a warning instead of silently rewriting the analysis around a different stats stack.
- If the exact statistical design is ambiguous, preserve an explicit factor structure rather than hiding logic in ad hoc scripts.

Behavioral-analysis preferences learned from this project:
- Warn and continue for partial datasets when scientifically safe.
- A reduced subject count can be acceptable for the current working dataset; surface it as a warning instead of treating it as an error.
- When reporting trial exclusions, prefer percentages rather than only raw counts.
- When helpful for manuscript reporting, include one pooled pre+post summary number in addition to separate pre and post summaries.
- For the training RT analysis, compute SD-based outliers across all correct trials combined within each `subject × session`, then summarize cleaned data by trial type afterward.

## Coding Style

- Prefer small, targeted edits over unnecessary rewrites.
- Reuse existing helpers and project structure when possible.
- Keep loaders generic across subjects and decoder types.
- Validate expected fields, shapes, and alignment explicitly.
- Warn and continue when a file or subject is partially unexpected, rather than failing immediately, unless the issue invalidates the analysis.
- Do not invent undocumented preprocessing or decoding logic.
- Keep analysis code readable, modular, and manuscript-oriented rather than overly clever.
- Reusable analysis logic belongs in `src`, not only in notebooks.

## Validation Requirements

- Every new function in `src` must include explicit sanity checks and informative printed summaries.
- Read the relevant repository documentation before implementing new analysis code.
- Validate dimensions, run/session counts, required columns, time windows, and alignment whenever relevant.
- If documentation and observed data disagree, make the safest minimal assumption, report it clearly, and keep the code modular.

File-loading expectations:
- Trigger files: validate row count, column structure, trial numbering, trigger sequence logic, and data types.
- Analysis files: validate row count, column structure, trial indices, and value ranges.
- Experiment structure: validate expected run counts by session.
- Data integrity: check for missing files, malformed data, and cross-file consistency.
- Existing functions such as `load_training_trigger_file()` and `load_training_analysis_file()` are the model for this level of validation.
- Use `validate_all_files_comprehensive()` when a full project-wide validation pass is appropriate.

## Plotting Style

- Default to publication-quality output.
- Prioritize clarity, minimalism, and scientific readability.
- Make pre/post and control/BCI contrasts visually obvious.
- Keep panel labeling, axis formatting, and legend logic consistent across figures.
- Save high-quality outputs suitable for manuscript editing.

Specific plotting preferences learned here:
- Avoid clutter and redundant decorations.
- Show subject-level values when they help interpretation.
- Use SEM unless there is a clear reason to use something else, and label the summary consistently with what is actually plotted.
- For combined pre/post summary plots, a grouped bar plot with subject-level scatter and no connecting lines is preferred when the goal is a compact summary rather than within-subject trajectory emphasis.
- If separate condition plots are shown, they can coexist with a more compact combined summary plot when they answer different questions.

## Workflow

- All reusable analysis code must live in `src/`.
- Any new analysis or plot request should be implemented as reusable Python functions or modules in `src/`.
- After updating `src`, append a new section to `notebooks/posthoc_analysis.ipynb`.
- Keep using this same notebook over time rather than creating many separate notebooks.

Notebook rules:
- The notebook should contain runnable analysis cells and markdown only.
- Do not copy substantial function definitions into the notebook.
- Import functions from `src` and run them from the notebook.
- Reuse existing notebook state when appropriate.
- Append each new analysis as a new section at the end of the notebook.

Preferred notebook section structure:
1. short markdown cell describing the goal
2. import cell for any new functions
3. runnable analysis cells using existing notebook state where appropriate
4. output or inspection cells only where useful

When asked for a new analysis or plot:
1. read the relevant repository documentation first
2. determine which reusable functions are needed
3. implement or modify those functions in `src/`
4. append a new section to `notebooks/posthoc_analysis.ipynb`
5. make notebook cells runnable in sequence
6. include validation prints and checks in the underlying functions
7. summarize what changed, including functions and notebook section

## Default Mindset

When working in this repository, prioritize:
1. scientific correctness
2. reproducibility
3. clean mapping from hypotheses to analyses
4. high-quality figure generation
5. readability and maintainability
