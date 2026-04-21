# AGENTS.md

## Project purpose

This repository is for **post hoc analysis** of the Pd-based closed-loop BCI study in healthy adults.

The primary purpose of this project is to:
- test whether Pd-based BCI training changes physiological, behavioral, and classifier outcomes relative to a mental-rehearsal control group
- generate high-quality, publication-ready figures in a clean **Nature Neuroscience-style**
- run statistical analyses that directly answer the project hypotheses
- run exploratory analyses

## Scientific context

This project evaluates whether modulation of distractor positivity (Pd) via a closed-loop BCI improves attentional control in a healthy population relative to a mental-rehearsal control group.

The key expected effects are:
- improved BCI modulation over runs/sessions in the BCI group
- improved online classifier performance over sessions
- increased Pd from Session 1 to Session 5 in the BCI group
- improved behavioral distractor suppression from Session 1 to Session 5
- transfer/generalization to Stroop performance

The control group is a **mental-rehearsal control**. Group identity will typically be specified in config.

## Primary analysis targets

Unless explicitly told otherwise, assume the most important analyses are:

### BCI training measures
- threshold trajectories across runs and sessions
- classification accuracy across runs and sessions
- session-wise AUPRC during training (session 1 vs 5)

### EEG measures
- Pd peak amplitude
- Pd positive area under the curve
- pre vs post comparisons (Session 1 vs Session 5)
- group comparisons (BCI vs control)
- ERP visualizations
- topoplots before vs after training

### Behavioral measures
- attentional capture cost from the Additional Singleton task
- Stroop interference effect
- pre vs post comparisons (Session 1 vs 5)
- group comparisons (BCI vs control)

### Statistical analyses
Default goal: test the stated hypotheses first, then support with exploratory analyses.

## Hypotheses

1. **Training modulation**
   - whether threshold values increase over runs/sessions in the BCI group more than in control
   - whether classification accuracy remains relatively stable under adaptive thresholding

2. **Classifier performance**
   - whether online/post hoc AUPRC improves over sessions in the BCI group more than in control

3. **EEG / Pd outcome**
   - whether Pd measures increase from Session 1 to Session 5 in the BCI group more than in control

4. **Behavioral distractor suppression**
   - whether attentional capture cost changes from Session 1 to Session 5 as predicted

5. **Generalization**
   - whether Stroop interference changes from Session 1 to Session 5 as predicted

## Figure priorities

This repository is optimized for making **high-quality, publication-ready figures**.

When generating figures:
- prioritize clarity, minimalism, and scientific readability
- prefer clean layouts with restrained text and strong labeling
- make figures appropriate for a **Nature Neuroscience-style** presentation
- preserve clear group distinctions, session distinctions, and condition distinctions
- make uncertainty/error bars and sample sizes easy to interpret
- do not clutter panels with unnecessary decorations
- avoid redundant plots unless they answer distinct questions
- make pre/post comparisons visually obvious
- make control vs BCI contrasts visually obvious

Common expected figure types:
- ERP waveforms (pre vs post; distractor vs no distractor; group comparisons)
- Pd-focused summary plots
- topoplots
- threshold trajectories
- session-wise AUPRC summaries
- behavioral summary plots
- statistical summary figures

## Documentation map

Use the following files depending on the task:

- `experiment_structure.md`
  - overall experiment design: sessions, tasks, runs, and trial counts

- `file_structure.md`
  - folder hierarchy, file locations, naming patterns, and session-level extra files

- `file_contents.md`
  - contents of non-decoder files, including `.analysis.txt`, `.triggers.txt`, `.behoutput.txt`, and `online_info` files

- `triggers.md`
  - trigger meanings, task-dependent trigger logic, trial reconstruction, and RT computation

- `decoder_mat.md`
  - decoder `.mat` structure, classifier interpretation, saved transforms, and performance fields

- `eeg_preprocessing.md`
  - preprocessing pipeline, feature extraction, data shapes, and decoder-matched transformations

## Critical analysis rules

- Exclude **practice runs by default**
- Treat trigger meanings as **task-dependent**.
- Do not confuse **decoder feature conventions** with **ERP plotting conventions**.
- For decoder-matched processing, use values stored in decoder files rather than re-deriving them.
- Do not recompute xDAWN filters, normalization, or feature selection for decoding-session inference; use the saved transforms from the decoder.
- Treat `baseline_idx` and `resample.time` as relative to epoch indexing with time 0 at `epochOnset`.
- Keep trial alignment explicit across all preprocessing and feature-extraction steps.
- **Always validate file integrity**: Use comprehensive validation for all loaded files to ensure data quality and catch issues early.

## Statistical analysis guidance

When implementing statistical analyses:
- default to the documented group and time/session contrasts unless explicitly asked otherwise
- keep dependent variables, factors, and contrasts explicit in code
- save intermediate summary tables used for stats when useful
- report effect sizes when appropriate

If the exact statistical design is ambiguous in code, prefer preserving a structure that makes the factors explicit rather than hiding them inside ad hoc scripts.

## Coding rules

- Prefer small, targeted edits over unnecessary rewrites.
- Keep loaders generic across subjects and decoder types.
- Validate expected field presence and expected data shapes.
- Warn and continue when possible if a file is partially unexpected, rather than failing immediately.
- Do not invent undocumented preprocessing or decoding logic.
- Reuse existing helper functions when possible.
- Keep plotting code modular and reusable.
- Keep analysis code readable and publication-focused rather than overly clever.
- Use **Jupyter notebooks** for most analysis and plotting workflows.
- Structure code so that:
  - preprocessing, feature extraction, and plotting can be run in separate cells
  - intermediate outputs can be inspected without rerunning the entire pipeline
- Avoid writing code that only runs as a full end-to-end script unless explicitly requested.

## Expectations for code changes

When implementing new analysis code:
- follow the documented preprocessing and decoder pipeline exactly
- make shape assumptions explicit in code
- add sanity checks where dimensionality is critical
- preserve compatibility with existing file organization
- prefer scripts/functions that produce directly interpretable outputs
- structure outputs so they can be used in manuscripts, presentations, or exploratory follow-up analyses

When implementing plotting code:
- default to publication-quality output
- use consistent panel labeling, axis formatting, and legend logic across figures
- save figures in high-quality formats appropriate for later editing or manuscript use

## Preferred Workflow

My preferred workflow:
- All reusable analysis code must live in `src/`.
- Any new analysis or plot request should be implemented as one or more reusable Python functions/modules in `src/`.
- After writing the function(s) in `src/`, update the notebook `notebooks/posthoc_analysis.ipynb` by appending a new section for that analysis.
- Keep adding new sections to this same notebook over time rather than creating separate notebooks.

Notebook rules:
- The notebook must contain only runnable analysis cells and markdown explanations.
- Do NOT copy function definitions into the notebook.
- The notebook should import functions from `src` and run them there.
- Cells should be written so they chain naturally from previous cells.
- Outputs and variables created in earlier notebook cells should be reused as inputs for later cells whenever appropriate.
- No demo/example cell is needed.
- Each new analysis should be added as a new section at the end of the notebook.

Structure for each new notebook section:
1. A short markdown cell describing the goal of the new analysis step
2. An import cell for any newly added functions from `src`
3. One or more runnable analysis cells that use the existing notebook state when appropriate
4. Output/inspection cells only where needed to verify correctness

Validation requirements:
- Every new function added to `src` must include explicit sanity checks and informative printed summaries.
- These checks should validate relevant expectations from the repository documentation and markdown files I provide.
- Before implementing a function, read the relevant markdown documentation (including `AGENTS.md` and other data/decoder documentation files) and use that information to guide assumptions and checks.
- Validate things such as, when relevant:
  - expected data dimensions/shapes
  - expected session/run/trial counts
  - required columns or fields
  - alignment between arrays/tables
  - time windows and sampling assumptions
  - preprocessing assumptions
- **File loading functions must include comprehensive validation**:
  - **Trigger files**: Validate row count (180 for training), column structure, trial numbering (1-60), trigger sequences ([4, stimulus, 64]), and data types
  - **Analysis files**: Validate row count (60 for training), column structure, trial indices (1-60), value ranges for all columns (task: 0-1, feedback: 1-3, positions: valid ranges, etc.)
  - **Experiment structure**: Validate file counts match expected runs per session (8 in session 1, 4 in session 5 for training)
  - **Data integrity**: Check for missing files, malformed data, and consistency between related files
  - **Error reporting**: Provide detailed error messages that help identify specific issues and affected files
  - **Existing functions**: `load_training_trigger_file()` and `load_training_analysis_file()` already implement this level of validation
  - **New functions**: Any new file loading functions (e.g., for .mat decoder files, .behoutput.txt Stroop files, etc.) must include equivalent comprehensive validation
- Use `validate_all_files_comprehensive()` for thorough validation of all data files against documentation specifications
- If documentation and actual code/data do not fully match, do not silently guess. Make the safest minimal assumption, surface it clearly in prints/comments, and keep the code modular.

Code style:
- Reusable logic belongs in `src`, not in the notebook.
- Prefer small, focused, reusable functions over monolithic code.
- Use clear names and concise docstrings.
- Reuse existing project structure when appropriate.
- Do not hardcode paths, subject IDs, or analysis settings unless clearly intended; prefer config-driven or notebook-variable-driven inputs.

Notebook style:
- Treat `notebooks/posthoc_analysis.ipynb` as a continuous step-by-step analysis notebook.
- Each section should be easy to run in sequence inside VS Code Jupyter.
- Assume prior cells may already define variables that later cells can use.
- Keep cells small enough that I can run one step at a time and inspect outputs before continuing.
- Add short markdown explanations before major code blocks.

When I ask for a new analysis or plot, do the following:
1. Read the relevant repository markdown/context files first
2. Determine what reusable functions are needed
3. Implement or modify those functions in `src/`
4. Append a new section to `notebooks/posthoc_analysis.ipynb`
5. Make the notebook cells runnable in sequence using prior notebook state when appropriate
6. Include validation prints/checks in the underlying functions
7. At the end, summarize what files you changed, what functions you added, and what notebook section you appended

Important rule:
- Do not put substantial analysis logic only in the notebook.
- The notebook is for orchestration, step-by-step execution, validation, and visualization.
- Reusable logic must remain in `src/`.

## Common tasks

Common tasks in this repository include:
- loading subject/session/run data
- dropping practice runs
- parsing behavioral and trigger files
- computing RT from trigger timing
- loading decoder files
- reproducing decoder-matched preprocessing
- generating ERP plots
- generating topoplots
- plotting before/after summaries
- plotting BCI vs control comparisons
- computing Pd summary measures
- computing behavioral summary measures
- running hypothesis-driven statistical analyses
- running exploratory analyses

## Default mindset for the agent

When working in this repository, prioritize:
1. scientific correctness
2. reproducibility
3. clean mapping from hypotheses to analyses
4. high-quality figure generation
5. readability and maintainability of code