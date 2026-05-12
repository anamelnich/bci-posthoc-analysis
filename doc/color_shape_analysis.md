# Color/Shape Pre/Post Group Analysis Specification

## Purpose

This document describes the full step-by-step workflow for preprocessing, plotting, and statistically analyzing the Color/Shape task data.

The goal is to compare performance across:

- Session: `pre` vs `post`
- Group: `bci` vs `control`
- Trial type: `congruent` vs `incongruent`

This file is intended to guide Codex implementation. Follow the steps exactly unless the code or data structure makes a step impossible. If any data issue is encountered, produce a warning and continue running.

---

# 1. Global Rule: Never Stop Execution for Data Issues

Do **not** stop execution because of missing, malformed, or incomplete data.

Instead:

- Use `warnings.warn(...)`
- Log the issue in an output dataframe
- Skip only the affected file, row, subject/session, or analysis subset
- Continue processing the remaining data

This applies to:

- missing files
- missing columns
- unexpected group labels
- malformed values
- empty datasets after cleaning
- missing subject/session combinations
- insufficient data for ANOVA, regression, or plotting

If an analysis cannot be completed for a subset, print a warning and continue.

---

# 2. Required Packages

Use:

```python
from pathlib import Path
from collections import Counter
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.auto import tqdm
import pingouin as pg
import statsmodels.formula.api as smf
```

Suppress non-critical warnings only if needed, but still allow custom data-quality warnings to print.

Set default plotting behavior:

```python
plt.rcParams["figure.figsize"] = (7, 4)
```

Remove top and right plot spines where appropriate.

---

# 3. Data Folder

Use this folder:

```python
DATA_DIR = Path("/Users/hililbby/Library/CloudStorage/Box-Box/CNBI/Attention_distraction/project_healthy/color_shape_task")
```

All files needed for this analysis are in this folder.

---

# Current Dataset Notes

The configured study groups in `code/src/posthoc_analysis/config.py` contain
32 subjects:

- BCI: `e21`, `e22`, `e25`, `e26`, `e29`, `e30`, `e31`, `e32`, `e38`,
  `e39`, `e41`, `e44`, `e46`, `e49`, `e51`, `e55`
- Control: `e23`, `e24`, `e27`, `e33`, `e36`, `e37`, `e40`, `e42`, `e43`,
  `e45`, `e47`, `e48`, `e50`, `e52`, `e53`, `e54`

The current Color/Shape mapping file contains 29 of those configured subjects.
The following configured subjects are missing from the Color/Shape mapping file
and therefore cannot be loaded for this analysis:

- Missing BCI subjects: `e21`, `e22`
- Missing control subjects: `e23`

The following subjects are excluded before Color/Shape run loading:

- `e24` / subject `24`: incomplete Color/Shape data because post run
  `47.csv` lacks the required task columns.
- `e43` / subject `43`: overall Color/Shape accuracy QC outlier because post
  run `131` was more than 2 SD below the subject/session mean accuracy
  distribution.
- `e46` / subject `46`: overall Color/Shape accuracy QC outlier because post
  run `136` was more than 2 SD below the subject/session mean accuracy
  distribution.
- `e55` / subject `55`: overall Color/Shape accuracy QC outlier because pre
  run `100` was more than 2 SD below the subject/session mean accuracy
  distribution.

After these exclusions, the current Color/Shape analysis set contains 25
subjects:

- 12 BCI subjects
- 13 control subjects

The implementation also keeps only main-task rows (`block_type == 3`) for the
analysis tables because the color-training and shape-training rows are not used
for the planned Color/Shape analyses.

---

# 4. Subject Run Mapping File

A separate CSV file exists inside `DATA_DIR`:

```python
subject_runs.csv
```

This file contains one row per subject.

Current data note: the documented file `subject_runs.csv` is not present in the
Color/Shape task folder. The implementation first checks for
`subject_runs.csv`, then warns and falls back to the available `subj_runs.csv`.

Required columns:

```python
subject_id
group
run1
run2
```

Column meanings:

- `subject_id`: subject identifier
- `group`: subject group, either `"bci"` or `"control"`
- `run1`: pre-session run number for that subject
- `run2`: post-session run number for that subject

Use:

```python
run1 -> pre
run2 -> post
```

Do **not** hard-code any subject/run mapping in the notebook or script.

---

# 5. Run CSV Files

Each run is stored as a separate CSV file in the same folder as `subject_runs.csv`.

For each subject:

```python
pre_csv  = DATA_DIR / f"{run1}.csv"
post_csv = DATA_DIR / f"{run2}.csv"
```

Each run CSV should be loaded independently, cleaned, labeled, and analyzed as one session.

Each subject/session corresponds to exactly one run CSV:

```python
subject × pre  -> run1.csv
subject × post -> run2.csv
```

Do **not** concatenate multiple runs for a subject/session.

---

# 6. Analysis Unit

The main analysis unit is:

```python
subject × session
```

where:

```python
session = "pre" or "post"
```

Groups are:

```python
GROUP_LABELS = ["bci", "control"]
```

Do not convert group codes. The group labels come directly from `subject_runs.csv`.

---

# 7. Required Run CSV Columns

Each run CSV should contain the following task columns:

```python
REQUIRED_COLS = [
    "chose_best",
    "left_color_val",
    "right_color_val",
    "left_shape_val",
    "right_shape_val",
    "block_type",
    "relevant_dimension",
    "color_value_difference",
    "shape_value_difference",
    "rt",
]
```

If any required columns are missing from a run CSV:

1. Issue a warning listing the missing columns.
2. Add the issue to the loading log.
3. Set `status = "invalid_columns"`.
4. Skip that file.
5. Continue execution.

Do **not** raise an error.

---

# 8. Load `subject_runs.csv`

Load:

```python
subject_runs = pd.read_csv(DATA_DIR / "subject_runs.csv")
```

Validate that it contains:

```python
subject_id
group
run1
run2
```

If any columns are missing:

- issue a warning
- continue if possible
- skip affected rows if necessary
- do not stop execution

Validate that `group` contains only:

```python
bci
control
```

If unexpected values appear:

- issue a warning
- keep the values in the loading log
- skip those subjects from group-level summaries if needed
- continue execution

---

# 9. Load and Clean Each Run CSV

For each subject in `subject_runs.csv`, load two run files:

- `run1` as session `"pre"`
- `run2` as session `"post"`

For each run:

1. Build the CSV path:

```python
csv_path = DATA_DIR / f"{run_number}.csv"
```

2. If the file is missing:
   - issue a warning
   - set `status = "missing"`
   - skip this subject/session
   - continue execution

3. If the file exists:
   - read it with `pd.read_csv`
   - record the raw row count as `n_rows_raw`

4. Validate required columns.
   - If required columns are missing, issue a warning and skip the file.

5. Keep only rows where all required task variables are non-missing:

```python
block_type
chose_best
rt
relevant_dimension
left_color_val
right_color_val
left_shape_val
right_shape_val
color_value_difference
shape_value_difference
```

6. Convert numeric columns to numeric using `pd.to_numeric(..., errors="coerce")`:

```python
chose_best
left_color_val
right_color_val
left_shape_val
right_shape_val
block_type
color_value_difference
shape_value_difference
rt
```

7. Standardize `relevant_dimension`:

```python
lowercase
strip whitespace
```

8. Drop rows that became invalid after numeric coercion.

9. Reset the index.

10. Add metadata columns:

```python
subject_id
group
session
run_number
```

11. Record cleaned row count as `n_rows_after_cleaning`.

---

# 10. Loading Log

Create a loading log with one row per subject/session.

Required columns:

```python
subject_id
group
session
run_number
csv_path
status
n_rows_raw
n_rows_after_cleaning
notes
```

Possible `status` values:

```python
loaded
missing
invalid_columns
empty_after_cleaning
malformed
unexpected_group
```

Display the loading log sorted by:

```python
subject_id
session
```

Also print:

- number of loaded files
- number of missing files
- number of invalid files
- distribution of cleaned trial counts across loaded runs

---

# 11. Trial Definitions

For each loaded subject/session dataframe, define trial type variables.

## 11.1 Main Task Trials

Main task trials are:

```python
block_type == 3
```

Create:

```python
main_task_ix
```

## 11.2 Congruent Trials

A trial is congruent if the better option is better on both color and shape dimensions.

Congruent if either:

```python
left_color_val > right_color_val
AND
left_shape_val > right_shape_val
```

or:

```python
right_color_val > left_color_val
AND
right_shape_val > left_shape_val
```

Create:

```python
congruent_ix
```

## 11.3 Incongruent Trials

Incongruent trials are:

```python
not congruent_ix
AND
main_task_ix
```

Create:

```python
incongruent_ix
```

---

# 12. Relevant and Irrelevant Value Differences

For each trial, define:

```python
relevant_val_diff
irrelevant_val_diff
```

If:

```python
relevant_dimension == "color"
```

then:

```python
relevant_val_diff = color_value_difference
irrelevant_val_diff = shape_value_difference
```

If:

```python
relevant_dimension == "shape"
```

then:

```python
relevant_val_diff = shape_value_difference
irrelevant_val_diff = color_value_difference
```

If `relevant_dimension` is neither `"color"` nor `"shape"`:

- issue a warning
- set relevant/irrelevant differences to `NaN`
- continue execution

---

# 13. Accuracy Analysis

Use `chose_best` as the accuracy variable.

Accuracy is computed as:

```python
mean(chose_best)
```

## 13.1 Congruent Accuracy

For each subject/session, compute mean `chose_best` for:

```python
main_task_ix
AND
congruent_ix
```

Save as:

```python
acc_congruent
```

## 13.2 Incongruent Accuracy

For each subject/session, compute mean `chose_best` for:

```python
incongruent_ix
AND
irrelevant_val_diff > 0
```

Save as:

```python
acc_incongruent
```

If no trials are available for a condition:

- save `NaN`
- issue a warning
- continue execution

---

# 14. Reaction Time Analysis

Use `rt` as the reaction time variable.

RT should be computed only on correct trials:

```python
chose_best == 1
```

## 14.1 Congruent RT

For each subject/session, compute mean `rt` for:

```python
main_task_ix
AND
congruent_ix
AND
chose_best == 1
```

Save as:

```python
rt_congruent_correct
```

## 14.2 Incongruent RT

For each subject/session, compute mean `rt` for:

```python
incongruent_ix
AND
irrelevant_val_diff > 0
AND
chose_best == 1
```

Save as:

```python
rt_incongruent_correct
```

If no correct trials are available for a condition:

- save `NaN`
- issue a warning
- continue execution

---

# 15. Analysis Log

Create an analysis log with one row per subject/session.

Required columns:

```python
subject_id
group
session
run_number
n_valid_trials_used
n_main_task_trials_used
n_congruent_trials
n_incongruent_trials
acc_congruent
acc_incongruent
rt_congruent_correct
rt_incongruent_correct
```

Display the analysis log sorted by:

```python
subject_id
session
```

---

# 16. Subject Exclusion Rule

Exclude subjects whose mean congruent accuracy across pre and post is below 0.5.

For each subject:

```python
mean_congruent_accuracy = mean(acc_congruent across pre and post)
```

Keep subjects where:

```python
mean_congruent_accuracy >= 0.5
```

If a subject has missing congruent accuracy for one session:

- compute the mean using available values
- issue a warning
- continue

If a subject has no congruent accuracy values:

- exclude the subject
- issue a warning

Print:

```python
number of subjects kept
number of total subjects
dropped subject IDs
```

Apply this subject filter to all later summaries, plots, ANOVAs, heatmaps, and regressions.

---

# 17. Long-Format Summary Table

Create one row per:

```python
subject × session × congruency
```

Columns:

```python
subject_id
group
session
run_number
congruency
accuracy
rt_correct
```

Where:

```python
congruency = "congruent" or "incongruent"
```

For congruent rows:

```python
accuracy = acc_congruent
rt_correct = rt_congruent_correct
```

For incongruent rows:

```python
accuracy = acc_incongruent
rt_correct = rt_incongruent_correct
```

Save this dataframe as:

```python
summary_long
```

---

# 18. Group-Level Summary Table

Using only kept subjects, compute summaries by:

```python
group
session
congruency
```

For both accuracy and RT, compute:

```python
mean
sd
sem
n
```

Required output columns:

```python
group
session
congruency
accuracy_mean
accuracy_sd
accuracy_sem
rt_mean
rt_sd
rt_sem
n
```

---

# 19. Bootstrap Confidence Intervals

For plotting, compute bootstrap confidence intervals for each group/session/congruency.

Use subject-level values, not trial-level values.

Bootstrap procedure:

1. Drop `NaN` values.
2. Resample subjects with replacement.
3. Compute the mean for each bootstrap sample.
4. Repeat many times, for example:

```python
n_boot = 5000
```

5. Use percentile 95% CI:

```python
2.5th percentile
97.5th percentile
```

If fewer than 2 non-missing subjects are available:

- issue a warning
- return `NaN` confidence intervals
- continue

---

# 20. Plot Pre/Post Accuracy and RT by Group

Create a 2 × 2 plot.

Rows:

```python
accuracy
rt_correct
```

Columns:

```python
congruent
incongruent
```

For each subplot:

1. Split by group.
2. Use subject-level data from `summary_long`.
3. Compute mean for `pre` and `post`.
4. Compute bootstrap confidence intervals.
5. Plot pre/post means connected by a line for each group.
6. Add markers.
7. Add error bars using bootstrap confidence intervals.

Axis labels:

```python
accuracy y-axis = "p(Choose Best)"
RT y-axis = "RT on correct trials"
```

Figure title:

```python
Pre/post averages by group
```

Use clear legends and readable axis labels.

---

# 21. Mixed ANOVA for Accuracy

Run mixed ANOVA separately for:

```python
congruent
incongruent
```

Use `pingouin.mixed_anova`.

Model:

```python
DV = accuracy
within factor = session
between factor = group
subject = subject_id
```

Before running:

1. Filter to the current congruency condition.
2. Drop rows with missing accuracy.
3. Ensure each included subject has both pre and post rows.
4. Ensure both groups are represented.
5. If requirements are not met:
   - issue a warning
   - skip the ANOVA
   - continue

Print and save ANOVA results for each congruency condition.

---

# 22. Mixed ANOVA for RT

Run mixed ANOVA separately for:

```python
congruent
incongruent
```

Use `pingouin.mixed_anova`.

Model:

```python
DV = rt_correct
within factor = session
between factor = group
subject = subject_id
```

Before running:

1. Filter to the current congruency condition.
2. Drop rows with missing RT.
3. Ensure each included subject has both pre and post rows.
4. Ensure both groups are represented.
5. If requirements are not met:
   - issue a warning
   - skip the ANOVA
   - continue

Print and save ANOVA results for each congruency condition.

---

# 23. Value-Difference Accuracy Grids

For each subject/session, create two accuracy grids:

```python
congruent_grid = 3 x 4
incongruent_grid = 3 x 4
```

Rows correspond to the first 3 sorted unique values of:

```python
relevant_val_diff
```

Columns correspond to the first 4 sorted unique values of:

```python
irrelevant_val_diff
```

For each grid cell:

1. Select trials matching the current relevant and irrelevant value difference.
2. For the congruent grid, compute mean `chose_best` for:

```python
main_task_ix
AND
congruent_ix
AND
matching value-difference cell
```

3. For the incongruent grid, compute mean `chose_best` for:

```python
incongruent_ix
AND
matching value-difference cell
```

Save grids per subject/session.

If a cell has no trials:

```python
cell value = NaN
```

If fewer than 3 relevant levels or fewer than 4 irrelevant levels exist:

- issue a warning
- fill missing grid cells with `NaN`
- continue

---

# 24. Target/Distractor Value Grid

For each subject/session, create:

```python
target_distractor_grid = 3 x 4
```

First define `target_value` and `distractor_value`.

## 24.1 If relevant dimension is color

If:

```python
left_color_val > right_color_val
```

then:

```python
target_value = left_color_val
distractor_value = right_shape_val
```

Otherwise:

```python
target_value = right_color_val
distractor_value = left_shape_val
```

## 24.2 If relevant dimension is shape

If:

```python
left_shape_val > right_shape_val
```

then:

```python
target_value = left_shape_val
distractor_value = right_color_val
```

Otherwise:

```python
target_value = right_shape_val
distractor_value = left_color_val
```

## 24.3 Fill Grid

1. Use incongruent trials only.
2. Get sorted unique target values.
3. Get sorted unique distractor values.
4. Use the first 3 target levels and first 4 distractor levels.
5. For each grid cell, compute mean `chose_best`.

Save as:

```python
target_distractor_grid
```

If a cell has no trials:

```python
cell value = NaN
```

If insufficient target or distractor levels exist:

- issue a warning
- fill missing cells with `NaN`
- continue

---

# 25. Plot Incongruent Value-Difference Heatmaps

Create a 2 × 2 heatmap figure.

Rows:

```python
bci
control
```

Columns:

```python
pre
post
```

For each group/session:

1. Select kept subjects in that group/session.
2. Average `incongruent_grid` across subjects using `np.nanmean`.
3. Plot heatmap using `imshow`.
4. Flip the grid vertically with:

```python
np.flipud(grid)
```

Use:

```python
vmin = 0.5
vmax = 1.0
```

Axis labels:

```python
x-axis = "Irrelevant value difference"
y-axis = "Relevant value difference"
```

Ticks:

```python
x ticks = 0, 1, 2, 3
y ticks = 3, 2, 1
```

Colorbar label:

```python
p(Choose Best)
```

If no valid subjects are available for a group/session:

- issue a warning
- leave the subplot blank or mark it as unavailable
- continue

---

# 26. Plot Target/Distractor Value Heatmaps

Create a 2 × 2 heatmap figure.

Rows:

```python
bci
control
```

Columns:

```python
pre
post
```

For each group/session:

1. Select kept subjects in that group/session.
2. Average `target_distractor_grid` across subjects using `np.nanmean`.
3. Plot heatmap using `imshow`.
4. Flip the grid vertically with:

```python
np.flipud(grid)
```

Use:

```python
vmin = 0.5
vmax = 1.0
```

Axis labels:

```python
x-axis = "Distractor value"
y-axis = "Target value"
```

Ticks:

```python
x ticks = 1, 2, 3, 4
y ticks = 4, 3, 2
```

Colorbar label:

```python
p(Choose Best)
```

If no valid subjects are available for a group/session:

- issue a warning
- leave the subplot blank or mark it as unavailable
- continue

---

# 27. Regression on Target/Distractor Grid

For each group/session:

```python
bci pre
bci post
control pre
control post
```

Use the subject-level `target_distractor_grid`.

Convert the grid into a long dataframe with columns:

```python
subject_id
choice_acc
target_val
distractor_val
```

Where:

```python
target_val = target_id + 2
distractor_val = distractor_id + 1
```

Drop rows with missing `choice_acc`.

Run OLS regression:

```python
choice_acc ~ target_val * distractor_val
```

Use:

```python
statsmodels.formula.api.ols
```

Before running:

1. Confirm there are enough rows.
2. Confirm `target_val` and `distractor_val` vary.
3. If requirements are not met:
   - issue a warning
   - skip regression for that group/session
   - continue

Display the coefficient table for each group/session.

---

# 28. Required Outputs

The notebook or script should produce:

1. Loading log
2. Distribution of cleaned trial counts per run
3. Analysis log
4. Subject exclusion summary
5. Long-format subject-level summary dataframe
6. Group-level summary dataframe
7. Pre/post accuracy and RT plots
8. Mixed ANOVA tables for accuracy
9. Mixed ANOVA tables for RT
10. Incongruent value-difference heatmaps
11. Target/distractor value heatmaps
12. Regression coefficient tables for each group/session

---

# 29. Important Implementation Notes

- Do not hard-code subject/run mappings.
- Use `subject_runs.csv`.
- Each run is a separate CSV file.
- Each subject has one pre run and one post run.
- `run1` is always pre.
- `run2` is always post.
- Groups are `"bci"` and `"control"`.
- The analysis unit is subject-level, not trial-level.
- Accuracy is mean `chose_best`.
- RT is mean `rt` on correct trials only.
- Main task trials are `block_type == 3`.
- Mixed ANOVAs are run separately for congruent and incongruent trials.
- Subject exclusion is based only on mean congruent accuracy across pre/post.
- All data issues should produce warnings, not errors.
- Functions should keep running whenever possible.
