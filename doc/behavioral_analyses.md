# 🧠 Behavioral Analyses Specification (Codex-Ready)

---

## 🎯 Purpose

This document defines **behavioral analyses** to be implemented by Codex using the dataset:

- all_subjects_training.csv

The goal is to:
- Compute behavioral metrics
- Generate publication-quality figures (Nature Neuroscience style)
- Run statistical analyses (ANOVA + post hoc tests)

---

## 📦 Input Data Schema (STRICT)

The CSV must contain:

| Column | Description |
|--------|------------|
| subject_id | Unique subject identifier |
| group | "control" or "experimental" |
| session_id | 1 = pre, 5 = post |
| run_id | Run number |
| trial_type | 0 = no distractor, 1 = distractor |
| feedback | 1 = correct, 2 = incorrect, 3 = timeout |
| rt_ms | Reaction time in ms |

---

## 🔁 Required Mappings

session_id:
- 1 → pre
- 5 → post

feedback:
- 1 → correct
- 2 → incorrect
- 3 → timeout

---

## ⚙️ Global Rules

### Aggregation
Compute ALL metrics at:
Subject × Session × Trial Type

NEVER run statistics at trial level.

---

### Trial Types
- distractor → trial_type == 1  
- no distractor → trial_type == 0  

---

### Plot Requirements
All plots must include:
- Mean ± SEM
- Individual subject points
- Within-subject lines (pre → post)

Color:
- control → blue  
- experimental → orange  

---

### Statistics Reporting
Always report:
- Test name
- Statistic
- Exact p-value
- Effect size (η² or Cohen’s d)
- 95% CI

---

### Outlier Reporting
Report % of removed trials per:
Subject × Session × Trial Type

---

# 📊 ANALYSIS 1 — Reaction Time (RT)

### Inclusion
Use:
- feedback == 1

Exclude:
- incorrect
- timeout
- rt_ms < 150

---

### Outlier Removal
Per Subject × Session × Trial Type:
- mean ± 3 SD

---

### Metric
Mean RT

---

### Plots
- RT (distractor)
- RT (no distractor)

Include:
- subject points
- within-subject lines

---

### Distribution Plot
Histogram BEFORE outlier removal with:
- mean
- ±3 SD

---

### Stats
3-way mixed ANOVA:
- Time (pre/post)
- Trial Type
- Group

---

# 📊 ANALYSIS 2 — Accuracy & Timeout

### Accuracy
Accuracy = correct / (correct + incorrect)

- exclude timeouts
- compute:
  - distractor
  - no distractor
  - overall

---

### Timeout Rate
timeouts / total trials

---

### Plots
- Accuracy (trial type)
- Overall accuracy
- Timeout rate

---

### Stats
- Accuracy → 3-way ANOVA
- Overall accuracy → 2-way ANOVA
- Timeout → 2-way ANOVA

---

# 📊 ANALYSIS 3 — Distractor Cost

### Definition
Cost = RT_distractor - RT_no_distractor

---

### Preprocessing
Same as RT:
- correct only
- rt_ms ≥ 150
- ±3 SD removal

---

### Computation
1. Mean RT per condition  
2. Compute cost per subject/session  

---

### Plots
- Bar plot (pre vs post, group)

Include:
- subject points
- within-subject lines

---

### Stats
2-way ANOVA:
- Time
- Group

---

# 🚀 Execution Steps

1. Validate columns  
2. Apply inclusion rules  
3. Remove outliers  
4. Aggregate  
5. Plot  
6. Run stats  
7. Return outputs  

---

# ❌ Do Not

- Include timeouts in accuracy  
- Run stats at trial level  
- Remove outliers globally  
- Use median RT  

---

# 🎨 Stroop Behavioral Analyses Specification

## 🎯 Purpose

This section defines **Stroop behavioral analyses** to be implemented by Codex using the dataset:

- all_subjects_stroop.csv

The goal is to:
- quantify Stroop accuracy and reaction time changes from pre to post
- compare experimental vs control group changes
- report timeout and RT-based trial exclusions clearly
- generate publication-quality figures that match the style of the training behavioral analyses

---

## 📦 Stroop Input Data Schema (STRICT)

The CSV must contain:

| Column | Description |
|--------|------------|
| subject_id | Unique subject identifier |
| group | "control" or "experimental" |
| session_id | 1 = pre, 5 = post |
| run_id | Stroop run number within session |
| trial_number | Trial number within run |
| trial_type | "congruent" or "incongruent" |
| stimulus | Word shown on the trial |
| ink_color | Ink color shown on the trial |
| response | 1 = correct, 2 = incorrect, 3 = timeout |
| reaction_time_ms | Reaction time in milliseconds |

---

## 🔁 Required Mappings For Stroop

session_id:
- 1 → pre
- 5 → post

trial_type:
- congruent → congruent trial
- incongruent → incongruent trial

response:
- 1 → correct
- 2 → incorrect
- 3 → timeout

reaction_time_ms:
- units are milliseconds

---

## ⚙️ Global Rules For Stroop

### Aggregation
Compute all summary metrics at:
Subject × Session × Trial Type

Also compute overall metrics at:
Subject × Session

NEVER run statistics at trial level.

### Timeout Handling
Remove all timeout trials before accuracy and reaction-time analyses.

Also report timeout exclusions as percentages:
- average percent of timeout trials removed per subject per session, averaged across runs
- overall average percent of timeout trials removed across all subjects and sessions

### Plot Requirements
Match the training behavioral-analysis plotting style as closely as possible.

All plots should include:
- Mean ± SEM
- Individual subject points
- Within-subject lines from pre to post when appropriate

Color:
- control → blue
- experimental → orange

### Statistics Reporting
Always report:
- Test name
- Statistic
- Exact p-value
- Effect size when appropriate
- 95% CI when appropriate

---

# 📊 STROOP ANALYSIS 1 — Timeout Summary

### Definition
Timeout trials are:
- response == 3

### Required Reporting
Report:
- average percent of timeout trials removed per subject per session, averaged across runs
- overall average percent of timeout trials removed across all subjects and sessions

Timeout percentage should be computed relative to all trials before any other exclusions.

---

# 📊 STROOP ANALYSIS 2 — Accuracy

### Inclusion
Use:
- response == 1 or response == 2

Exclude:
- response == 3

### Metric
Accuracy = correct / (correct + incorrect)

Compute:
- congruent accuracy
- incongruent accuracy
- overall accuracy with congruent and incongruent trials combined

### Plots
Create plots similar to the training accuracy analyses:
- congruent accuracy
- incongruent accuracy
- overall accuracy

### Statistics
Perform pre vs post experimental vs control accuracy analysis.

At minimum, statistically test:
- Time (pre/post)
- Group (experimental/control)

For trial-type-specific accuracy plots, keep congruent and incongruent analyses explicit rather than collapsing them silently.

---

# 📊 STROOP ANALYSIS 3 — Reaction Time

### Inclusion
Use:
- correct trials only

Exclude from RT computation:
- incorrect trials
- timeout trials

### Incorrect-Trial Reporting
Before RT outlier removal, report the overall percent of incorrect trials across all subjects.

### Outlier Removal
For each run separately:
- remove trials with reaction time outside mean ± 3 SD

### Required Reporting
Report:
- average percent of RT trials removed for each subject across all runs and sessions
- overall average percent of RT trials removed across all subjects

### Metric
Mean reaction time

Compute:
- congruent RT
- incongruent RT
- overall RT if useful for compact summary

### Plots
Create plots similar to the training RT analyses:
- congruent RT
- incongruent RT
- overall RT if included in the analysis summary

Plots should mirror the training-analysis visual style as closely as possible.

### Statistics
Perform experimental vs control reaction-time analysis with pre/post structure kept explicit.

At minimum, statistically test:
- Time (pre/post)
- Group (experimental/control)

If congruent and incongruent RT are analyzed separately, keep those analyses separate and explicit.

---

# 📊 STROOP ANALYSIS 4 — Stroop Effect

### Definition
Stroop effect = RT_incongruent - RT_congruent

### Preprocessing
Use the same RT cleaning rules as Stroop Analysis 3:
- correct trials only
- exclude incorrect trials
- exclude timeout trials
- remove outliers within each run using mean ± 3 SD

### Computation
1. Compute mean RT for congruent trials per subject × session
2. Compute mean RT for incongruent trials per subject × session
3. Compute Stroop effect per subject × session as incongruent minus congruent

### Required Reporting
Report:
- the same incorrect-trial reporting used for Stroop Analysis 3
- the same RT exclusion summaries used for Stroop Analysis 3

### Plots
Plots should be similar to Analysis 3 distractor cost:
- pre vs post grouped by experimental vs control
- mean ± SEM
- subject-level points
- within-subject lines

### Statistics
Statistical analyses should be similar to Analysis 3 distractor cost.

At minimum, statistically test:
- Time (pre/post)
- Group (experimental/control)

Keep the factor structure explicit in code and reporting.

---

# 🚀 Stroop Execution Steps

1. Validate Stroop CSV columns and value mappings.
2. Remove timeout trials and summarize timeout exclusion percentages.
3. Compute accuracy for congruent, incongruent, and overall trials.
4. For reaction time, remove trials at the run level using mean ± 3 SD.
5. Summarize RT exclusion percentages by subject and overall.
6. Compute Stroop effect from cleaned congruent and incongruent RT summaries.
7. Aggregate to the subject level before inferential statistics.
8. Generate publication-quality plots matched to the training-analysis style.
9. Run pre/post and group statistical analyses.
10. Return tables, figures, statistics, and exclusion summaries.

---

# ❌ Stroop Do Not

- Do not include timeout trials in accuracy or RT analyses.
- Do not compute RT outliers across all runs combined.
- Do not run statistics at the trial level.
- Do not hide congruent vs incongruent logic inside undocumented shortcuts.
- Do not compute Stroop effect from uncleaned RT data.
