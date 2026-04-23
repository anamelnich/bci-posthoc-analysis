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
