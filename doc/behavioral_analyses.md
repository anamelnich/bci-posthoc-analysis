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
- nonmissing rt_ms ≥ 150 (all retained values remain GDF Status-derived)

Exclude:
- incorrect
- timeout
- missing RT (including unresolved GDF response alignment)
  - Exclusion tables count missing correct-trial RT separately. Incorrect/timeout
    trials with missing RT remain counted only under non-correct responses.
- rt_ms < 150

The audited 47 absent GDF responses use aligned text RTs of 1–46 ms in
`rt_ms`, explicitly labeled `trigger_text_fallback`, while `gdf_rt_ms` stays
NaN. They contribute to descriptive <150 ms counts but cannot enter the
retained RT analysis. Sequential exclusion tables count only correct trials
under `n_removed_rt_lt_150`; incorrect fast trials are already counted as
non-correct. For all-response fast-RT counts, use `rt_ms < 150` before the
feedback filter. Current counts: 53 fast trials total, 27 correct and 26
incorrect, including the 47 text fallbacks. See `training_response_qc.md`.

---

### Outlier Removal
Per Subject × Session, pooling both conditions and all runs:
- Compute mean ± 3 **sample** SD (`ddof=1`) after the inclusion filter.
- Exclude values strictly outside those bounds, then average retained trials
  within Subject × Session × Trial Type.
- Change scores are Session 5 minus Session 1.
- Rebuild trial-level RTs and rerun this cleaning after clock corrections; do
  not merely rescale previously cleaned means.

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
Balanced 2 × 2 × 2 mixed-design ANOVA, implemented by
`run_training_rt_mixed_anova()` on the existing participant-level mean RT table:
- Group (BCI/control): between participants
- Session (pre/post): within participants
- Distractor Condition (absent/present): within participants

The helper requires one finite mean per participant × session × condition,
complete four-cell coverage, and equal group sizes. It does not drop participants
or modify preprocessing. Group uses Subject(Group) as its error term. Session,
Condition, and Session × Condition each use their corresponding interaction
with Subject(Group); interactions with Group use the same respective stratum.
Orthogonal normalized contrasts of the four repeated cells give the classical
balanced mixed ANOVA sums of squares. All seven effects have numerator df=1
and error df=N−2 (30 for the current 32 participants), so no sphericity correction
is required. Report F, exact p, and partial η². This replaces the former
subject-fixed-effect OLS model with a shared residual; existing RT post hoc
contrasts and their 95% CIs are unchanged and remain unadjusted for multiplicity.
No new ANOVA effect-size confidence intervals are calculated.

The continuous notebook contains **Distractor-task RT: 2 × 2 × 2 mixed ANOVA**,
which reuses `analysis1_results["subject_session_summary"]`. Primary Analysis 1
also calls this helper when rerun. `stats_anova` retains `anova_table` and
`anova_report`, adds error strata and diagnostics, and no longer returns the
obsolete OLS `model` or `r_squared`.

---


### Secondary between-group RT-change comparisons by distractor condition
`compare_training_rt_changes_by_condition()` reuses the same participant mean
RTs as the mixed ANOVA. For distractor-present and distractor-absent trials
separately, compare individual post-minus-pre changes between BCI and control
using two-sided Welch tests. Report unadjusted p values (no Holm correction),
pointwise 95% Welch confidence intervals for the BCI-minus-control difference,
and Cohen's d using the pooled SD of participant change scores. Negative
estimates indicate greater RT reduction in BCI. The helper shares validation
and computation with the Stroop change comparisons; the existing raw-training
change-score workflow also calls it. No extra trial exclusions are applied.
The appended **Distractor RT: BCI versus control change by condition** section
reuses `analysis1_results["subject_session_summary"]` and reports all outputs.


# 📊 ANALYSIS 2 — Accuracy & Timeout

### Accuracy
Accuracy = correct / (correct + incorrect)

**Current distractor-accuracy analysis:** first exclude every trial with
`rt_ms < 150`, regardless of feedback, using the corrected training table
(including explicitly labeled sub-50-ms text fallbacks). Exactly 150 ms is
retained. Then exclude timeouts from the denominator and count both correct
and incorrect responses within participant × session × distractor condition.
Do not use the correct-only, SD-cleaned RT table. No ±3-SD or upper-RT exclusion
applies to accuracy. Missing RT is not classified as below 150 ms: otherwise
eligible correct/incorrect trials remain included, with explicit missing-RT
counts and a warning (none in the current corrected table).

`summarize_training_accuracy_after_rt_filter()` validates the input and returns
participant accuracies, sequential exclusions, group means/SEM and descriptive
95% t confidence intervals. `analyze_training_accuracy_mixed_anova()` combines
this with `run_training_accuracy_mixed_anova()`, which shares the balanced
2 × 2 × 2 mixed-ANOVA engine and effect-specific error strata with RT. The
analysis requires complete four-cell coverage and equal group sizes; partial
η² and unadjusted p values are reported for all seven effects. No accuracy
post hoc tests or ANOVA effect-size confidence intervals are added.

This supersedes the earlier accuracy analysis without a latency filter and its
OLS ANOVA. The historical notebook section and accuracy figures remain
labeled as historical; `analyze_accuracy_and_timeout()` preserves unfiltered
accuracy definitions but now uses valid mixed-ANOVA error strata;
use the appended **Distractor-task accuracy: exclude RT <150 ms, 2 × 2 × 2 mixed ANOVA**
section for current accuracy results. Overall accuracy and timeout definitions
remain unchanged; their inferential error strata were corrected in the
2026-11-05 rerun audit below. Missing RT, if
present, is handled as described above.

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
Use a **2 × 2 × 2 mixed-design ANOVA**: Group (BCI/mental-rehearsal control)
between participants, with Session (pre/post) and Congruency
(congruent/incongruent) within participants. Accuracy remains
correct/(correct + incorrect), pooled across non-practice runs within each
participant × session × congruency. Exclude timeouts only; no minimum RT,
upper RT, or SD-based RT exclusion is applied to Stroop accuracy.

`summarize_stroop_accuracy()` validates the original consolidated trials with
the existing per-run behavioral validator, checks identifiers and group
assignments, and returns participant counts, unchanged accuracies, exclusions,
and group means/SEM with descriptive 95% t confidence intervals.
`run_stroop_accuracy_mixed_anova()` uses the same balanced mixed-ANOVA engine
as distractor RT/accuracy, with Congruency explicitly labeled. Each effect uses
its corresponding participant error stratum. Complete four-cell coverage,
equal group sizes, and finite accuracies in [0,1] are required; participants
are never silently dropped. All seven numerator dfs are 1, error df is N−2,
and sphericity correction is unnecessary. Report unadjusted p values and
partial η². No follow-up contrasts or effect-size confidence intervals are added.

`analyze_stroop_accuracy_mixed_anova()` combines the summary and statistics
without generating figures. The existing `analyze_stroop_accuracy()` workflow
now also uses this joint test: `stats['mixed']` replaces the historical
separate congruent/incongruent/overall OLS tests. Overall accuracy remains
available descriptively. The appended notebook section **Stroop accuracy:
2 × 2 × 2 mixed ANOVA** contains the current executed results; statistics in
older saved notebook outputs are superseded. Accuracy figures and all Stroop
RT preprocessing/statistics are unchanged.

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
For each participant × session:
- first exclude timeout and incorrect responses
- pool correct-response RTs across both non-practice runs and both congruency conditions
- calculate one mean and sample SD (`ddof=1`) on that pooled set
- remove trials strictly outside mean ± 3 SD in a single, non-iterative pass
- apply no minimum RT cutoff

The previous per-run threshold calculation has been removed. All Stroop RT,
Stroop-effect, and RT change-score workflows use this shared filter.
`summarize_stroop_reaction_time()` returns cleaned trials, participant means,
participant-session thresholds/counts, and an `exclusion_breakdown` reporting
timeouts, incorrect responses, SD outliers, total removed, and retained trials.
Every percentage in that breakdown uses all non-practice trials as its
denominator; the three exclusion categories are mutually exclusive.
`outlier_report` now has one row per participant/session (including `n_runs`),
and subject reports use `mean_percent_removed_across_sessions`. The existing
`overall_exclusion_summary` describes SD exclusions among correct trials;
use `percent_removed_total` for all exclusions among all trials.
The appended notebook section **Stroop RT: pooled-session ±3 SD exclusions**
contains the current executed exclusion counts. Earlier saved RT results and
figures used per-run trimming and are historical until rerun.

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
Use a **2 × 2 × 2 mixed-design ANOVA**: Group (BCI/mental-rehearsal control)
between participants, with Session (pre/post) and Congruency
(congruent/incongruent) within participants. Use the unchanged participant mean
RTs after the pooled-session cleaning above; exclude descriptive `overall`
rows from this ANOVA. Participants are weighted equally, regardless of their
retained trial counts.

`run_stroop_rt_mixed_anova()` reuses the shared balanced mixed-ANOVA engine:
complete four-cell coverage, finite positive means and equal group sizes are
required. Each effect uses its corresponding participant error stratum; all
seven numerator dfs are 1 and error df is N−2, so sphericity correction is
unnecessary. Report unadjusted p values and partial η². The function also
returns group means/SEM and descriptive 95% t confidence intervals in ms;
no post hoc tests or effect-size confidence intervals are added.

The existing `analyze_stroop_reaction_time()` workflow now returns
`stats['mixed']` (also `stats_anova`) instead of separate condition/overall
OLS tests. The old private Stroop RT OLS helper has been removed. Overall RT
remains descriptive. The appended **Stroop RT: 2 × 2 × 2 mixed ANOVA** notebook
section reuses `stroop_rt_pooled_results` without repeating exclusions and
contains the current executed statistics. Earlier saved OLS outputs are
superseded. Stroop-effect preprocessing is unchanged; its Group × Session
error strata were corrected in the 2026-11-05 rerun audit below.


### Secondary between-group comparisons by congruency
`compare_stroop_rt_changes_by_congruency()` reuses the four participant means
from the updated RT analysis. For each congruency, compute each participant's
post-minus-pre change and compare BCI versus mental rehearsal using a two-sided
Welch independent-samples t test. The contrast is (BCI post−pre) − (control
post−pre); negative values indicate a larger RT decrease in BCI. Per user
request, report unadjusted p values for both comparisons; no multiple-comparison
correction is applied. Report Welch df, pointwise unadjusted 95% CIs in milliseconds and
Cohen's d based on the pooled SD of participant changes. Complete four-cell
coverage and at least two participants per group are required.

These are secondary comparisons of changes, not tests of group differences at
a single session. They do not establish a difference between the congruency
conditions; that question is addressed by the three-way ANOVA interaction.
The appended **Stroop RT: BCI versus control change by congruency** section
reuses `stroop_rt_mixed_summary`. The existing change-score summary workflow
also uses this helper and reports the same unadjusted Welch results.

---

# 📊 STROOP ANALYSIS 4 — Stroop Effect

### Definition
Stroop effect = RT_incongruent - RT_congruent

### Preprocessing
Use the same RT cleaning rules as Stroop Analysis 3:
- correct trials only
- exclude incorrect trials
- exclude timeout trials
- remove outliers using participant-session mean ± 3 SD, pooling runs and congruency

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
4. For reaction time, retain correct responses and apply participant-session mean ± 3 SD, pooling runs and congruency.
5. Summarize RT exclusion percentages by subject and overall.
6. Compute Stroop effect from cleaned congruent and incongruent RT summaries.
7. Aggregate to the subject level before inferential statistics.
8. Generate publication-quality plots matched to the training-analysis style.
9. Run pre/post and group statistical analyses.
10. Return tables, figures, statistics, and exclusion summaries.

---

# ❌ Stroop Do Not

- Do not include timeout trials in accuracy or RT analyses.
- Do not pool participants or pre/post sessions when computing RT thresholds; pool runs and congruency within each participant/session.
- Do not run statistics at the trial level.
- Do not hide congruent vs incongruent logic inside undocumented shortcuts.
- Do not compute Stroop effect from uncleaned RT data.

## Rerun audit correction (2026-11-05): two-level mixed ANOVA

`run_behavioral_prepost_mixed_anova()` now supplies the Group × Session test
for distractor cost, Stroop interference, overall accuracy and timeout rate.
Group uses Subject(Group); Session and Group × Session use Session ×
Subject(Group). All three numerator dfs are 1 and error df is N−2. Complete
pre/post pairs, finite outcomes and balanced groups are required; no trials,
participants, outcome definitions or planned contrasts change. P values remain
unadjusted. The old cost/interference OLS model included redundant subject and
group columns and did not provide a valid between-group error term. The old
overall-accuracy/timeout OLS models ignored repeated measurements.

`analyze_accuracy_and_timeout()` retains its **historical unfiltered condition
accuracy definition** and descriptive figures, but now runs valid mixed ANOVAs:
the existing 2 × 2 × 2 engine for historical condition accuracy and the new
2 × 2 engine for overall accuracy and timeout. Current RT-filtered condition
accuracy remains in `analyze_training_accuracy_mixed_anova()` and is unchanged.
The rerun audit stores the former and corrected inferential tables separately;
old notebook OLS outputs are superseded.
