# eeg_preprocessing.md

## When to use this file
Use this file when:
- reproducing the EEG preprocessing and feature-extraction pipeline
- generating ERP plots that should match the decoder pipeline
- applying the same transforms used during decoder training
- understanding which steps are required before classification

Do NOT use this file for:
- locating files on disk (see `file_structure.md`)
- interpreting trigger codes or event timing (see `triggers.md`)
- understanding overall experiment/session structure (see `experiment_structure.md`)
- detailed decoder field documentation (see `decoder_mat.md`)

---

## Overview

This document describes the EEG preprocessing and feature-extraction pipeline used for decoder construction and decoding analyses.

Most of the key preprocessing settings are recoverable from the saved decoder `.mat` files, including:
- sampling rate
- temporal filter settings
- epoch onset
- baseline indices
- ROI/channel definitions
- xDAWN spatial-filter settings
- feature window
- resampling settings
- feature-selection settings
- classifier type
- saved normalization transform

---

## Pipeline Summary

The preprocessing and feature pipeline should be applied in the following order:

- 512 Hz EEG
- bandpass filtering from 0.1–20 Hz
- epochs from approximately -0.5 to +1.0 s relative to stimulus onset
- baseline correction using -0.2 to 0 s
- posterior P/PO homologous channels
- difference-wave representations
- xDAWN spatial filtering with 2 components
- feature extraction from the 0.2–0.5 s post-stimulus window
- temporal resampling with ratio 8
- saved z-score normalization from decoder training
- top 30 r²-ranked features
- regularized linear discriminant analysis
- iterative pruning during training

---

## 1. Sampling Rate

EEG is processed at:

- **512 Hz**

This value is stored in decoder files as `fsamp`.

---

## 2. Temporal Filtering

The pipeline applies a zero-phase bandpass filter to the analysis EEG channels
from:

- **0.1 to 20 Hz**

This filter definition is stored in decoder files in `spectralFilter`.

Use zero-phase filtering for offline post hoc preprocessing so phase timing of
ERP components is not shifted. This is the temporal filter that should be used
when reproducing the decoder preprocessing pipeline.

---

## 3. Epoching

Data are epoched relative to stimulus onset using a window of approximately:

- **-0.5 s to +1.0 s**

For training and decoding runs, stimulus onset means the second trigger in each
trial:

- `8` = no distractor stimulus
- `32` = right distractor stimulus
- `44` = left distractor stimulus

Do not use fixation trigger `4` as the default EEG epoching anchor. Fixation can
be used only for analyses that explicitly target fixation-locked activity.

Decoder files store:
- `epochOnset` = sample index corresponding to time 0

---

## 4. Baseline Correction

Baseline correction is applied using the pre-stimulus interval:

- **-0.2 to 0 s**
- For the default -0.5 to +1.0 s epoch at 512 Hz, this corresponds to epoch
  sample indices **154:256** using Python stop-exclusive indexing.
- Compute the mean over the baseline samples separately for each
  channel × trial, then subtract that mean from every timepoint in the matching
  channel × trial epoch.

Decoder files store:
- `baseline_iscompute`
- `baseline_idx`

- `baseline_idx` is defined relative to the epoch indexing, where `epochOnset` corresponds to time 0.
- These indices should be used **as-is**, assuming the same epoch alignment (i.e., time 0 at `epochOnset`).
- For example, if `epochOnset = 256`, then baseline indices such as `154:256` correspond to the time window -0.2 to 0 s.
- These saved baseline indices should be treated as the authoritative baseline definition for decoder-matched processing.

---

## 5. Channel Selection / ROI

Before any EEG plotting, epoching, ERP computation, topoplotting, or decoder-style
feature extraction, remove channels that are present in the raw `.gdf` files but
are not scalp EEG analysis channels:

- `M1`
- `M2`
- `EOG`
- `sens7`
- `sens8`

`M1` and `M2` are mastoid/reference channels. `EOG`, `sens7`, and `sens8` are
eye/sensor channels. These channels may be useful for artifact inspection, but
they should not be included in Pd/ERP plotting, topoplots, or downstream EEG
feature analyses unless an analysis explicitly targets artifact/EOG signals.

The decoder focuses on posterior homologous channels in the:

- **P/PO ROI**

Relevant decoder fields include:
- `roi`
- `leftElectrodeIndices`
- `rightElectrodeIndices`

These channel groups define the homologous posterior electrodes used for lateralized processing.

---

## 6. Difference-Wave Representation

Difference-wave features are always used in this project.

Stable rules:
- `features.erp_iscompute = 0`
- `features.diffwave_iscompute = 1`

Difference waves are constructed from homologous left/right posterior channels and are used as the input to spatial filtering and feature extraction.

### Implementation

- Difference waves are computed at the **sensor level (before spatial filtering)**.
- Homologous channel pairs are defined using:
  - `leftElectrodeIndices`
  - `rightElectrodeIndices`

For each homologous pair:
- **left-distractor trials**: compute (right − left)
- **right-distractor trials**: compute (left − right)

The resulting difference signals are then passed into the xDAWN spatial filtering step.

### Pd PO7/PO8 difference wave

For post hoc Pd plotting and measurement, compute the Pd as a PO7/PO8
difference wave on baseline-corrected stimulus-locked epochs. Ignore
no-distractor trials (`8`) for Pd averaging.

For each distractor trial:
- right distractor (`32`): `PO7 - PO8`
- left distractor (`44`): `PO8 - PO7`

The resulting single-trial Pd waveforms have shape `samples x distractor_trials`.
`compute_pd_difference_wave()` returns these waveforms and original trial
indices without an unfiltered grand average or plot. Apply the behavioral
retention mask before pooling distractor trials within participant/session.
Group figures then average participants equally and plot SEM across subjects.
The default pre/post plot window is `-0.2` to `0.7` s, with the decoder
construction window `0.2` to `0.5` s shaded and a common y-axis range of
`-1.5` to `1.5` microvolts.

### Pd positive area under the curve

For subject-level Pd summary statistics, compute positive area under the curve
within the decoder construction window (`0.2` to `0.5` s). Positive AUC is the
trapezoidal integral of `max(Pd, 0)` over time, producing units of
microvolts × seconds. Compute this separately for each subject and session from
the subject/session Pd waveform, then use those values for Group × Session
statistics.

### Behavior-filtered Pd revision

The notebook section **Pd pre/post waveforms and positive AUC after behavioral
exclusions** is the current Pd workflow. Historical unfiltered plotting, positive/negative AUC, mean-amplitude, and
cluster-permutation workflows are retained separately in `pd_historical.py`
and the explicitly historical notebook section. They do not replace the
current behavior-filtered Pd workflow. The current workflow uses corrected
training RTs and the existing behavioral filter:
remove incorrect responses and timeouts, missing RTs, and RT <150 ms, then
apply one pass of mean ±3 sample SD within each participant/session, pooling
all non-practice runs and all stimulus conditions. Exactly 150 ms and exactly
three SD are retained. Sub-50-ms text fallbacks necessarily fail the RT minimum.

Apply that mask to aligned raw stimulus-locked epochs before Pd averaging.
At the user's request, pool every retained distractor trial with equal weight
within participant/session; do not average runs or distractor sides equally
after exclusions. Right-distractor PO7−PO8 and left-distractor PO8−PO7 signals
are pooled; no-distractor trials influence the RT bounds but not Pd. Group means
and SEM weight participants equally. Canonical filtering, baseline correction,
native amplitude scaling, and positive-AUC integration remain unchanged.
Use the participant/session mean waveform for positive AUC, rather than
averaging single-trial positive areas. The original Group × Session mixed
ANOVA and four planned contrasts (unadjusted P values and pointwise 95% CIs)
are retained. Earlier cluster-permutation tables and their rerunnable code are preserved
as historical unfiltered sensitivity analyses.

The following **Filtered Pd figures with mean distractor RT reference lines**
section adds dashed Session-1/5 RT lines in the corresponding waveform hues.
`plot_behavior_filtered_pd_with_mean_rt()` reuses the exact retained distractor
mask and Status-derived RTs, averages trials within participant/session, then
averages participant means equally. These descriptive lines do not change the
Pd waveforms or positive-AUC statistics.

The separate **Exploratory positive Pd AUC: 350–450 ms** section calls
`analyze_filtered_pd_auc_window()` in `pd_window_stats.py` on the same filtered
waveforms. It changes only the integration window and retains the existing
ANOVA/contrast procedures. As for 200–500 ms, integration uses recorded samples
inside the requested bounds without interpolation: 51 samples from 351.5625
to 449.21875 ms at 512 Hz. Results are stored separately in the notebook;
P values and pointwise CIs are unadjusted, including across windows.

---

## 7. xDAWN Spatial Filtering

xDAWN spatial filtering is always used.
Difference-wave signals are computed **before applying xDAWN spatial filtering**.

Stable rules from decoder files:
- `spatialFilter.type = 'xDAWN'`
- 2 retained spatial components
- the saved learned projections in the decoder should be reused

Important rule:
- use the **same xDAWN components saved in the decoder files**
- do **not** recompute xDAWN components from new data if the goal is to reproduce the trained decoder pipeline

### xDAWN implementation note

The MATLAB xDAWN implementation computes class-evoked responses, estimates evoked covariance, and solves a generalized eigendecomposition relative to signal covariance. The leading components are retained as spatial filters, and their corresponding patterns are also stored. In this project, the component count is stable and can be assumed to be 2.

---

## 8. Feature Window and Temporal Resampling

Features are extracted from a post-stimulus time window and then temporally downsampled.

### Decoder-defined fields

- `resample.time`  
  - vector of sample indices defining the feature window within each epoch
  - corresponds to **0.2–0.5 s post-stimulus**

- `resample.ratio`  
  - temporal downsampling factor (typically `8`)

#### Implementation

**Feature window selection**

   - Restrict the signal to indices specified by `resample.time`
   - These indices are defined relative to epoch indexing, where:
     - time 0 = `epochOnset`

**Temporal resampling**

   - Apply subsampling using the saved ratio:
     - `ratio = resample.ratio`

   - Resampling is performed by **stride-based indexing**:
     - no filtering or interpolation is applied
     - every `ratio`-th sample is selected along the time dimension

   - In other words:
     - `resampled_signal = signal[resample.time][::resample.ratio]`

#### Important notes

- Resampling is applied **only within the feature window**, not on the full epoch
- The effective sampling rate after resampling is:
  - `fsamp / resample.ratio` (e.g., 512 / 8 ≈ 64 Hz)

- Feature dimensionality is determined by:
  - `n_resampled_timepoints × n_components`

- In this project:
  - expected: **40 features = (resampled timepoints) × 2 components**

- If feature dimensionality differs from expectation, the issue is likely due to:
  - incorrect `resample.time`
  - incorrect `resample.ratio`
  - or incorrect ordering of preprocessing steps

---

## 9. Normalization

Features are z-scored before classification.

Important rule:
- use the **saved normalization transform** in the decoder file
- do **not** recompute z-scoring from decoding-session data if the goal is to match trained-decoder inference

Relevant decoder field:
- `classify.funNormalize`

This is especially important when applying the pipeline to decoding-session data.

---

## 10. Feature Selection

After feature construction and normalization, the pipeline keeps:

- **top 30 r²-ranked features**

Stable rules:
- `classify.reduction.type = 'r2'`
- `numFeatures = 30`
- selected feature indices are stored in `classify.keepIdx`

These saved feature indices should be used when reproducing the trained decoder pipeline.

---

## 11. Classification

The classifier is regularized linear discriminant analysis (LDA).

Stable rules:
- `classify.type = 'linear'`
- the trained classifier is stored in `classify.model`

The MATLAB training code fits LDA using uniform priors and a regularization parameter (`Gamma`) from the classifier config. The raw LDA distance is then transformed into a posterior probability using a sigmoid fit defined from lower and upper quantiles of the distance distribution.

`decoder.classify.model` stores the exact callable classification function used for inference.

The model has the form:

`p(x) = 1 / (1 + exp(-b * (x * w + mu_coef)))`

Where:
- `x` = feature matrix after normalization and feature selection
- `w` = LDA weight vector
- `mu_coef` = intercept term from the fitted LDA model
- `b` = sigmoid slope parameter derived from training data

The output is posterior probability of class 1, where class 1 means distractor present.

To reproduce decoder inference, pass the normalized r2-selected features into `decoder.classify.model`

Do not retrain or refit the classifier during decoding.

---

## 12. Posterior Calibration

After LDA is fit, raw classifier distance is converted to posterior probability using a sigmoid transformation.

At a high level, the MATLAB implementation:
- extracts the linear discriminant weights and intercept from the fitted LDA model
- computes trial-wise signed distance from the decision boundary
- estimates a sigmoid slope parameter using lower and upper distance quantiles
- converts distance values into posterior probabilities using a logistic function

### Classification Model

`decoder.classify.model` stores the exact callable classification function used for inference.

The model has the form:

`p(x) = 1 / (1 + exp(-b * (x * w + mu_coef)))`

Where:
- `x` = feature matrix after normalization and feature selection
- `w` = LDA weight vector
- `mu_coef` = intercept term from the fitted LDA model
- `b` = sigmoid slope parameter derived from training data

The output is posterior probability of class 1, where class 1 means distractor present.

### Important rule

To reproduce decoder inference:
1. apply the saved normalization transform
2. apply the saved feature selection
3. pass the resulting features into `decoder.classify.model`

Do not retrain or refit the classifier during decoding.

---

## 13. Iterative Pruning

Iterative pruning is part of decoder training.

Iterative pruning, run-wise balancing, leave-one-run-out cross-validation, and
the exact posterior-based removal rules are documented in
`decoder_training.md`. Read that file before rebuilding a decoder. The saved
decoder files preserve outcomes through `performance`, `performance.history`,
and `performance.nTrials`, but do not by themselves specify the full training
procedure.

---

## 14. Data Shape Conventions

This section defines the expected data shapes at each stage of the preprocessing and feature-extraction pipeline.

### Raw EEG

- Shape:  
  `n_samples × 67 channels`

- Channels include:
  - 64 EEG
  - 2 EOG (removed before further processing)
  - 1 Status (removed before further processing)

### Epoched Data

- Shape:  
  `n_samples × n_channels × n_trials`

### Baseline

- Shape:  
  `1 × n_channels × n_trials`

- After baseline correction:
  - baseline is subtracted from each timepoint
  - epoch shape remains unchanged:

  `n_samples × n_channels × n_trials`

### After ROI Selection + Difference Wave

- ROI selection reduces channels to posterior homologous pairs

- Shape:  
  `n_samples × 7 channels × n_trials`

- These 7 channels correspond to the difference-wave signals computed from homologous left/right electrode pairs
#### Difference-wave convention (used for decoder features)

- Difference waves are computed using a **fixed subtraction rule**:
  - **right − left electrodes** (R − L)

- This rule is applied to all trials, regardless of distractor side.

- These difference-wave signals are used as input to spatial filtering and feature extraction.

#### Difference-wave convention for plotting (ERP)

For visualization (e.g., Pd plots), difference waves are computed using a **contralateral convention**:

- **left-distractor trials**: compute (right − left)
- **right-distractor trials**: compute (left − right)

This ensures that:
- positive values correspond to **contralateral activity**
- signals are aligned across distractor conditions

#### Important distinction

- **Decoder features** use a fixed subtraction rule:  
  → (right − left)

- **ERP plots** use a condition-dependent contralateral rule

These two conventions are intentionally different and should not be confused.

### xDAWN Spatial Filter

- Spatial filter matrix shape:  
  `7 channels × 2 components`

### After xDAWN Projection

- Shape:  
  `n_samples × 2 components × n_trials`

### After Feature Window + Resampling

- Data are restricted to the 0.2–0.5 s window and temporally resampled

- After reshaping into feature vectors:

  `n_features × n_trials`

- In this project:

  - `n_features = n_resampled_timepoints × 2 components`
  - Expected shape:  
    **40 features × n_trials**

- This value should be consistent across all decoders if preprocessing is correct

### After Normalization

- Shape remains unchanged:

  `n_features × n_trials`

### After Feature Selection (r²)

- Top 30 features are retained

- Final feature matrix:

  `30 × n_trials`

## Important Notes

- All transformations preserve trial alignment across steps
- Feature matrices are typically transposed to `n_trials × n_features` before classification
- If feature dimensionality differs from expected values (e.g., not 40 before selection), preprocessing may be incorrect

---

## Practical Guidance

If the goal is to generate ERP plots or decoder-matched feature representations:

- use decoder-stored sampling, filtering, baseline, ROI, xDAWN, feature-window, resampling, and normalization settings
- use saved xDAWN components rather than recomputing them
- use saved z-score normalization rather than recomputing it from decoding-session data
- treat the 0.2–0.5 s window as the main decoder-relevant feature interval
- treat iterative pruning as a training-stage procedure whose outcome is reflected in decoder performance fields, but whose exact logic lives in code

## Seven-point Pd positive AUC trajectory

The notebook's **Pd positive AUC across pre, five training sessions, and post**
section reuses the filtered pre/post waveforms, and adds all distractor trials
from decoding Sessions 1–5. `pd_longitudinal.py` uses the same canonical
stimulus-locked preprocessing and PO7/PO8 contra-minus-ipsi single-trial
waveforms. All available distractor trials are pooled equally before positive
rectification and 200–500 ms integration; participants are weighted equally
for group mean ± SEM. Do not average rectified single-trial areas.

Named practice folders are omitted. Explicit user-approved raw-run exceptions
retain all five e27 Session-1 runs and all seven e22/e37 Session-5 runs, and
report the missing e52 Session-3 GDF while retaining available runs. These
choices are specific to this Pd figure and do not change online-posterior
alignment rules. The figure uses poster typography (12-pt axis labels, 10-pt ticks and legend) and labels the
behavior-filtered pre/post versus unfiltered training inclusion policies.

## Posterior theta seven-point trajectories

`posterior_theta.py` adapts the separately validated midfrontal theta pipeline
for PO7/P7/P5, PO8/P8/P6, and the mean of all six electrodes. Its notebook
section uses 4:0.5:8 Hz zero-mean Morlet power (original 3.83–4.90-cycle schedule),
100–500 ms target, −300 to −100 ms power baseline, and −1 to +1.5 s epochs at
512 Hz. Continuous zero-phase 0.1–20 Hz FIR filtering and the original
61-channel template surface Laplacian precede posterior channel selection.
There is no voltage baseline, ICA, amplitude rejection, or decoder transform.

Pre/post reuse the exact original theta behavioral mask: correct responses,
observed Status RT ≥150 ms, then one-pass participant/session ±3 sample-SD
bounds pooling runs and conditions. Intervention includes all outcomes and RTs.
Practice/missing-run exceptions match the theta inventory, detailed in
`analyses.md`. Keep all 32 participants. The five plotted conditions are right
distractor (32), left distractor (44), pooled distractors (32/44), no distractor
(8), and all trials. Pooled distractors are not simultaneous bilateral stimuli.

Each condition shares a participant/session/channel/frequency baseline pooled
across all retained trials and runs. Pool stimulus linear power before dB;
then average time, frequencies, and ROI electrodes. Both numerator and baseline
use the same eligibility mask. Group summaries give participants equal weight;
error bars are between-participant SEM. Plots only; no inferential statistics.

### Posterior theta factorial ANOVA

`posterior_theta_stats.py` implements Group (2, between participants) ×
Session (7, within) × Trial type (left distractor/right distractor/no distractor,
within) × EEG side (left/right ROI, within). The final notebook section also
fits intervention-only Session (5) and assessment pre/post Session (2) models.
All models retain 16 participants/group with complete repeated cells (1,344,
960, and 384 observations). Pooled distractor/all-trial categories and the
bilateral ROI are excluded from these models; no new raw computation, trial
exclusion, participant exclusion or imputation occurs.

Generalize the validated midfrontal mixed-ANOVA method using tensor products
of orthonormal mean/Helmert contrasts. Each within effect has its own
Subject(Group) error stratum. Estimate GG epsilon from pooled within-group
residual covariance, applying it to every multi-df within effect and matching
Group interaction. Retain raw and corrected df/p values, SS/error SS, F, and
partial eta squared. Independent original-cell marginal decompositions verify
all SS/F; original-space covariance eigenvalues verify epsilon; scalar OLS
regressions verify one-df effects via t². The complete SS partition and partial
eta-squared identity are checked. Tests additionally cover factorial contrast
regressions, level/subject order and offset/scale invariance, and rejection of
missing/duplicate/nonfinite data or unbalanced groups.

The seven-point model mixes phases with different contexts and trial masks.
No-distractor target positions also differ from distractor-present trials;
these models do not isolate target location. Marginal effects equally weight
trial-type/hemisphere cells rather than pooling all trials before dB. GG is a
sphericity correction, not a multiple-testing adjustment. All 15 effects in
each of the three overlapping models are exploratory; there is no additional
multiplicity adjustment, post hoc testing or midline-target sensitivity analysis.

Executed results: Group × Session p = .511191 (seven points), .486152
(intervention), .136769 (pre/post). The four-way interaction p = .287645,
.206878, .535319, respectively. Group × Trial type p = .015090 and .019628
in the first two models, versus .440805 pre/post; this concerns mean condition
profiles rather than change over time. Report p_GG with its corrected df;
one-df terms have epsilon = 1. Full tables are saved in the executed notebook.

### Posterior theta: distractor presence across five intervention sessions

`run_posterior_theta_condition_anova()` in `posterior_theta_stats.py` fits
Group (2) × Condition (distractor present/absent, 2) × Session (1–5, 5).
Five intervention levels match the original midfrontal ANOVA; pre/post remain
excluded. The ROI averages all six posterior electrodes (PO7/P7/P5/PO8/P8/P6).
Reuse `both_distractors` and `no_distractor` summaries: the former pools left
and right trial linear power before dB, rather than averaging their dB values.
All other theta settings and intervention inclusion rules remain unchanged.

Validate 320 complete cells, 32 participants (16/group), 1,150 runs and 69,000
trials. The two conditions each contain 30 trials per retained run. Use the
same audited mixed-ANOVA estimator, with GG correction for Session and its
interactions, and retain raw p values and partial eta squared. There is no
adjustment across these seven tests or earlier exploratory analyses. Original
target-position differences between present and absent conditions remain.

Executed results: Group × Condition F(1,30) = 6.691416, p = .014782,
partial eta squared = .182370. Equal-session present-minus-absent means are
BCI +.291150 dB and mental rehearsal −.091370 dB. Group × Session p_GG =
.431101; Group × Condition × Session F(3.213335,96.400052) = .260459,
p_GG = .866223, partial eta squared = .008607. These results concern an
average condition contrast, without evidence of differential session change.
The appended notebook section displays all seven effects, descriptives and
validation/provenance; no raw EEG recomputation or additional post hoc tests.

### Exploratory posterior Group × Condition follow-ups

`posterior_theta_posthoc.run_posterior_theta_group_condition_followups()`
matches the latest midfrontal follow-up notebook (`correction=None`). Average
the five session dB estimates equally per participant/condition, then run four
two-sided tests: paired present-minus-absent t tests within each group and
Welch BCI-minus-mental-rehearsal t tests within each condition. Report raw
exploratory p values, marginal 95% t intervals, paired Cohen dz, and independent
pooled-SD Cohen d. Optional Holm correction applies to all four simple effects
as one family; it was not applied in this matched analysis.

The present-minus-absent difference is +.291150 dB in BCI (95% CI
[.030409, .551892], t(15)=2.380026, p=.031011, dz=.595007) and −.091370 dB
in mental rehearsal (CI [−.268451, .085711], t(15)=−1.099784, p=.288764,
dz=−.274946). Between-group comparisons are nonsignificant for distractor
present (difference +.701226 dB, p=.300222) and absent (+.318705 dB, p=.599314).
The between-group difference in condition contrasts is +.382520 dB (95% CI
[.080519, .684522], t(30)=2.586777, p=.014782, d=.914564). This reconstructs
the existing Group × Condition interaction, not a fifth follow-up hypothesis.

The implementation validates the complete cohort/cells and trial counts,
independently checks t/df/CI formulas, and requires interaction t²/F and p
agreement. All numeric results were also compared with the original midfrontal
function on identical posterior inputs. No session-specific tests, new trial
exclusions, or change-over-session claims. The final notebook section retains
results, participant summaries and provenance in memory and displayed outputs.

### Separate posterior condition trajectory figures

`plot_posterior_theta_condition_trajectories()` in `posterior_theta.py` exports
separate distractor-present and distractor-absent seven-point figures for the
bilateral six-electrode ROI. It matches the midfrontal trajectory style: BCI
orange, mental rehearsal blue, equal-participant mean ± SEM, 16/group,
Pre/1/2/3/4/5/Post labels, and dashed assessment/intervention boundaries.
Both conditions share y limits derived from all mean ± SEM values. Retain
original phase-specific inclusion and pooled linear-power distractor estimates.
Validate 448 complete observations and original phase trial totals before
plotting. PDF/SVG/PNG exports and input/output hashes are saved in
`figures/posterior_theta/`; notebook section **Posterior theta: separate
distractor and no-distractor seven-point figures** displays both previews.
