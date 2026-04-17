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

The pipeline bandpass filters the EEG from:

- **0.1 to 20 Hz**

This filter definition is stored in decoder files in `spectralFilter`.

This is the temporal filter that should be used when reproducing the decoder preprocessing pipeline.

---

## 3. Epoching

Data are epoched relative to stimulus onset using a window of approximately:

- **-0.5 s to +1.0 s**

Decoder files store:
- `epochOnset` = sample index corresponding to time 0

---

## 4. Baseline Correction

Baseline correction is applied using the pre-stimulus interval:

- **-0.2 to 0 s**

Decoder files store:
- `baseline_iscompute`
- `baseline_idx`

- `baseline_idx` is defined relative to the epoch indexing, where `epochOnset` corresponds to time 0.
- These indices should be used **as-is**, assuming the same epoch alignment (i.e., time 0 at `epochOnset`).
- For example, if `epochOnset = 256`, then baseline indices such as `154:256` correspond to the time window -0.2 to 0 s.
- These saved baseline indices should be treated as the authoritative baseline definition for decoder-matched processing.

---

## 5. Channel Selection / ROI

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

At a high level:
- training data are repeatedly pruned across iterations based on posterior probabilities
- balancing across runs is applied to ensure equal number of trials between classes
- leave-one-file-out cross-validation is used during evaluation
- performance is tracked across iterations
- the best iteration is selected using AUPRC
- retained trials from the best iteration are used to define the final decoder/training summary

The saved decoder files preserve pruning outcomes through:
- `performance`
- `performance.history`
- `performance.nTrials`

However, the **exact pruning logic** is not fully recoverable from decoder files alone.

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
