# decoder_mat.md

## When to use this file
Use this file when:
- loading subject decoder `.mat` files in Python or MATLAB
- understanding what fields exist in decoder files
- interpreting classifier outputs and decoder metadata
- reproducing decoder-based inference using saved transforms
- validating decoder structure before analysis

Do NOT use this file for:
- locating files on disk (see `file_structure.md`)
- understanding overall experiment/session design (see `experiment_structure.md`)
- interpreting trigger codes or event timing (see `triggers.md`)
- parsing non-decoder task files such as `.analysis.txt`, `.triggers.txt`, or `.behoutput.txt` (see `file_contents.md`)

---

## Overview

Decoder files are stored as subject-specific MATLAB `.mat` files and represent complete saved decoding pipelines, not just classifier weights.

Each subject has 3 decoder files:

- `subject_decoderR.mat`
- `subject_decoderL.mat`
- `subject_decoderN.mat`

All three decoder types are expected to be present for every subject.

All three decoder types have the same top-level fields and overall struct layout. They differ in which decoder they represent, not in file structure.

---

## Important loading note

Each decoder `.mat` file contains a single decoder struct.

When loading decoder `.mat` files in Python, code should not rely unnecessarily on a hard-coded MATLAB variable name. Instead, loader code should identify the main decoder struct and validate that expected top-level fields are present.

If expected fields are missing, loader code should warn and continue when possible rather than failing immediately.

Decoder-loading code should be generic across `decoderR`, `decoderL`, and `decoderN`.

---

## Key concept

A decoder file is a complete saved decoding pipeline. It contains:

- feature extraction configuration
- learned transforms
- trained classifier
- performance summaries
- metadata

Most analyses will use only a subset of fields, but loader code should still be aware of the full field set.

---

## Top-Level Fields

Top-level fields are stable and always present across subjects and decoder types.

Typical fields include:

- `Classes`
- `fsamp`
- `epochOnset`
- `numFeatures`
- `roi`
- `fisher_iscompute`
- `classify`
- `resample`
- `statsfeatures`
- `features`
- `spatialFilter`
- `leftElectrodeIndices`
- `rightElectrodeIndices`
- `psd`
- `baseline_iscompute`
- `baseline_idx`
- `balance_iscompute`
- `eegChannels`
- `eogChannels`
- `spectralFilter`
- `threshold`
- `thresholdMargin`
- `performance`
- `subjectID`
- `onlinePosteriors`
- `datetime`
- `params`

---

## Core Interpretation Rules

### Classes
- Always `[0 1]`
- `1` always means **distractor present**
- `0` always means **distractor absent**
- This interpretation does **not** depend on decoder type (`R`, `L`, or `N`)

### Posterior meaning
- Posterior outputs should be interpreted as the probability of class `1`
- Therefore, posterior values represent **probability of distractor present**

### Top-level vs `params`
- If the same information appears both as a top-level field and inside `params`, trust the **top-level field**
- In most standard analysis code, `params` can be ignored

### Threshold fields
- `threshold` and `thresholdMargin` are present in the decoder files
- These fields do not need to be used in standard offline analysis unless a specific analysis explicitly requires them

---

## `classify` Sub-Struct

The `classify` sub-struct contains the classifier pipeline and should be treated as essential for reproducing inference.

### Stable rules
- `classify.type` is always `'linear'`
- `classify.reduction.type` is always `'r2'`
- `classify.model` is the trained classifier used for inference
- `classify.funNormalize` is the saved normalization transform and should always be used
- normalization should **not** be recomputed from new data
- `classify.keepIdx` is applied before classification
- `applyPCA` can be ignored in this project

---

## Feature-Related Fields

### `features`
- `erp_iscompute` is always `0`
- `diffwave_iscompute` is always `1`

ERP features are not used in this project. Difference-wave features are always used.

### `resample`
- `resample.time` is the time window actually used for decoder features
- `resample.ratio` is applied after spatial filtering

### `statsfeatures`
- Present but unused in this project
- Can be ignored in standard analysis code

---

## Spatial Filter Fields

### `spatialFilter`
- Spatial filtering is always used
- `spatialFilter.type` is always `'xDAWN'`
- `spatialFilter.diff` contains the learned projection used for inference
- Dimensions are stable across subjects
- It is safe to assume 2 retained components in this project

---

## Fields Present but Usually Ignored

The following fields exist and should be recognized by loader code, but can usually be ignored in standard analyses unless explicitly needed:

- `threshold`
- `thresholdMargin`
- `psd`
- PSD-related subfields
- `statsfeatures`
- `applyPCA`
- `params`
- `onlinePosteriors` inside the decoder struct

### `onlinePosteriors`
- Always empty in saved decoder files
- Should be ignored
- Actual online posterior data should be taken from separate files in `online_info/`

---

## `performance` Sub-Struct

`performance` is always present and always includes its expected subfields.

Important interpretations:

- `performance.posteriors` = offline cross-validated evaluation posteriors
- `performance.labels` aligns one-to-one with `performance.posteriors`
- `performance.file_id` indexes specific runs used in training/evaluation
- `performance.nTrials` = number of retained trials after pruning
- `performance.history` is always present
- `performance.acc` and `performance.accuracy` are redundant summary fields
- `performance.thr` and `performance.threshold` are redundant summary fields

---

## Data-Type Assumptions

The following assumptions are safe in this project:

- `fsamp` is scalar numeric
- `epochOnset` is a sample index
- `numFeatures` matches the selected feature dimensionality
- `baseline_idx` is a vector of sample indices
- `leftElectrodeIndices` and `rightElectrodeIndices` are channel-index vectors
- `subjectID` is a subject string such as `'e21'`
- `datetime` is a MATLAB datetime object

---

## Practical Coding Guidance

Decoder-loading code should:

- be generic across `decoderR`, `decoderL`, and `decoderN`
- validate that expected top-level fields are present
- warn and continue when possible if something unexpected is encountered
- use saved normalization and classifier objects rather than recomputing transforms
- keep awareness of unused fields, since they may become relevant in future analyses
