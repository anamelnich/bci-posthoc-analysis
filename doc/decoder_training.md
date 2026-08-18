# Decoder Training Specification

## When to use this file

Read this file before:

- rebuilding a decoder rather than applying a saved decoder;
- reproducing cross-validation, class balancing, iterative pruning, or AUPRC;
- creating a new Session 5-derived feature set for longitudinal tracking; or
- interpreting the difference between the original online decoder and a new
  post-hoc decoder.

For saved-decoder field names and inference, see `decoder_mat.md`. For epoching
and feature shapes, see `eeg_preprocessing.md`.

## Scope and provenance

This specification records the training procedure in the original MATLAB
decoder implementation, verified from the `decoding-mode` branch of the
`anamelnich/distractor-classification` repository:

- `code/decoder/computeModel.m`
- `code/decoder/computeDecoderRight.m`
- `code/functions/iterativePrune.m`
- `code/functions/pruneTrialsMask.m`
- `code/functions/balanceRuns.m`
- `code/functions/xdawn.m`
- `code/functions/compute_r2.m`

It is a model-training specification, not a license to reuse the historical
saved transforms. A new post-hoc model must fit its own xDAWN filters,
normalization, selected features, LDA model, and probability calibration from
its own training data.

## Binary decoder definitions

The original training code constructs independent binary datasets:

- **Right decoder:** right distractor (`label 1`) versus no distractor
  (`label 0`). Left-distractor trials (`label 2`) are excluded.
- **Left decoder:** left distractor (`label 2`) versus no distractor
  (`label 0`). After exclusion of right-distractor trials, left-distractor
  trials are recoded to `label 1`.

Thus, neither lateralized decoder is trained against a pooled class containing
the other distractor side.

## Feature pipeline used inside each model fit

Each cross-validation fold must fit the following quantities using its training
trials only:

1. Baseline-correct each epoch when enabled.
2. Form the posterior ROI fixed-orientation difference waves, `right - left`,
   for conventional homologous P/PO electrode pairs: P2-P1, P4-P3, P6-P5,
   P8-P7, PO4-PO3, PO6-PO5, and PO8-PO7. This intentionally corrects the
   historical independent channel-ordering behavior of `find(ismember(...))`.
3. Fit xDAWN to the fold's difference-wave epochs and binary labels. The
   implementation builds simple class-average prototypes, uses a pooled
   signal covariance, and retains the two filters associated with class 1.
4. Project epochs with those two xDAWN filters.
5. Select the configured time indices and stride-subsample by
   `resample.ratio`; flatten as timepoints within component.
6. Z-score every candidate feature using the training-trial mean and sample
   standard deviation.
7. Compute each candidate's r2 as the squared Pearson correlation between
   that feature and the binary labels; retain the 30 largest values.
8. Fit the regularized LDA and its posterior calibration described below.

The held-out run receives only the transforms fitted in its corresponding
training fold. It must never contribute to fold-specific xDAWN fitting,
normalization, r2 ranking, feature selection, LDA fitting, or calibration.

## Class balancing and leave-one-run-out validation

Before each fold fit in iterative pruning, training data are balanced within
each run (`file_id`). For every run, randomly retain
`min(n_distractor, n_no_distractor)` trials from each class. The held-out run
is evaluated in full: it is not downsampled and includes trials that may no
longer be active in the pruning mask.

Cross-validation is leave-one-file/run-out. Fold predictions are combined over
held-out runs and scored with precision-recall AUC under a uniform class prior.
Do not substitute an ordinary prevalence-weighted average precision without
showing that it matches MATLAB's `perfcurve(..., 'Prior', 'uniform',
'xCrit', 'reca', 'yCrit', 'prec')` for the same labels and scores.

For the binary case, the verified MATLAB curve uses recall = TPR and
uniform-prior precision = `TPR / (TPR + FPR)` at each descending unique score
threshold. Its AUC is trapezoidal integration of precision against recall;
this is distinct from prevalence-weighted average precision.

Record the random seed, run/trial identifiers retained by balancing, all
iteration masks, and the fold-specific predictions. This is required for a
reproducible reimplementation.

## Iterative pruning

The original procedure runs 20 cumulative iterations. At each iteration it:

1. cross-validates the current active trial mask as above;
2. stores one posterior probability per original trial;
3. prunes with a fixed decision threshold of 0.5, not the adaptive online
   threshold; and
4. uses the resulting mask as the next iteration's active mask.

For each active class separately, `pruneTrialsMask` removes up to
`ceil(0.05 * n_active_class)` highly confident errors:

- class 1 trials with posterior below 0.2; and
- class 0 trials with posterior above 0.8.

It also removes up to `ceil(0.05 * n_active_trials)` active trials with
posteriors closest to 0.5. Removal is cumulative. The selected clean dataset
is the mask *before* the iteration having the largest cross-validated AUPRC.

The final original decoder is refit on all trials retained by that best mask;
there is no additional final `balanceRuns` call in `computeDecoderRight`.

For a new post-hoc model, a late pruning iteration can leave a fold with no
balanced training trials. The implementation must stop before that untrainable
iteration, retain all completed masks/history, and select the maximum-AUPRC
mask among valid completed iterations. It must report the stop reason rather
than fabricate trials or silently alter the pruning rule.

## Classifier and posterior calibration

The saved right and left decoders use the same classifier settings:

- MATLAB `fitcdiscr` with `DiscrimType = 'linear'`;
- uniform class priors;
- regularization `Gamma = 0.05` (verified across the saved right/left
  decoder files);
- feature scaling disabled in the classifier because z-scoring has already
  occurred upstream.

After LDA fitting, the code does **not** use the classifier's default
posterior. It extracts the linear discriminant coefficients `w` and intercept
`c`, calculates training distances `d = Xw + c`, then applies a custom sigmoid:

```text
p_low  = 0.025
p_high = 0.975
b_low  = -log((1 - p_low) / p_low) / percentile(d, 2.5)
b_high = -log((1 - p_high) / p_high) / percentile(d, 97.5)
b      = (b_low + b_high) / 2
P(class 1 | X) = 1 / (1 + exp(-b * (Xw + c)))
```

Here class 1 is the decoder's relevant distractor side. The custom probability
is essential to reproduce pruning, whose rules use posterior cut-offs. AUPRC
is rank-based, but its numerical replication still requires the original
uniform-prior PR definition.

## Post-hoc Session 5 feature-tracking analysis

For the planned analysis, new participant-specific right and left models will
be trained on Session 5 training-task data using the procedure above, with the
following intentional post-hoc settings:

- use zero-phase 0.1--20 Hz filtering;
- fit two xDAWN components afresh on the selected Session 5 clean trials;
- use the 200 ms-to-final-sample feature window with temporal stride 8;
- do not reuse saved `decoderR`/`decoderL` xDAWN filters, normalization,
  classifier objects, or `keepIdx`; and
- freeze the resulting Session 5 transform and top-30 feature coordinates
  before applying it to decoding runs in Sessions 1--5.

For each evaluation dataset, calculate r2 for these fixed features after the
frozen transform. Compute the full eligible Session 5 training-data value as a
descriptive reference, but label it as selection-linked because its features
were chosen from that same task. The independent Session 1--5 decoding runs
are the primary longitudinal evaluation dataset.

Persist the final transform per participant and decoder: the xDAWN filters,
full epoch-time axis, difference-channel order, feature-window/resampling
indices, z-score means and standard deviations, and selected feature indices.
Do not refit any of these quantities on Session 1 or decoding data. The
selection summary, retained clean-trial identities, and top-30 clean-training
r2 values should remain in cohort-level tables; the numeric transform itself
should be stored in a compact per-reference binary artifact.

### Longitudinal run availability

The evaluation manifest contains Session 1 training as the pre-intervention
baseline and every complete, non-practice decoding run from Sessions 1--5.
It preserves physical folder run identifiers rather than renumbering after a
missing file.  A run is eligible only when exactly one GDF, trigger, and
analysis file are present and its Status-channel stimulus events agree with
the trigger/analysis condition sequence.  Missing or malformed runs are
recorded as structured manifest issues and are never imputed.

For this feature-tracking analysis, use all five direct e27 Session 1 decoding
folders as decoding runs. This is intentionally distinct from the legacy
online-posterior reconciliation, which has a separate practice-inclusive
inventory and should not be used to alter the direct EEG run set.

A missing decoding run does not prevent fitting the Session 5 training
reference, provided all four Session 5 training runs needed for that
participant/decoder are valid. Apply the frozen reference to each complete
evaluation run only. Longitudinal statistics should use the available
subject-session values in a mixed-effects model; do not impute a missing EEG
run or silently convert a partial session into a complete one.

## Implementation checks

Before trusting a reimplementation, verify at least one representative fold
against MATLAB-compatible calculations for:

- selected balancing mask and class counts per run;
- xDAWN filter dimensions and component ordering;
- feature matrix shape and flattening order;
- training-only z-score parameters and top-30 indices;
- LDA coefficients, `Gamma`, and uniform prior;
- calibrated posterior values; and
- uniform-prior PR-AUC and chosen pruning iteration.

Do not describe a Python implementation as an exact reproduction until these
checks pass.
