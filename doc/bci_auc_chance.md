# Online ROC AUC label-shuffle reference

## Scope and interpretation

The notebook's **BCI AUC Across Sessions** plot calls
`compute_bci_subject_session_auc()` in `bci.py`. It computes sklearn ROC AUC
from `decoding_task` (1 = distractor, 0 = absent) and the single recorded
`posterior_probability`. It pools runs within participant/session, then
averages participants equally for each group/session. Error bars are SEM
across participants. Ambivalent trials are included. This is not the
uniform-prior precision–recall AUC used in decoder training.

For label-independent scores, ROC AUC has expected value 0.5, with half credit
for ties. The new analysis supplies a **conditional fixed-score label-shuffle
reference**, not a null distribution for the full online BCI system.

## Source audit

Inspected the `decoding-mode` branch of `anamelnich/distractor-classification`
at tree commit `bdf2a9ff03119b94826d9865eff5e0e4190836ad`:

- [`code/cnbiLoop/ndf_main.m`](https://github.com/anamelnich/distractor-classification/blob/bdf2a9ff03119b94826d9865eff5e0e4190836ad/code/cnbiLoop/ndf_main.m):
  stimulus trigger 32 selects decoderR, 44 decoderL, and 8 decoderN. The log
  saves `[ex_posterior, threshold, code]`, one selected score per trial.
- [`code/decoder/computeModel.m`](https://github.com/anamelnich/distractor-classification/blob/bdf2a9ff03119b94826d9865eff5e0e4190836ad/code/decoder/computeModel.m):
  decoderN is a copy of R or L, chosen by larger offline TNR (L on a tie).
- [`visualInterface/Pd_utils.py`](https://github.com/anamelnich/distractor-classification/blob/bdf2a9ff03119b94826d9865eff5e0e4190836ad/visualInterface/Pd_utils.py):
  `generate_trials()` shuffles 30 absent/30 distractor labels and rejects any
  order containing four consecutive identical labels. It assigns 15 right
  and 15 left distractors. Position 2 is right, position 4 is left.

The analyzed table's 1,150 labelled runs satisfy these class-count and
maximum-streak constraints. This supports using the documented constraint;
it does not establish the exact historical software revision for every run.

Condition-dependent decoder selection can create score/label association even
without informative EEG. For example, suppose R always returns 0.2, L always
returns 0.8, and N is R. Equal numbers of right and left distractors then give
ROC AUC 0.75 against absent trials, even though both decoders ignore EEG.
Shuffling the recorded scores against labels would instead center at 0.5.

Consequently, the reported `p_label_shuffle` values are reference tail
probabilities for an artificial label-independence benchmark. They must not be
reported as evidence of above-chance EEG discrimination by the routed system.
They also do not test training improvement, a group-by-session interaction,
or generalization to the participant population. Holding observed scores
fixed does not simulate feedback-induced changes in later EEG.

A routing-preserving evaluation requires scores from every relevant saved
decoder on every trial (including the unselected decoder), validated against
the recorded selected score. For each randomized condition, select its decoder
score anew. The current `all_subjects_bci.csv` and logged posterior arrays lack
those counterfactual scores; they must not be inferred or filled from other
trials. A full closed-loop null would additionally need an explicit model of
feedback/history effects. No such reconstruction is performed here.

## Implementation and validation

Reusable functions live in `code/src/posthoc_analysis/bci_auc_chance.py`:

- `validate_bci_auc_reference_input()`: checks participant/group assignments,
  integer identifiers, unique keys, complete 60-trial runs, trial/analysis
  alignment, finite [0,1] scores, exact binary labels, 30/30 counts, streak
  constraints, and 15/15 side counts when positions are supplied. Reports
  missing participant/session/run coverage. Entire unlabelled runs are
  excluded as in the original plot; partial missing labels fail explicitly.
- `compute_bci_auc_shuffle_reference()`: uses 10,000 draws and seed 20261103
  by default. A dynamic-programming suffix count samples uniformly among
  permitted label orders, equivalent to shuffle-and-reject without the
  rejection overhead. Each run, participant, and session receives independent
  draws. Binary label symmetry gives an exact theoretical reference mean of
  0.5; Monte Carlo means fluctuate around it.
- ROC AUC uses average ranks over the **whole session**, preserving score
  ties and cross-run comparisons. Every observed rank AUC is checked against
  the original sklearn calculation. It does not average run-level AUCs.
- Each permutation averages participant/session AUCs equally within each
  group/session. The central 95% reference interval uses quantiles .025/.975;
  the one-sided 95th percentile uses .95. Neither is an observed-mean CI.
- Upper-tail probabilities use `(1 + count(null >= observed))/(B + 1)`;
  comparisons include floating-point ties within 1e-12. Holm adjustment uses
  all returned group/session points (10 in this cohort). Participant-level
  intervals are descriptive; no participant significance claims are made.
- Settings include the seed, NumPy version, analysis-input SHA-256, and
  interpretation. Arrays and summaries remain in notebook state rather than
  generating unnecessary repository-local analysis tables.
- `plot_bci_auc_shuffle_reference()`: paper typography, equal y-axis limits,
  BCI/mental-rehearsal panels, observed mean ± SEM, gray central reference
  bands, and a dotted one-sided threshold. Saves PDF and 300-dpi PNG under
  `figures/bci_auc_label_shuffle_reference.*`, preserving the original plot.
  With `layout="combined"`, both groups share one axis, with color-matched
  central 95% reference bands and actual mean shuffled AUC lines. The combined
  plot omits upper-percentile cutoffs and the separate theoretical 0.5 line,
  and matches the attached `figures/for paper/bci_auc_across_sessions.pdf`
  export: 255.4946095 × 228.5775 PDF points, axes at
  (54.848513, 32.35625)–(155.442112, 208.09625) points, Arial typography,
  the original title and right-side group legend. A compact shuffle key is
  added below the group legend. Fixed-page export avoids recropping and
  preserves these exact dimensions. The reference PDF remains unchanged. This version is exported as
  `figures/bci_auc_across_sessions_with_shuffle_reference.*` and has its own
  notebook section and plot cell, reusing the already computed results.

The notebook section **Online ROC AUC: constrained label-shuffle reference**
reuses `bci_auc_results['dataframe']`. The saved output records the validated
exclusion of 60 unlabelled e27 Session 1 run 1 rows, leaving 69,000 trials,
160 participant/session AUCs and 16 participants per group.

Tests in `code/tests/test_bci_auc_chance.py` compare the sampler with exhaustive
small legal sequences, check balance/streaks/symmetry, verify constant-score
ties, demonstrate pooled AUC differs from mean run AUC, check equal participant
weighting and reproducibility under input reordering, and reject malformed
labels, duplicate keys, missing labels, misaligned trials, and invalid scores.

## Execution results

Executed the new cells sequentially in the Pd_AI Python environment. All 160
observed participant/session AUCs reproduce the existing computation to
absolute tolerance 1e-12; all pre-existing notebook cells and metadata were
preserved. Six focused tests pass, including an independent sklearn comparison
of sampled null AUCs. The PDF/PNG layout was visually inspected.

With seed 20261103 and 10,000 draws, group reference means range from
0.499942 to 0.500064. The following one-sided 95th-percentile cutoffs are
**pointwise, unadjusted reference quantiles**, not Holm-adjusted thresholds:

| Session | BCI | Mental rehearsal |
|---|---:|---:|
| 1 | 0.512522 | 0.512656 |
| 2 | 0.510866 | 0.510910 |
| 3 | 0.510959 | 0.510851 |
| 4 | 0.510861 | 0.510940 |
| 5 | 0.512219 | 0.512601 |

The comprehensive source-file validator also checked 383 training runs: all
383 trigger and 383 analysis files passed content validation. It reported the
already documented e30 Session 1 exception (7 training runs rather than 8).
This is separate from the current decoding-run AUC input, whose retained runs
passed the targeted validation above.
