> Consolidated on 2026-11-05: use the **Consolidated midfrontal theta, Stroop theta, and intervention behavior** section of the continuous notebook. Original subsection numbers below identify the migrated steps; alpha sections 9–11 were removed.

# Midfrontal theta over the intervention

Section 14 of `notebooks/posthoc_analysis.ipynb` uses reusable functions in
`code/src/posthoc_analysis/theta_intervention.py` to create seven-point
trajectories: Pre, intervention sessions 1–5, Post.

Use the original distractor theta parameters: 4:0.5:8 Hz zero-mean Morlet power,
100–500 ms target, −300 to −100 ms baseline, −1 to +1.5 s padded epochs,
continuous zero-phase 0.1–20 Hz filtering, and the same full 61-channel
standard_1020 spherical Laplacian. Compute power at Fz/FCz/Cz after the full
spatial transform. No decoder transforms or classifiers are used.

The existing behavior-screened pre/post assessment values are reused exactly.
For intervention `_decoding` files, keep every experimental trial, including
incorrect responses, timeouts and all RT values. Do not add EEG amplitude,
ICA or other trial exclusions. Practice runs remain outside the experimental
sample. Follow documented e22/e37 Session 5 first-folder practice-like exclusions;
retain all five e27 Session 1 experimental-labeled recordings. The user explicitly
approved using seven available runs for e52 Session 3: Run 6
`e52_20260330155058_decoding.gdf` is missing, although its task files are present.
Record run counts and the missing source; retain original run numbers.

Validate all paired file shapes, integers, documented value ranges, 60-trial
indices, 180 text trigger rows, 30 no-distractor / 15 right / 15 left stimuli,
EEG/text/analysis trial alignment, sampling, channel order and real epoch padding.
Response trigger issues are reported without excluding stimulus-valid trials.
Run `validate_all_files_comprehensive()` for the assessment text-file audit;
its documented run-count warnings are separate from the new intervention checks.

Within each participant/intervention session/channel/frequency, average baseline
linear power across every trial and run, pooling both conditions. Each condition
uses this common session-specific baseline. Pool stimulus linear power across
its trials, divide by the baseline, apply 10log10, then average target samples,
frequencies and the three electrodes. Compute the all-trial condition by pooling
linear power before dB, not by averaging condition dB values. Weight participants
equally for group means; SEM is sample SD / sqrt(16).

Outputs reside in `exploratory_results/theta_paper_reproduction/intervention_trajectory/`:
`cache/` contains the validated source inventory, trial ledger, settings, run
checkpoints, session run counts, participant trajectories and audit files;
`figures/` contains the three-panel PDF/SVG/PNG, group summary CSV and figure
metadata. These local checkpoints are retained for reproducibility and reuse.
Section 14 is descriptive. Assessment and intervention eligibility differ
by request; phase boundaries are marked in the figure and documented in the notebook.

One e29 Session 5 run (`e29_20260309133610_decoding`) has an extra fixation
marker 8 samples after stimulus 23. All 60 stimulus identities align exactly
with text and analysis, and each has a distinct preceding approximately 1 s
fixation. The unmatched marker is reported; it changes no trial anchor or
trial inclusion. Other unexpected timing/alignment failures still halt processing.

The complete available intervention dataset contains 1,150 runs and 69,000
trials: 56,095 correct, 3,394 incorrect and 9,511 timeout trials. All are retained.
There are 160 participant/intervention-session cells and 32 complete
seven-point trajectories for each of the three conditions.

## Intervention mixed ANOVA

Section 15 uses `theta_intervention_anova.py` to test Group (between participants) × Distractor presence × Session (both within participants). The latter factors have two and five levels. Only intervention sessions 1–5 and the two separate conditions enter the model; neither assessment endpoints nor pooled-all-trial values enter. The 320 complete repeated cells represent 32 participants (16/group) and all 69,000 available intervention trials. No imputation or additional exclusions are used; e52 Session 3 retains its seven-run estimate.

`prepare_intervention_theta_anova()` verifies cache checksums/settings and writes the exact input and provenance. `run_intervention_theta_mixed_anova()` uses orthonormal repeated contrasts with separate subject-within-group error strata. Classical marginal-mean sums of squares independently verify every numerator, denominator, F, and total SS partition. Partial eta squared is SS_effect / (SS_effect + SS_error).

Greenhouse–Geisser epsilon is estimated from pooled within-group residual contrast covariance S: tr(S)² / [k tr(S²)], where k = 4 for session-containing contrasts. Correct both numerator and denominator degrees of freedom, preserving F; report these corrected p values for all Session terms. Two-level terms require no sphericity correction. Uncorrected p values are also saved. No multiplicity adjustment across the seven omnibus effects and no post hoc tests are applied. This is a balanced design, so conventional Type II/III omnibus sums of squares coincide.

Outputs under `intervention_trajectory/anova_2x2x5/` include `anova_input.csv`, `mixed_anova_2x2x5.csv`, `error_strata.csv`, `descriptives.csv`, provenance, and validation JSON. Input missing cells or inconsistent group assignments halt processing. Independent validation additionally checks one-dimensional effects against OLS t², invariance to row/condition/session order and common offsets/scaling, synthetic-data decomposition, and rejection of incomplete data.

The Group × Distractor presence effect is F(1,30) = 8.373, p = .00703, partial η² = .218. Group × Session (p = .847) and the three-way interaction (p = .389) are not significant. All other effects have p ≥ .082. These tests do not establish an intervention-specific change over sessions.

## Exploratory Group × Distractor presence follow-up

Section 16 calls `theta_intervention_posthoc.run_intervention_theta_group_condition_followups()` using the existing Section 15 input and omnibus table. It averages each participant's five session-level dB values within condition (equal session/participant weights). Four two-sided simple effects form one Holm-adjusted family: paired condition tests in each group and Welch group tests within each condition. Report Cohen dz for paired contrasts and pooled-SD Cohen d for group contrasts. Confidence intervals are marginal 95% t intervals, not adjusted for multiplicity. No additional trial/participant exclusions, raw EEG computation, or session-specific tests occur.

The interaction is also reconstructed using an equal-variance independent t test on participant condition differences, giving t² exactly equal to the original omnibus F. This is a re-expression of the existing interaction, outside the new four-test family, not an independent additional discovery. Numerical t/df/CI formulas and a manual Holm calculation independently verify the library outputs.

All results are saved in `intervention_trajectory/anova_2x2x5/group_condition_followups/`: participant condition means and contrasts, four-cell descriptives, simple effects, interaction contrast, and validation/provenance. BCI distractor − no-distractor is +0.242 dB (Holm p = .0127), control −0.056 dB (Holm p = .487); neither condition-specific group contrast survives Holm correction. The original interaction is +0.299 dB, 95% CI [0.088, 0.510], p = .00703. Interpretation concerns average condition selectivity over sessions, not change caused by training.

## Current uncorrected reporting (user-requested)

Sections 15–16 now explicitly pass `correction=None` to the ANOVA and follow-up functions. Current exports are under `intervention_trajectory/anova_2x2x5_uncorrected/`; previous corrected exports remain in `anova_2x2x5/` for provenance. Earlier descriptions of GG and Holm above document the historical options, which remain supported by the functions. The current analysis uses raw ANOVA degrees of freedom (1,30 or 4,120), raw two-sided simple-effect p values, and no GG or Holm adjustment. F/t statistics, participant values, confidence intervals and effect sizes are unchanged.

Uncorrected ANOVA p values: Group .091899; condition .081981; Session .101031; Group × condition .007031; Group × Session .934526; condition × Session .071851; three-way .396181. Uncorrected follow-ups: BCI condition contrast .003187; control condition contrast .472930; group contrast for distractor .038455; group contrast for no-distractor .243580.

## Independent statistical audit and exploratory linear trends

Section 17 uses `theta_intervention_audit.py` functions `audit_intervention_theta_statistics()` and `analyze_intervention_theta_linear_trends()`. The audit verifies participant cache hashes, figure hashes, all 320 input values, 20 plotted intervention group means/SEMs, session positions/labels and configured group assignments. Seven omnibus effects are independently reconstructed by statsmodels formula OLS with participant intercepts and sum-coded numeric factors. Separate models use participant means, condition values averaged over sessions, session values averaged over conditions, and distractor-minus-no-distractor session values. Subject-error dfs are 30, 30, 120 and 120. No original ANOVA helper or orthonormal-contrast implementation is reused. Maximum F disagreement is below 4e-14; no implementation bug was found.

Exploratory slopes use weights [-2,-1,0,1,2]/10 over intervention sessions 1–5, with independent `linregress` validation. Four within-group one-sample t tests and two between-group Welch t tests use participant slopes as observations. All six p values are two-sided and unadjusted per user preference. Marginal 95% CIs and standardized effects are saved. These tests were motivated by visual inspection, not prespecified.

On distractor trials, control slope is −0.1454 dB/session (p = .00572), BCI −0.0728 (p = .338), and BCI-minus-control +0.0726 (95% CI [−0.1052,0.2504], p = .408). Control declines within group, but the direct group slope difference is uncertain. BCI nonsignificance does not establish stability. No-distractor slopes are −0.0806 and −0.0837, with group difference p = .975. The plot shows between-participant SEM, not uncertainty of changes; Pre/Post assessment points are not part of the five-session ANOVA.

Outputs are under `intervention_trajectory/anova_2x2x5_uncorrected/statistical_audit/`: independent regression comparisons, verified plot values, individual slopes, within/between-group trends and audit/validation JSON. Earlier estimates, tests, and figures remain unchanged.

## Intervention response-outcome trajectories

Section 18 adds correct, incorrect and timeout response percentages using reusable `intervention_response_rates.py` functions `prepare_intervention_response_rates()` and `plot_intervention_response_rates()`. Feedback codes 1/2/3 map to correct/incorrect/timeout. All available experimental behavioral trials are retained, without RT or response exclusions. The source inventory/practice exceptions match the theta workflow, but the behaviorally complete e52 Session 3 Run 6 is included despite its missing EEG. This adds 59 correct and 1 timeout trial: total 69,060 trials, 1,151 runs, 56,154 correct, 3,394 incorrect and 9,512 timeouts.

The loader validates source/checkpoint hashes, existing 60-trial analysis schemas and value ranges, balanced condition counts, participant/session/run alignment, and the added analysis/trigger pair. Every outcome denominator is all trials within participant/session/condition. Three rates sum to 100%; group summaries are equally weighted means ± between-participant SEM, with 16 participants per cell. Trial counts and denominators are saved.

Outputs under `intervention_trajectory/response_outcomes/` contain participant rates, group rates, session run counts, source/QC provenance, and a nine-panel PDF/SVG/PNG. Columns show pooled/distractor/no-distractor trials and rows show correct/incorrect/timeout. Group colors are BCI orange and mental rehearsal blue. All nine panels use a common 0–100% y-axis range, with 20-percentage-point ticks. No inferential tests are performed.

## Cleaned intervention RT trajectories

Section 19 adds `intervention_rt_trajectory.py`: `load_intervention_rt_trials()`, `summarize_clean_intervention_rt()`, and `plot_intervention_rt_trajectory()`. GDF-derived RTs are reused from the main repository's existing `analyses/feedback_next_trial_rt/` reconstruction cache (canonical study-root cache absent). Cache hashes/source signatures and raw path/trial/stimulus/condition alignment are validated against the intervention inventory. All 60 trial positions per available run enter, without prior-feedback restrictions. The documented five-run e27 inventory is retained. The e52 missing-EEG run is a technical loss outside the participant-loss denominator; no text timing fallback is used.

Sequential trial exclusions: incorrect, timeout, missing/nonpositive/nonfinite RT, RT <150 ms, then one-pass mean ±3 sample SD among the remaining values within participant × session, pooling runs and conditions. Participants with >50% loss across all sessions are then excluded; exactly 50% is retained. This differs from the earlier n+1 analysis's participant/session/type outlier strata and first-trial exclusion.

Of 69,000 candidate trials, 3,394 incorrect, 9,511 timeout, 248 missing/invalid, 35 <150 ms and 583 3-SD outlier trials are excluded. 55,229 survive trial cleaning; 53,850 survive participant exclusions. Participants e29, e31, e41 and e46 are excluded. BCI n = 12 in Sessions 1–4 and 11 in Session 5 because e26 has no eligible RT then; mental rehearsal n = 16 throughout. Empty cells remain missing, with no extra whole-participant exclusion or imputation.

Outputs in `intervention_trajectory/rt_trajectory/` include source provenance, trial exclusions, outlier bounds, participant exclusions, group/participant RT tables, validation, and a three-panel pooled/distractor/no-distractor PDF/SVG/PNG. Group means equally weight participants, with between-participant SEM and shared millisecond limits. No inference is performed.
