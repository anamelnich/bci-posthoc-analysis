> Consolidated on 2026-11-05: use the **Consolidated midfrontal theta, Stroop theta, and intervention behavior** section of the continuous notebook. Original subsection numbers below identify the migrated steps; alpha sections 9–11 were removed.

# Standalone midline-theta reproduction

Open `notebooks/posthoc_analysis.ipynb`, select **Pd_AI**, and run from the
first cell. The migrated section uses its own path/import cell and
reproduces the numerical results and three figures in the investigator's
four-page **Pd BCI – Midline theta result (22 October 2026)** PDF.

## Inputs

- This repository, including `code/src/posthoc_analysis/`.
- The 32 configured participants' Session 1 and Session 5 non-practice
  `_training.gdf` recordings and their paired `.analysis.txt` and
  `.triggers.txt` files, in the hierarchy documented in `file_structure.md`.
- The default study root is `PROJECT_ROOT` in `config.py`. Change
  `STUDY_ROOT` in the first notebook cell or set `PD_STUDY_ROOT` if needed.
- The Pd_AI Python environment. `requirements-theta.txt` records the versions
  used for this reproduction. The project package imports other project
  modules, so the file also includes their required runtime packages.

No prior notebook state, historical behavioral CSV, neural candidate table,
decoder `.mat`, Stroop data, source PDF, or `chanlocs64.mat` is a computational
input. The PDF is the numerical reference, not an input that must remain in
Downloads. Its SHA-256 is recorded in the notebook.

## Execution and caches

1. Discover source runs and validate GDF sampling/channel layouts and paired
   task files. Re-extract RT from Status events, then apply the existing
   behavioral cleaning and independently check the resulting mask.
2. Compute continuous-filtered surface-Laplacian single-trial Morlet power,
   or reuse exact matching checkpoints. Rebuild all necessary summaries.
3. Run the condition-specific midline mixed ANOVA.
4. Produce the pooled pre/post/change topography, baseline/stimulus/dB power
   table and three-panel bar figure, and pooled/within-group RT correlations
   and scatterplot.
5. Assert agreement with every numerical result printed in the supplied PDF.

`OPTIONAL_CACHE = None` in the consolidated notebook. The retained local cache
is `exploratory_results/theta_paper_reproduction/theta_cache/`. Reuse requires identical settings, matching raw source
metadata and paired-file hashes, intact power-cache hashes, and a signature
matching the freshly reconstructed RT mask and extraction code. Validated
checkpoints are copied into this notebook's own output directory; they do not
retain external checkpoint-path dependencies. If the optional cache is absent
or incompatible, the same cells compute from raw EEG. Set
`OPTIONAL_CACHE = None` to disallow importing old checkpoints.

The generated local run cache is resumable. A first raw computation can take
several minutes or longer and requires approximately 0.6 GB of output storage.
One complete run is also recomputed without any cache as a numerical QC check.
Two workers bound memory usage; set `N_JOBS = 1` for lower concurrent memory.

## Exact analysis choices

- 383 runs / 22,980 original trials / 21,940 retained trials; 47 missing Status
  responses. The e30 Session 1 seven-run exception is explicit.
- Correct responses, RT at least 150 ms, one-pass mean +/-3 sample SD per
  participant/session after minimum-RT exclusion, pooling conditions and runs.
- Recorded CPz reference; 61 recorded scalp channels; no inserted CPz channel.
- The original standard_1020 template and fitted spherical Laplacian geometry
  are retained. The later supplied `chanlocs64.mat` is not substituted.
- Zero-phase 0.1–20 Hz FIR; Laplacian before power; -1 to +1.5 s epochs;
  nine 4–8 Hz frequencies and the original 3.83–4.90 wavelet-cycle schedule.
- Baseline: -300 to -100 ms, pooled retained linear power. Target: 100–500 ms.
  No voltage-baseline subtraction, amplitude rejection, ICA, or channel repair.
- The ANOVA equally averages condition-specific change scores. The descriptive
  figures and RT associations pool trial power before dB normalization. These
  are deliberately different averages and yield slightly different changes.
- Theta change is post minus pre (Session 5 minus Session 1). At the user's
  manuscript-matching request, RT improvement is pre minus post: positive values
  mean faster responses. Correlations and figures use `rt_improvement_ms`.
  The compatibility field `rt_change_ms` still means post minus pre and is its
  negative. Compared with earlier post-minus-pre RT results, Pearson r,
  Spearman rho, slopes and CI bounds reverse sign; two-sided p values are
  unchanged. Figure 3 shows BCI, mental rehearsal and pooled participants.
- Only the PDF's unadjusted ANOVA and within-group/pooled correlations are
  displayed. The broader exploratory analyses, additional ROIs, group-adjusted
  correlation, and leave-one-out analyses are omitted from this notebook.

## Outputs and verification

By default, all generated files go to
`exploratory_results/theta_paper_reproduction/`, including its `figures/`
subdirectory. Override `OUTPUT_ROOT` or `PD_THETA_OUTPUT` to use another
location. The theta–RT section of the continuous posthoc notebook reruns these
theta-change/RT-improvement correlations from the same validated caches. Figures are exported as PDF, SVG and PNG.

Section 8 writes `validation/pdf_numerical_reproduction.csv`. Saved RT
correlations and plots now use pre-minus-post improvement and match the PDF
directly, with no sign conversion in the comparison. Reference
numbers are used only for assertions, never to generate data or results.
The raw-checkpoint validation, source inventory, trial mask, library versions,
and cache hashes provide a reproducibility record. Figure rasterization may
vary with fonts or library versions; numerical results are checked at the
precision printed in the PDF.

During validation, the repository's `validate_all_files_comprehensive()` found
all 383 trigger files and all 383 analysis files valid. Its legacy run-count
checker nevertheless returned false: it expects four training runs in each of
Sessions 2–4 (96 count warnings), contrary to `experiment_structure.md`, and
also flags the documented e30 Session 1 seven-run exception. The standalone
notebook validates the documented Session 1/5 design explicitly and does not
silently treat those legacy count warnings as data corruption. The audit log
is retained alongside the reproduction validation files.

Power-table columns before baseline normalization use arbitrary stored power
units; they are not calibrated microvolts squared or power spectral density.
The normalized values are dimensionless dB ratios. Descriptive baseline and
stimulus changes have no new statistical tests. Correlations are exploratory
and do not establish causality or an intervention-independent association.

## Single-panel pooled theta–RT figure

The migrated Section 12 in the continuous notebook calls `plot_theta_rt_changes(..., pooled_only=True)`. Theta uses
post minus pre; RT improvement uses pre minus post (positive means faster).
All 32 participants are shown, with BCI orange (#DD8452)
and mental rehearsal blue (#4C72B0), following the behavioral figures. A single
gray pooled least-squares line and the two-sided Pearson result are unadjusted
for group. The same colors are used in the three-panel correlation figure.
Exports: `figures/theta_rt_association/theta_rt_change_association_pooled`
with PDF, SVG and PNG extensions under the configured `OUTPUT_ROOT`.

## Stroop theta extension

Section 13 adds congruent, incongruent and pooled-condition Stroop topographies
with the same Laplacian theta pipeline, using the current participant/session
RT exclusions. The current target is 100–300 ms with the same −300 to −100 ms
baseline; earlier 100–500 ms and 300–700 ms results are retained separately. No statistics are
run at this stage. See [stroop_theta.md](stroop_theta.md).

## Intervention-session theta trajectories

Section 14 adds Fz/FCz/Cz theta across Pre, five intervention sessions, and Post.
It uses the original distractor 100–500 ms window and retains the existing
behavior-screened assessment endpoints. All trials in available experimental
intervention recordings are included. Three panels show distractor,
no-distractor and pooled-all-trial group means ± SEM (BCI orange; mental rehearsal
blue), with equal participant weights. See [theta_intervention.md](theta_intervention.md)
for validation, normalization, documented practice exceptions and the approved
missing-e52-run policy. Section 14 is descriptive; Section 15 adds the intervention mixed ANOVA.

## Intervention theta mixed ANOVA

Section 15 tests Group × Distractor presence × Session (2 × 2 × 5) on the complete intervention-only ROI values. Group is between participants; condition and session are within participants. It uses the existing 100–500 ms estimator, all available intervention trials and equal participant weights. Greenhouse–Geisser-adjusted session terms, uncorrected results, partial η², exact inputs, and independent validation are saved under `intervention_trajectory/anova_2x2x5/`. See [theta_intervention.md](theta_intervention.md) for error strata and missing-run handling.

## Exploratory interaction follow-up

Section 16 decomposes Group × Distractor presence across intervention sessions with two paired within-group condition tests and two Welch within-condition group tests, Holm-adjusted together. It reports participant-level session averages, descriptive means, marginal 95% CIs and standardized effects, and reconstructs the original interaction in dB units. See [theta_intervention.md](theta_intervention.md).

## Current uncorrected reporting (user-requested)

Sections 15–16 now explicitly pass `correction=None` to the ANOVA and follow-up functions. Current exports are under `intervention_trajectory/anova_2x2x5_uncorrected/`; previous corrected exports remain in `anova_2x2x5/` for provenance. Earlier descriptions of GG and Holm above document the historical options, which remain supported by the functions. The current analysis uses raw ANOVA degrees of freedom (1,30 or 4,120), raw two-sided simple-effect p values, and no GG or Holm adjustment. F/t statistics, participant values, confidence intervals and effect sizes are unchanged.

Uncorrected ANOVA p values: Group .091899; condition .081981; Session .101031; Group × condition .007031; Group × Session .934526; condition × Session .071851; three-way .396181. Uncorrected follow-ups: BCI condition contrast .003187; control condition contrast .472930; group contrast for distractor .038455; group contrast for no-distractor .243580.

## Statistical audit and exploratory session trends

Section 17 independently reproduces all seven intervention ANOVA effects with fixed-participant regression models and checks exact agreement between plotted values and statistical inputs. It also reports exploratory participant slopes for distractor/no-distractor trials, with unadjusted within-group and direct between-group tests. The audit finds no implementation discrepancy; the control distractor decline is detectable within group, but the group slope contrast is uncertain. See [theta_intervention.md](theta_intervention.md).

## Behavioral response outcomes during intervention

Section 18 plots correct, incorrect and timeout percentages for pooled and separate distractor conditions over intervention sessions 1–5. All behavioral trials are included, including the 60 trials from the missing-EEG run, with participant-weighted mean ± SEM by group. See [theta_intervention.md](theta_intervention.md).

## Intervention RT trajectories

Section 19 plots RT across five intervention sessions after correct/valid ≥150 ms and one-pass participant/session ±3 sample-SD cleaning, followed by the user-requested >50% participant-loss rule. Pooled and separate condition panels use group means ± SEM; missing participant/session means remain missing and sample sizes are labeled. See [theta_intervention.md](theta_intervention.md).

## Theta Figure 1 display revision

Section 5 retains the original pooled theta values and shared ±1.5 dB scale.
Its titles now read Pre-intervention and Post-intervention. Panel a groups the
four pre/post maps; panel b groups the two post–pre maps, separated by extra
space and a thin gray divider. Only Fz, FCz, Cz, P3/P4, P5/P6, P7/P8, PO3/PO4,
PO5/PO6 and PO7/PO8 have names and dots; all other electrode dots are hidden. All 61 channels still enter the maps. The shared
plotting helper exposes these as optional display settings, preserving other
figures' existing defaults. The standalone notebook embeds the revised preview.

The 2026-11-05 rerun audit retains the requested ±1.5 dB scale and adds colorbar
overflow triangles wherever rendered scalp values exceed a displayed limit.
Both ends are marked for the assessment overview: BCI post exceeds the upper
limit, and the mental-rehearsal change map slightly exceeds the lower limit.
The saved `colorbar_extensions` and existing per-panel saturation fractions
document this display choice. Numerical maps, normalization, and tests are
unchanged. Stroop maps keep their existing unclipped shared scales.
