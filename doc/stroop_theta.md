> Consolidated on 2026-11-05: use the **Consolidated midfrontal theta, Stroop theta, and intervention behavior** section of the continuous notebook. Original subsection numbers below identify the migrated steps; alpha sections 9–11 were removed.

# Stroop Laplacian theta topographies

Section 13 of `notebooks/posthoc_analysis.ipynb` extends the distractor-task
pipeline to Stroop. Run the notebook path/import cell first; this section does
not require running the alpha sections. It uses the validated distractor theta
settings in `CACHE_DIR` and writes a separate `OUTPUT_ROOT/stroop_100_300ms/` tree.
No ROI tests, ANOVA, or theta–RT correlations are performed at this stage.

## Files, events and behavioral exclusions

- 32 configured participants, Sessions 1 and 5, two non-practice Stroop runs
  per session: 128 paired GDF/behoutput files and 7,680 trials.
- Use the existing comprehensive Stroop behavioral validator and loader; check
  trial numbering 1–60, all fields/types/ranges, word–ink congruency, 30 trials
  per condition, correct/incorrect/timeout coding, and RTs in milliseconds.
- GDF Status events alternate fixation 4 and stimulus 8 (congruent) or 32
  (incongruent). Every condition is checked against the corresponding behavior
  row. Two observed code-1 markers bracket the task and are not trial anchors.
- The task-specific mapping is verified in
  [Stroop.py at bdf2a9f](https://github.com/anamelnich/distractor-classification/blob/bdf2a9ff03119b94826d9865eff5e0e4190836ad/visualInterface/Stroop.py).
  The stimulus marker follows `pygame.display.flip()`. No unmeasured display
  latency is corrected. There is no response trigger used for Stroop RT;
  behavioral RT is the first recognized word onset relative to audio recording.
- Exclude timeouts and incorrect responses first. Among remaining correct
  trials, compute one mean and sample SD (`ddof=1`) per participant/session,
  pooling both runs and both congruencies. Exclude strictly outside ±3 SD in
  one pass; no minimum RT cutoff. Threshold equality is retained.
- The current data retain 7,073 trials: 438 timeouts, 57 incorrect responses,
  112 session-pooled SD outliers. The mask and all 64 thresholds were checked
  against the current primary checkout's `summarize_stroop_reaction_time()`.
  The older worktree `stroop_audit.py` uses per-run trimming (7,082 retained),
  so it must not supply this analysis's eligibility mask.

## EEG pipeline and normalization

Reuse the exact distractor theta settings (checksum-validated): 512 Hz,
continuous zero-phase 0.1–20 Hz FIR, signal-level spherical-spline Laplacian
(lambda 1e-5, stiffness 4, 50 terms), and the same 61 recorded scalp channels,
standard_1020 montage and template sphere. Retain FCz; do not insert CPz.
Epoch from −1 to +1.5 s around the congruency stimulus marker. Check actual
recording padding, finite EEG, exact event ordering and full trial alignment.
No ICA, amplitude rejection, voltage baseline subtraction or extra exclusion.

Compute single-trial zero-mean Morlet power at 4:0.5:8 Hz, using the same
frequency-dependent cycle schedule as distractor theta. The baseline centers
are −300 to −100 ms (102 samples), and the target centers are 100–300 ms
(102 samples: 101.5625–298.828125 ms at 512 Hz). Check wavelet boundary energy against the existing 1e-8 limit.
The measured fixation-to-stimulus interval is 502–510 ms. The nominal baseline
is within fixation, but low-frequency wavelet support extends into adjacent
activity; this is not an isolated fixation-period power estimate. Stroop is a
vocal task, and this unchanged pipeline does not remove vocal/other artifacts.

Pool retained baseline linear power across both conditions and both runs
within participant/session/channel/frequency. Each condition and the overall
pooled estimate share that baseline. Pool stimulus linear power over the
relevant trials, divide by the common baseline, apply `10*log10`, and then
average target times and frequencies. Overall pooled theta is computed from
pooled linear power, not by averaging congruent and incongruent dB maps.
Average participants equally within each group; change is within-participant
Session 5 minus Session 1. Whole-scalp maps come before selecting an ROI for
any later inferential analysis. Fz/FCz/Cz remain visible in the labeled montage.

## Notebook workflow and outputs

Reusable implementation: `code/src/posthoc_analysis/stroop_theta.py`.

1. `prepare_stroop_theta_inputs`: comprehensive paired-file preflight, current
   RT mask, thresholds, exclusions, inventory, and separate Stroop settings.
2. `rewindow_stroop_theta_cache`: validate the original 100–500 ms cache and
   select 100–300 ms from its linear-power arrays. The original
   `build_stroop_theta_cache` remains available for windows requiring new EEG
   extraction. Invalid/incomplete input halts map generation.
3. `summarize_stroop_theta`: check every checkpoint's hash, axes, shape,
   retained trial IDs/counts and positivity, then write participant maps,
   pooled baselines and condition counts.
4. `plot_stroop_theta_topographies`: one six-panel page per congruency and one
   for pooled conditions. Rows: BCI and mental rehearsal (16 participants each).
   Columns: pre, post, post − pre. Reuse the existing regularized spherical
   display (1e-5; display only) and all 61 electrode labels. Use shared symmetric
   pre/post limits and shared symmetric change limits across all conditions,
   selected to include all rendered and electrode values without clipping.

`stroop_100_300ms/cache/` holds source/QC tables, settings, run checkpoints and summaries.
`stroop_100_300ms/figures/` holds individual PDF/SVG/PNG exports, plotted group values,
figure/QC manifests and `stroop_theta_congruent_incongruent_pooled.pdf`.
These retained run-power checkpoints are expensive-to-recreate inputs for
future ROI work; summary tables are kept alongside this self-contained workflow
under `exploratory_results`, not added to the repository-local `analyses/` tree.

The notebook explicitly passes `target_s=(0.1, 0.3)` to input preparation.
The reusable function retains its original 100–500 ms default for compatibility.
Original 100–500 ms checkpoints and figures remain under `OUTPUT_ROOT/stroop/`.
The current window is derived using `rewindow_stroop_theta_cache`: all source
hashes, preprocessing settings, masks, axes and power arrays are validated before
selecting the exact 102 time samples from the original linear-power checkpoints.
Normalization and participant/group summaries are recalculated for 100–300 ms.
Earlier 300–700 ms results remain under `OUTPUT_ROOT/stroop_300_700ms/`.
Window sample axes and figure labels are derived from the saved settings.

## Incongruent minus congruent contrast page

The fourth page of the combined PDF displays the within-participant difference
between incongruent and congruent baseline-normalized theta (dB) at 100–300 ms.
Compute the difference separately for each participant/session/channel, then
average participants equally within group. Columns show pre, post, and the
within-participant post-minus-pre change in the condition contrast. Both groups
and all three columns share one symmetric, unclipped contrast scale; the first
three pages retain their existing shared condition scales. The contrast is a
difference of normalized dB values, not a subtraction of linear power before a
log transform. Save the paired participant values and plotted group values as
CSV alongside PDF/SVG/PNG exports. No additional exclusions or statistics.
