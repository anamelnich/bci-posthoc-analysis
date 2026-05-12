# Experiment Structure

## When to use this file
Use this file when:
- understanding the overall experimental design
- determining how many sessions, runs, and tasks exist
- identifying which tasks occur in each session
- calculating expected number of runs or trials per session
- planning analyses that depend on session structure (e.g., early vs late sessions, training vs decoding phases)

Do NOT use this file for:
- locating files on disk (see file_structure.md)
- parsing file contents or column meanings (see file_contents.md)
- interpreting trigger codes or computing timing (see triggers.md)

## Overview

The experiment consists of **5 in-person sessions per subject**, conducted across multiple days. Each session contains a fixed sequence of tasks, with specific numbers of runs per task. Each run consists of a fixed number of trials depending on the task type.

---

## Trial Counts by Task

- **Training / Decoding runs**:  
  - 60 trials per run

- **Stroop task**:  
  - 60 trials per run

- **Stroop practice**:  
  - 24 trials per run

- **EOG calibration**:  
  - 40 trials per run

---

## Session-by-Session Structure

### Session 1

- 1 × Stroop Practice (24 trials)
- 2 × Stroop (60 trials each)
- 1 × EOG Calibration (40 trials)
- 1 × Training Practice (60 trials)
- 8 × Training (60 trials each)
- 1 × Decoding Practice (60 trials)
- 6 × Decoding (60 trials each)

---

### Sessions 2, 3, and 4 (identical structure)

Each of these sessions includes:

- 1 × EOG Calibration (40 trials)
- 1 × Decoding Practice (60 trials)
- 8 × Decoding (60 trials each)

---

### Session 5

- 1 × EOG Calibration (40 trials)
- 1 × Decoding Practice (60 trials)
- 6 × Decoding (60 trials each)
- 4 × Training (60 trials each)
- 1 × Stroop Practice (24 trials)
- 2 × Stroop (60 trials each)

---

## Summary Across All Sessions

- **Total sessions**: 5

- **Training runs**:
  - Session 1: 8 runs
  - Session 5: 4 runs  
  → **Total: 12 runs**

- **Decoding runs**:
  - Session 1: 6 runs
  - Sessions 2–4: 8 runs each (24 total)
  - Session 5: 6 runs  
  → **Total: 36 runs**

- **EOG Calibration runs**:
  - 1 per session  
  → **Total: 5 runs**

- **Stroop runs**:
  - Session 1: 2 runs
  - Session 5: 2 runs  
  → **Total: 4 runs**

- **Practice runs**:
  - Present in multiple sessions for Stroop, Training, and Decoding
  - Always precede the corresponding main task

---

## Key Structural Notes

- Each **run is independent** and consists of a fixed number of trials.
- Practice runs should be excluded by default from all analyses.
- Sessions 2–4 focus exclusively on **decoding**, while Sessions 1 and 5 include a broader set of tasks (training, Stroop, calibration).

## Subject-Specific Exceptions

- `e30` Session 1 has only **7 non-practice training runs** instead of the
  expected 8.
  - For session-level EEG condition averages, report the mismatch and average
    across the 7 available run-level averages.
  - Keep `n_runs` and `expected_n_runs` in output manifests so the incomplete
    session is explicit in downstream analyses.
- `e27` Session 1 completed only **5 decoding runs total, including practice**.
  - For BCI online-posterior consolidation, retain all 5 runs and treat them as experimental runs.
  - Do not remove a practice run from `e27` Session 1 online posterior data.
  - Align threshold-log rows to these 5 retained runs before writing trial-level BCI CSVs.
- `e42` Session 2 has malformed online-info tail records.
  - The online posterior file has 544 rows: 9 complete 60-trial runs plus 4 extra trailing rows.
  - Drop the final 4 online-posterior rows, then remove the leading practice run normally.
  - The threshold log has 10 rows for Session 2 instead of the expected 9 rows.
  - Drop the 10th threshold-log row for Session 2, then remove the leading practice/log row normally.
- `e43` Session 5 has malformed online-info tail records.
  - The online posterior file has 422 rows: 7 complete 60-trial runs plus 2 extra trailing rows.
  - Drop the final 2 online-posterior rows, then remove the leading practice run normally.
  - The threshold log has 8 rows for Session 5 instead of the expected 7 rows.
  - Drop the 8th threshold-log row for Session 5, then remove the leading practice/log row normally.
- `e46` Session 4 has a shortened decoding practice run in the online posterior file.
  - The practice run has only 32 trials, so the online posterior file has 512 rows.
  - Drop only the first 32 online-posterior rows, then treat the remaining 480 rows as the 8 experimental runs.
  - This exception does not change threshold-log handling; remove the leading Session 4 threshold-log practice row normally.
