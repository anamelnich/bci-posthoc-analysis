# File Contents

## When to use this file
Use this file when:
- parsing any data file
- understanding columns, fields, and structures
- reconstructing trial-level or feature-level data

Do NOT use this file for:
- locating files on disk (see file_structure.md)
- interpreting trigger semantics (see triggers.md)
- working with decoder `.mat` files (see decoder_mat.md)

---

## 1. EEG Files (.gdf)

Each `.gdf` file contains:

### Signal
- Matrix: **n_samples × n_channels**

### Channels (typically 67 total)
- Channels 1–64: BioSemi EEG-style channel block, but this block includes
  non-scalp-EEG channels:
  - `M1`, `M2`: mastoid/reference channels
  - `EOG`: eye channel
- Channels 65–66: additional EOG/sensor channels, typically `sens7` and `sens8`
- Channel 67: trigger channel (`Status`)

### Analysis EEG channel convention
- For EEG plotting and downstream EEG analyses, exclude non-scalp-EEG channels:
  `M1`, `M2`, `EOG`, `sens7`, and `sens8`.
- The resulting analysis EEG channel set should contain scalp electrodes only.
- Keep the `Status` channel for trigger/event validation, but never include it in
  EEG signal analyses.

### Header fields
- `SampleRate`
- `Label` (channel labels)

### Important note
- All trigger events are present in the **Status channel (channel 67)**
- For EEG epoching in training/decoding, use stimulus-presentation Status
  triggers `8`, `32`, and `44` as time 0. Do not use fixation trigger `4` as
  the default trial-start anchor for ERP/Pd analyses.
- If a response Status event (`64`) is missing but all 60 stimulus Status events
  are present and valid, stimulus-locked EEG epoching can continue with a clear
  warning because response events are not the epoch anchor.

---

## 2. Training / Decoding: analysis.txt

### Structure
- No column headers
- Exactly **60 rows (60 trials)**

### Columns

1. **trial index**
2. **task**
   - 1 = distractor
   - 0 = no distractor
3. **feedback**
   - 1 = correct
   - 2 = incorrect
   - 3 = timeout
4. **target position**
   - values: 1, 2, 3, 4
5. **distractor position**
   - values: 0, 2, 4
6. **dot side**
   - 1 = right
   - 0 = left
7. **intertrial interval**
   - units: milliseconds
8. **BCI output**
   - training: always 99
   - decoding:
     - 1 = correct
     - 0 = incorrect
     - 3 = ambivalent

---

## 3. Training / Decoding: triggers.txt

### Structure
- **180 rows** (3 triggers × 60 trials)

### Columns
1. trial number
2. trigger type
3. time (in samples)

### Notes
- Same trigger information is also present in `.gdf` Status channel
- Timing values are in **samples (not milliseconds)**

---

## 4. Stroop: behoutput.txt

### Structure
- Includes column headers
- Trial count:
  - 60 trials (stroop)
  - 24 trials (stroop_practice)

### Columns

- trial number
- trial type (string)
  - congruent / incongruent
- stimulus (string)
  - blue, green, yellow, red
- ink color (string)
- response
  - 1 = correct
  - 2 = incorrect
  - 3 = timeout
- reaction time
  - units: milliseconds

---

## 5. EOG Calibration: triggers.txt

### Structure
- **80 rows** (40 trials × 2 triggers)

### Columns
1. trial number
2. trigger type
3. time (in samples)

### Notes
- No response trigger
- Same triggers are present in `.gdf` Status channel

---

## 6. Online Info Files

### Threshold Logs (`*_thrlog.mat`)

- Struct array (one per session)
- Fields:
  - subjectID
  - timestamp
  - margin
  - thrR, thrL, thrN

---

### Online Posteriors (`*_OnlinePosteriors_*.mat`)

### Shape
- 420 × 3 (6 runs + practice)
- 540 × 3 (8 runs + practice)

### Notes
- First 60 rows = practice run (need to be dropped and excluded from analyses)
- Exception: `e27` Session 1 has 300 rows total (5 runs including practice). For BCI
  online-posterior consolidation, keep all 300 rows and treat all 5 runs as
  experimental runs.
- Exception: `e42` Session 2 has 544 rows total. Drop the final 4 rows, then
  remove the leading 60-row practice run normally.
- Exception: `e43` Session 5 has 422 rows total. Drop the final 2 rows, then
  remove the leading 60-row practice run normally.
- Exception: `e46` Session 4 has 512 rows total because the practice run has
  32 trials. Drop only the first 32 rows, then treat the remaining 480 rows as
  8 experimental runs.

### Columns

1. posterior probability
   - probability of distractor
2. threshold used
3. classification
   - 1 = no distractor
   - 2 = distractor
   - 3 = ambivalent

## 7. Units and Timing

- Trigger timing values are in **samples**, not milliseconds.
- Behavioral reaction times (e.g., in Stroop files) are in **milliseconds**.
- To convert samples to time:
  - `time (seconds) = samples / fsamp`
