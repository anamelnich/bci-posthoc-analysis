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
- Channels 1–64: EEG
- Channels 65–66: EOG
- Channel 67: trigger channel (`Status`)

### Header fields
- `SampleRate`
- `Label` (channel labels)

### Important note
- All trigger events are present in the **Status channel (channel 67)**

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

- Struct array (one per run)
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