# Triggers

## When to use this file
Use this file when:
- interpreting trigger codes in `.triggers.txt` or EEG Status channel
- reconstructing trial structure from events
- computing reaction time (RT)
- determining distractor presence and side

Do NOT use this file for:
- locating files on disk (see file_structure.md)
- parsing column formats (see file_contents.md)

---

## 1. General Notes

- Triggers are stored in:
  - `.triggers.txt` files
  - EEG `.gdf` files in the **Status channel (channel 67)**
- Timing values are in **samples**, not milliseconds
- To convert to time:  
  `time (seconds) = samples / fsamp`

---

## 2. Trigger Codes (Core Set)

These codes are used in training and decoding tasks:

- **4** → fixation
- **8** → no distractor
- **32** → distractor on the right
- **44** → distractor on the left
- **64** → response

---

## 3. Trial Structure: Training and Decoding

Each trial consists of **3 triggers** in sequence:

1. **Fixation**
   - trigger = 4

2. **Stimulus / Condition**
   - 8 → no distractor
   - 32 → distractor right
   - 44 → distractor left

3. **Response**
   - trigger = 64

So each trial has:

`[4 → (8 | 32 | 44) → 64]`

---

## 4. Determining Trial Type

From the second trigger:

- **8** → no-distractor trial
- **32 or 44** → distractor trial

---

## 5. Determining Distractor Side

Only relevant for distractor trials:

- **32** → distractor on the **right**
- **44** → distractor on the **left**

---

## 6. Computing Reaction Time (RT)

RT is computed using trigger timestamps:

- Identify:
  - stimulus trigger (8, 32, or 44)
  - response trigger (64)

- Compute:

`RT (samples) = time_response - time_stimulus`

- Convert to milliseconds:

`RT (ms) = (RT_samples / fsamp) * 1000`

---

## 7. Relationship to analysis.txt

For training/decoding:

- `task` column:
  - 1 = distractor → corresponds to triggers 32 or 44
  - 0 = no distractor → corresponds to trigger 8

- `dot side`:
  - 1 = right
  - 0 = left  
  (independent of distractor side)

---

## 8. EOG Calibration Triggers

EOG calibration uses a different mapping.

Each trial has **2 triggers**:

1. fixation (4)
2. dot position:

- **8** → top
- **32** → right
- **44** → bottom
- **64** → left

### Notes
- There is **no response trigger**
- Each run has **40 trials → 80 triggers total**

---

## 9. Stroop Task

- Stroop tasks **do not use triggers.txt**
- Behavioral timing (including RT) is provided directly in:
  - `.behoutput.txt`

---

## 10. Practical Parsing Rules

### Training / Decoding

- Every **3 consecutive triggers = 1 trial**
- Pattern must follow:
  - 4 → (8 / 32 / 44) → 64

If this pattern breaks, data may be corrupted or misaligned.

---

### EOG Calibration

- Every **2 consecutive triggers = 1 trial**
- Pattern:
  - 4 → (8 / 32 / 44 / 64)

---

## 11. Sequence and Trial-Number Validation

Parsing code should validate trigger files using both the **trial number column** and the **expected trigger sequence**. Do not assume that grouping every 3 consecutive rows (or every 2 consecutive rows for EOG) is sufficient without checking trial-number consistency.

### Training / Decoding
For each trial number, exactly 3 trigger rows are expected, in this order:

1. `4` = fixation  
2. `8`, `32`, or `44` = stimulus / condition  
3. `64` = response  

Expected sequence per trial:

`[4 → (8 | 32 | 44) → 64]`

### EOG Calibration
For each trial number, exactly 2 trigger rows are expected, in this order:

1. `4` = fixation  
2. `8`, `32`, `44`, or `64` = dot position  

Expected sequence per trial:

`[4 → (8 | 32 | 44 | 64)]`

### Required checks
Code should verify all of the following:
- rows for a given trial share the same trial number
- trial numbers increase in the expected order
- each trial has the expected number of trigger rows
- trigger codes follow the correct task-specific sequence

If these checks fail, the file should be treated as misaligned, incomplete, or corrupted rather than parsed silently.

---

## 12. Summary

Triggers encode:

- trial structure
- condition (distractor vs no distractor)
- distractor side
- timing for RT computation

They are the **ground truth timing source** for all event-based analyses.
