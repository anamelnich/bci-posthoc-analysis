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
- Training `.triggers.txt` times are **milliseconds** from
  `pygame.time.get_ticks()`. They are not EEG sample indices.
- GDF `Status` events use the recording's sample clock (verify **512 Hz**).
  Use GDF Status samples to compute primary training RT and slice EEG epochs.
- To convert a GDF Status sample to time:
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

Each trial logs **3 trigger requests** in sequence (hardware delivery is
separate and must be checked against GDF Status):

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

### EEG epoching anchor

For EEG epoching, ERP plotting, Pd measurement, and decoder-style event-locked
feature extraction, the trial event of interest is the **stimulus presentation**
trigger, not the fixation trigger.

- Epoch to trigger `8`, `32`, or `44`
- Do **not** epoch to fixation trigger `4` unless explicitly analyzing fixation
- Treat time 0 as stimulus onset / condition onset

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

Primary training RT is reconstructed from non-practice `*_training.gdf` files:

```python
events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose="ERROR")
assert raw.info["sfreq"] == 512
```

- Stimulus onset is code **8, 32, or 44**, never fixation code 4.
- For each stimulus, search for code 64 strictly after that sample and strictly
  before the next stimulus (recording end for the final trial).
- Exactly one response: `rt_ms = (response_sample - stimulus_sample) / sfreq * 1000`.
- The GDF extractor returns NaN without a response. Multiple responses flag
  an alignment error and remain NaN. Consolidation preserves the extractor's
  result in `gdf_rt_ms` before applying the explicit exception below.
- Align GDF, `.triggers.txt`, and `.analysis.txt` by subject/session/run/trial.
  Require 60 stimuli, counts **30 code 8 / 15 code 32 / 15 code 44**, identical
  stimulus order, and agreement with analysis task/distractor position.
- Correct responses are `feedback == 1`.

Text-only RT is `response_time - stimulus_time`, already in **milliseconds**.
`compute_training_reaction_times_gdf()` supplies the GDF-only training RT and
retains text-only RT in `text_rt_ms`. Consolidation then calls
`apply_training_text_rt_fallback()`: absent GDF responses with aligned text
RT in [0, 50) ms receive that value in analysis `rt_ms`, labeled
`rt_source = trigger_text_fallback`. The GDF-only `gdf_rt_ms`, `rt_samples`,
and `response_sample` stay NaN. This explicit exception covers the audited
47 trigger-lockout cases and allows them to contribute to <150 ms reporting.
No ambiguous response or missing-file/alignment failure receives a fallback;
other absent responses remain NaN. The obsolete
standalone text-RT and single-run 2-SD preview helpers have been removed.

The former calculation divided text milliseconds by 512 and multiplied by
1000, inflating RT by **1.953125**. Its `rt_samples` label was incorrect.
Example: GDF 3813→4081 gives **523.4375 ms**; text difference **522 ms**;
legacy saved RT **1019.53125 ms**. Rebuild trial outputs and rerun cleaning and
summaries, rather than rescaling cleaned averages. Stroop `.behoutput.txt`
verbal-response RTs are already milliseconds and are unchanged.

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

This grouping applies to the text log. Do not group GDF events in fixed
triplets: a logged response request can be rejected by the hardware driver.
The training audit found 47 such absent responses, all with text RT below
50 ms; see [training_response_qc.md](training_response_qc.md).

For stimulus-locked EEG epoching, the required Status-channel anchors are the
60 stimulus events (`8`, `32`, or `44`). If a response event (`64`) is missing
but all stimulus events are present and each stimulus follows fixation (`4`),
epoching may continue with a warning because the response event is not used as
time 0.

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

GDF Status is the timing source for EEG and primary training RT. Text timing
is a separate millisecond clock; Stroop RT comes from `.behoutput.txt`.
