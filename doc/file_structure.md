# File Structure

## When to use this file
Use this file when:
- locating data for a subject, session, or run
- understanding how data are organized on disk
- identifying where specific task data are stored

Do NOT use this file for:
- interpreting file contents (see file_contents.md)
- understanding trigger meanings (see triggers.md)

---

## Base Directory

All data are stored under:

Box/CNBI/Attention_distraction/project_healthy

The structure follows:

Subject → Session → Task/Run → Files

---

## Subject Level

Each subject has a dedicated folder:

- Format: `subjectID`
- Example: `e39`

---

## Session Level

Each subject folder contains **5 session folders**, one per experimental day.

- Format: `subjectID_date`
- Example: `e39_20260303`

---

## Task / Run Level

Inside each session folder are subfolders for each run:

- Format: `subjectID_datetime_tasktype`
- Example: `e39_20260303092652_stroop`

### Task Types

- stroop_practice
- stroop
- EOGcalibration
- training_practice
- training
- decoding_practice
- decoding

If a task has multiple runs within a session, multiple folders exist with different timestamps.

---

## Files Within Run Folders

Each run folder contains files sharing the same base name as the folder.

Possible file types:

- `.gdf` → EEG recording
- `.log` → not used
- `.analysis.txt` → behavioral/task data (training/decoding)
- `.triggers.txt` → trigger timing information
- `.behoutput.txt` → Stroop task behavioral output

Not all task types include all file types.

---

## Session-Level Extra Files

Some sessions contain additional files directly inside the session folder (not inside run folders).

### Session 1

- demographics file  
- color test file  
- NASA-TLX file  

### Session 2

- NASA-TLX file  

---

## Decoder Files

Located in:

Box/CNBI/Attention_distraction/project_healthy/decoders

Each subject has 3 decoder files:

- `subject_decoderR.mat` → right distractor
- `subject_decoderL.mat` → left distractor
- `subject_decoderN.mat` → no distractor

---

## Online Information Folder

Located at:

project_healthy/online_info

Contains:

- threshold logs (`*_thrlog.mat`)
- online posterior files (`*_OnlinePosteriors_*.mat`)

---

## Session and Run Distribution

### Session 1
- stroop_practice: 1 run
- stroop: 2 runs
- EOGcalibration: 1 run
- training_practice: 1 run
- training: 8 runs
- decoding_practice: 1 run
- decoding: 6 runs

### Sessions 2–4
- EOGcalibration: 1 run
- decoding_practice: 1 run
- decoding: 8 runs

### Session 5
- EOGcalibration: 1 run
- decoding_practice: 1 run
- decoding: 6 runs
- training: 4 runs
- stroop_practice: 1 run
- stroop: 2 runs
