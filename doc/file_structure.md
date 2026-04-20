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

path = `/Users/hililbby/Library/CloudStorage/Box-Box/CNBI/Attention_distraction/project_healthy`

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

## Session-Level Extra Files

Some files are stored directly inside the session folder rather than inside individual run folders.

### Session 1

Session 1 contains the following extra files:

- **Demographics**
  - Filename format: `demographics_subjectID_date_time.txt`
  - Example: `demographics_subject39_20260303_090523.txt`

- **Color test**
  - Filename format: `colortest_subjectID_date_time.txt`
  - Example: `colortest_subject39_20260303_090811.txt`

- **NASA-TLX**
  - Filename format: `NASAtli_subjectID_date_time.txt`
  - Example: `NASAtli_subject36_20260207_155327.txt`

### Session 2

Session 2 also contains:

- **NASA-TLX**
  - Filename format: `NASAtli_subjectID_date_time.txt`

### Other Sessions

No other session-level extra files are currently documented here.
