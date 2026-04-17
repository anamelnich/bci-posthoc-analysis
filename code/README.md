# CNBI Attention-Distraction Posthoc Analysis (Python)

This repo now implements a Python pipeline tailored to the provided `project_healthy` data layout:

- `project_healthy/eXX/eXX_YYYYMMDD/eXX_<datetime>_<task>/`
- EEG in `.gdf` (`samples x channels`, channel 67 status trigger)
- task files (`analysis.txt`, `triggers.txt`, `.behoutput.txt`)

## What is implemented

- Run discovery by subject/session/task folder naming
- GDF loading + 0.1-20 Hz filtering + epoching around stimulus triggers
- Baseline correction via epoch settings
- Posterior ROI + left-right lateralized features in the 0.2-0.5s interval
- xDAWN helper function for decoder parity workflows
- R² top-k (default 30) feature ranking
- LDA one-vs-rest decoding and offline cross-validation
- Permutation testing of multiclass accuracy
- Notebook walkthrough for module-by-module analysis

## Main entry point

```bash
python scripts/run_posthoc_analysis.py \
  --project-root /path/to/Box/CNBI/Attention_distraction/project_healthy \
  --output-dir outputs \
  --n-perm 200
```

Outputs:
- `run_manifest.csv`
- `offline_predictions.csv`
- `metrics.json`
- `perm_hist.png`

## Notebook

Open: `notebooks/posthoc_analysis_walkthrough.ipynb`
