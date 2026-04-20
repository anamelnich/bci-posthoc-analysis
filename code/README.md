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
