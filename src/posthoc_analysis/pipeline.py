"""End-to-end project_healthy pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import discover_runs
from .preprocessing import EpochConfig, make_epochs_from_gdf, posterior_and_lateralized_features, read_gdf_array
from .modeling import cross_validated_side_decoding
from .stats import permutation_test_multiclass_accuracy
from .viz import plot_permutation_histogram


def run_project_pipeline(project_root: str | Path, output_dir: str | Path, n_perm: int = 200) -> dict[str, object]:
    """Run offline decoding analysis across all discovered decoding/training runs.

    Data assumptions from provided spec
    -----------------------------------
    - EEG GDF is samples x channels with status in channel 67.
    - Decoding labels come from stimulus trigger code: 8(no), 32(right), 44(left).
    - Practice variants can be excluded by task naming convention.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(project_root)
    selected = [r for r in runs if r.task in {"decoding", "training"} and r.gdf_path is not None]

    rows = []
    feature_blocks = []
    label_blocks = []

    for run in selected:
        epochs, labels, times = make_epochs_from_gdf(run.gdf_path, EpochConfig())
        _, _, ch_names = read_gdf_array(run.gdf_path)
        feats = posterior_and_lateralized_features(epochs, ch_names[: epochs.shape[1]], times)

        # Enforce documented 60-trial convention when extra events are present.
        if feats.shape[0] >= 60:
            feats = feats[:60]
            labels = labels[:60]

        feature_blocks.append(feats)
        label_blocks.append(labels)
        rows.append(
            {
                "subject_id": run.subject_id,
                "session_id": run.session_id,
                "run_id": run.run_id,
                "task": run.task,
                "n_trials": int(len(labels)),
                "n_features": int(feats.shape[1]),
            }
        )

    if not feature_blocks:
        raise RuntimeError("No decoding/training runs with .gdf files found")

    X = np.vstack(feature_blocks)
    y = np.concatenate(label_blocks)

    cv = cross_validated_side_decoding(X, y)
    perm = permutation_test_multiclass_accuracy(X, y, n_perm=n_perm)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "run_manifest.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "trial_index": np.arange(len(y)),
            "true_class": y,
            "predicted_class": cv["predicted_class"],
            "p_no": cv["posteriors"][:, 0],
            "p_right": cv["posteriors"][:, 1],
            "p_left": cv["posteriors"][:, 2],
        }
    )
    pred_df.to_csv(output_dir / "offline_predictions.csv", index=False)

    metrics = {
        "offline_cv_accuracy": float(cv["accuracy"]),
        "permutation_observed_accuracy": float(perm["observed_accuracy"]),
        "permutation_p_value": float(perm["p_value"]),
        "n_trials": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_runs": int(len(selected)),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig = plot_permutation_histogram(metrics["permutation_observed_accuracy"], perm["null_distribution"], output_dir / "perm_hist.png")

    return {
        "metrics": metrics,
        "manifest": manifest,
        "predictions": pred_df,
        "figure_path": str(fig),
    }
