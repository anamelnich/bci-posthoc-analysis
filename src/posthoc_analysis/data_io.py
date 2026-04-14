"""I/O helpers for the CNBI attention-distraction project folder layout.

This module is intentionally notebook-friendly: each function can be called in
isolation and returns plain Python objects, pandas tables, or NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TASK_NAMES = {
    "stroop",
    "stroop_practice",
    "eogcalibration",
    "training",
    "training_practice",
    "decoding",
    "decoding_practice",
}


@dataclass(frozen=True)
class RunPaths:
    """Resolved paths for a single run folder.

    Attributes
    ----------
    subject_id : str
        Subject identifier (e.g., ``e39``).
    session_id : str
        Session folder name (e.g., ``e39_20260303``).
    run_id : str
        Run folder name (e.g., ``e39_20260303_120102_decoding``).
    task : str
        Task name parsed from ``run_id``.
    gdf_path : Path | None
        EEG file path if found.
    analysis_txt : Path | None
        Behavior table for training/decoding runs.
    triggers_txt : Path | None
        Trigger timestamps/codes text file if found.
    behoutput_txt : Path | None
        Stroop behavior file if found.
    """

    subject_id: str
    session_id: str
    run_id: str
    task: str
    gdf_path: Path | None
    analysis_txt: Path | None
    triggers_txt: Path | None
    behoutput_txt: Path | None


def discover_runs(project_root: str | Path) -> list[RunPaths]:
    """Discover all runs under ``project_healthy`` style folder tree.

    Expected tree
    -------------
    ``root/eXX/eXX_YYYYMMDD/eXX_<datetime>_<task>/``
    """

    root = Path(project_root)
    runs: list[RunPaths] = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("e")):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
                task = _infer_task_from_run_name(run_dir.name)
                gdf_files = sorted(run_dir.glob("*.gdf"))
                runs.append(
                    RunPaths(
                        subject_id=subject_dir.name,
                        session_id=session_dir.name,
                        run_id=run_dir.name,
                        task=task,
                        gdf_path=gdf_files[0] if gdf_files else None,
                        analysis_txt=_first_existing(run_dir, ["analysis.txt"]),
                        triggers_txt=_first_existing(run_dir, ["triggers.txt"]),
                        behoutput_txt=_first_existing(run_dir, [".behoutput.txt", "behoutput.txt"]),
                    )
                )
    return runs


def read_analysis_txt(path: str | Path) -> pd.DataFrame:
    """Read 60-row analysis table for training/decoding.

    Output columns
    --------------
    ``trial_idx, trial_type, feedback, target_pos, distractor_pos, dot_side,
    iti_ms, bci_output``.
    """

    cols = [
        "trial_idx",
        "trial_type",
        "feedback",
        "target_pos",
        "distractor_pos",
        "dot_side",
        "iti_ms",
        "bci_output",
    ]
    df = pd.read_csv(path, header=None, names=cols)
    return df


def read_triggers_txt(path: str | Path) -> pd.DataFrame:
    """Read trigger file with flexible delimiter handling.

    Returns a table with at least ``sample`` and ``code`` when parseable.
    """

    df = pd.read_csv(path, header=None, sep=r"\s+|,|;|\t", engine="python")
    if df.shape[1] >= 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["sample", "code"]
    elif df.shape[1] == 1:
        df.columns = ["code"]
        df["sample"] = np.arange(len(df))
        df = df[["sample", "code"]]
    else:
        raise ValueError(f"Unable to parse triggers file: {path}")
    return df


def read_stroop_behoutput(path: str | Path) -> pd.DataFrame:
    """Read stroop behavior file with headers."""

    return pd.read_csv(path, sep=r"\s+|,|;|\t", engine="python")


def drop_practice_trials(df: pd.DataFrame, n_drop: int = 60) -> pd.DataFrame:
    """Drop first ``n_drop`` rows used by practice blocks."""

    return df.iloc[n_drop:].reset_index(drop=True)


def _first_existing(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _infer_task_from_run_name(run_name: str) -> str:
    parts = run_name.split("_")
    for k in (2, 1):
        if len(parts) >= k:
            candidate = "_".join(parts[-k:])
            if candidate.lower() in TASK_NAMES:
                return candidate.lower()
    # If naming differs slightly from documented examples, keep suffix.
    return parts[-1].lower()
