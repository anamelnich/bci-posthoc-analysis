"""Isolated worker for one participant's longitudinal EEG preprocessing QC."""

import argparse
import io
import json
from contextlib import redirect_stdout


def main():
    """Run one participant in a fresh process and emit only JSON to stdout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()

    captured = io.StringIO()
    with redirect_stdout(captured):
        from .decoder_training import (
            build_longitudinal_evaluation_manifest,
            run_longitudinal_preprocessing_qc,
        )

        manifest_inputs = build_longitudinal_evaluation_manifest(
            subject_ids=[args.subject_id], project_root=args.project_root
        )
        qc = run_longitudinal_preprocessing_qc(manifest_inputs)

    payload = {
        "subject_qc": qc["subject_qc"].to_dict(orient="records"),
        "run_qc": qc["run_qc"].to_dict(orient="records"),
        "failures": qc["failures"].to_dict(orient="records"),
    }
    print(json.dumps(payload, default=str))


if __name__ == "__main__":
    main()
