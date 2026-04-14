#!/usr/bin/env python3
"""Run posthoc analysis on project_healthy directory tree."""

from __future__ import annotations

import argparse
import json

from posthoc_analysis import run_project_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", required=True, help="Path like Box/.../project_healthy")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--n-perm", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_project_pipeline(args.project_root, args.output_dir, n_perm=args.n_perm)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
