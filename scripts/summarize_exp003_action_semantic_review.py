#!/usr/bin/env python3
"""Validate and summarize the paired EXP-003 action semantic review."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_semantic_review import (  # noqa: E402
    summarize_action_semantic_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=PROJECT_ROOT
        / "annotations/exp003_action_semantic_review_43.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp003/"
        "qwen2_5_vl_3b_action_semantic_review_summary.json",
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.review_csv.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    summary = summarize_action_semantic_review(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_complete and summary["status"] != "complete":
        raise SystemExit(
            "Human action review is incomplete: "
            f"{summary['human_confirmed_count']}/{summary['row_count']}"
        )


if __name__ == "__main__":
    main()

