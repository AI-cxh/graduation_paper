#!/usr/bin/env python3
"""Validate and summarize completed EXP-001 human semantic labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_review import summarize_semantic_review  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=PROJECT_ROOT / "annotations/exp001_smoke_semantic_review.csv",
    )
    parser.add_argument(
        "--candidate-pairs",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_reference_pair_deltas.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_human_review_summary.json",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit with an error if any required human field is blank.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    with args.review_csv.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    candidate_pairs = (
        read_jsonl(args.candidate_pairs)
        if args.candidate_pairs.is_file()
        else None
    )
    summary = summarize_semantic_review(records, candidate_pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_complete and summary["status"] != "complete":
        raise SystemExit(
            f"Human review is incomplete: "
            f"{summary['fully_labeled_row_count']}/{summary['row_count']} rows"
        )


if __name__ == "__main__":
    main()
