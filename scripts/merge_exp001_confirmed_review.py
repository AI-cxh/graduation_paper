#!/usr/bin/env python3
"""Merge explicitly confirmed pre-review rows into the formal review table."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from review_confirmation import merge_confirmed_prereview  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prereview",
        type=Path,
        default=PROJECT_ROOT / "annotations/exp001_smoke_ai_prereview.csv",
    )
    parser.add_argument(
        "--formal-review",
        type=Path,
        default=PROJECT_ROOT / "annotations/exp001_smoke_semantic_review.csv",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    args = parse_args()
    fields, formal = read_csv(args.formal_review)
    _, prereview = read_csv(args.prereview)
    merged, changed = merge_confirmed_prereview(formal, prereview)
    with args.formal_review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged)
    complete = sum(
        bool(row["human_semantic_correct"] and row["human_stance"])
        for row in merged
    )
    print(
        f"merged={changed} complete={complete}/{len(merged)} "
        f"file={args.formal_review}"
    )


if __name__ == "__main__":
    main()
