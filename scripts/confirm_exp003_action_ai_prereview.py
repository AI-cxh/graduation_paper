#!/usr/bin/env python3
"""Record explicit user acceptance of selected action pre-review rows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_semantic_review import (  # noqa: E402
    confirm_action_prereview_by_confidence,
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
        "--confidence",
        nargs="+",
        default=["low", "medium"],
    )
    parser.add_argument("--note", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.review_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        records = list(reader)
    updated, changed = confirm_action_prereview_by_confidence(
        records,
        set(args.confidence),
        args.note,
    )
    with args.review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(updated)
    confirmed = sum(
        bool(row["human_confirmation"].strip()) for row in updated
    )
    print(
        f"updated={changed} confirmed={confirmed}/{len(updated)} "
        f"file={args.review_csv}"
    )


if __name__ == "__main__":
    main()

