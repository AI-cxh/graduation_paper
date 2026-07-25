#!/usr/bin/env python3
"""Record the user's explicit confirmation of selected AI pre-review rows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from review_confirmation import confirm_by_confidence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prereview",
        type=Path,
        default=PROJECT_ROOT / "annotations/exp001_smoke_ai_prereview.csv",
    )
    parser.add_argument(
        "--confidence",
        nargs="+",
        default=["low", "medium"],
    )
    parser.add_argument(
        "--note",
        default=(
            "Chen Xuhao confirmed all low/medium AI judgments after review "
            "on 2026-07-25."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.prereview.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        records = list(reader)
    updated, changed = confirm_by_confidence(
        records,
        set(args.confidence),
        args.note,
    )
    with args.prereview.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(updated)
    confirmed = sum(bool(row["human_confirmation"].strip()) for row in updated)
    print(
        f"updated={changed} confirmed={confirmed}/{len(updated)} "
        f"file={args.prereview}"
    )


if __name__ == "__main__":
    main()
