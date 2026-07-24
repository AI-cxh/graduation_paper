#!/usr/bin/env python3
"""Evaluate EXP-001 predictions with deterministic, auditable diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from exp001_evaluation import evaluate_record, summarize_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/predictions/exp001/qwen2_5_vl_3b_smoke.jsonl",
    )
    parser.add_argument(
        "--scored-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_smoke_scored.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_smoke_summary.json",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=PROJECT_ROOT / "annotations/exp001_smoke_semantic_review.csv",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No prediction records found in {path}")
    keys = [(int(row["image_id"]), str(row["condition"])) for row in records]
    duplicates = [key for key, count in _counts(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate prediction keys: {duplicates[:10]}")
    return records


def _counts(items: list[tuple[int, str]]) -> dict[tuple[int, str], int]:
    result: dict[tuple[int, str], int] = {}
    for item in items:
        result[item] = result.get(item, 0) + 1
    return result


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "image_id",
        "condition",
        "conflict_type",
        "question",
        "reference_answer",
        "generated_text",
        "stance",
        "stance_rule",
        "stance_evidence",
        "reference_content_overlap",
        "normalized_exact_match",
        "reference_contained",
        "token_f1",
        "rouge_l_f1",
        "manual_review_required",
        "manual_review_reason",
        "human_semantic_correct",
        "human_stance",
        "human_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in fields}
            row["reference_content_overlap"] = "|".join(
                record["reference_content_overlap"]
            )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.predictions)
    evaluated = [evaluate_record(record) for record in predictions]
    summary = summarize_records(evaluated)

    write_jsonl(args.scored_output, evaluated)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_review_csv(args.review_output, evaluated)

    print(
        json.dumps(
            {
                "predictions": len(predictions),
                "review_required": summary["overall"]["manual_review_count"],
                "scored_output": str(args.scored_output),
                "summary_output": str(args.summary_output),
                "review_output": str(args.review_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
