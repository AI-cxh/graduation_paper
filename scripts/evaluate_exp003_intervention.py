#!/usr/bin/env python3
"""Evaluate and compare EXP-003 intervention generations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from exp001_evaluation import evaluate_record  # noqa: E402
from intervention_evaluation import (  # noqa: E402
    align_intervention_utility_with_features,
    compare_intervention_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/predictions/exp003/"
        "qwen2_5_vl_3b_premise_verification.jsonl",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_smoke_scored.jsonl",
    )
    parser.add_argument(
        "--scored-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp003/"
        "qwen2_5_vl_3b_premise_verification_scored.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp003/"
        "qwen2_5_vl_3b_premise_verification_summary.json",
    )
    parser.add_argument(
        "--reliability-features",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp002/qwen2_5_vl_3b_reliability_features.jsonl",
    )
    parser.add_argument(
        "--utility-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp003/"
        "qwen2_5_vl_3b_intervention_utility.jsonl",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.predictions)
    evaluated = [evaluate_record(record) for record in predictions]
    selected_ids = {int(record["image_id"]) for record in evaluated}
    baseline = [
        record
        for record in read_jsonl(args.baseline)
        if int(record["image_id"]) in selected_ids
        and record["condition"] in {"multimodal_conflict", "multimodal_clean"}
    ]
    summary = compare_intervention_outputs(baseline, evaluated)
    reliability_features = [
        record
        for record in read_jsonl(args.reliability_features)
        if int(record["image_id"]) in selected_ids
    ]
    utility_records, feature_alignment = (
        align_intervention_utility_with_features(
            baseline,
            evaluated,
            reliability_features,
        )
    )
    summary["feature_alignment"] = feature_alignment
    write_jsonl(args.scored_output, evaluated)
    write_jsonl(args.utility_output, utility_records)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
