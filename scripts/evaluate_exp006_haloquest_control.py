#!/usr/bin/env python3
"""Evaluate EXP-006 control outputs with deterministic diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from haloquest_control_evaluation import (  # noqa: E402
    evaluate_control_record,
    pair_control_actions,
    summarize_control_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/predictions/exp006/"
        "qwen2_5_vl_3b_native_vs_verification.jsonl",
    )
    parser.add_argument(
        "--scored-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp006/"
        "qwen2_5_vl_3b_native_vs_verification_scored.jsonl",
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp006/"
        "qwen2_5_vl_3b_control_action_pairs.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp006/"
        "qwen2_5_vl_3b_control_summary.json",
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
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.predictions)
    scored = [evaluate_control_record(record) for record in predictions]
    pairs = pair_control_actions(scored)
    summary = summarize_control_evaluation(scored, pairs)
    write_jsonl(args.scored_output, scored)
    write_jsonl(args.paired_output, pairs)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"scored_output={args.scored_output}")
    print(f"paired_output={args.paired_output}")
    print(f"summary_output={args.summary_output}")


if __name__ == "__main__":
    main()

