#!/usr/bin/env python3
"""Evaluate EXP-009 with official-style VQAv2 soft accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vqav2_control import (  # noqa: E402
    evaluate_vqav2_predictions,
    pair_vqav2_actions,
    summarize_vqav2_control,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/exp009_vqav2_control.yaml",
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
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    predictions_path = PROJECT_ROOT / str(
        config["outputs"]["predictions"]
    )
    predictions = read_jsonl(predictions_path)
    expected = sum(int(value) for value in config["sample"]["quotas"].values())
    if len(predictions) != expected * len(config["inference"]["actions"]):
        raise ValueError(
            f"EXP-009 predictions incomplete: {len(predictions)}"
        )
    evaluated = evaluate_vqav2_predictions(predictions)
    pairs = pair_vqav2_actions(
        evaluated,
        effect_epsilon=float(config["evaluation"]["effect_epsilon"]),
    )
    summary = summarize_vqav2_control(
        evaluated,
        pairs,
        bootstrap_resamples=int(
            config["evaluation"]["bootstrap_resamples"]
        ),
        seed=int(config["experiment"]["seed"]),
    )
    summary["experiment"] = config["experiment"]
    summary["dataset"] = config["dataset"]
    summary["sample"] = config["sample"]
    summary["model"] = config["model"]
    summary["inference"] = config["inference"]
    summary["runtime"] = {
        "total_generation_seconds": sum(
            float(record["seconds_per_item"]) for record in predictions
        ),
        "mean_generation_seconds": sum(
            float(record["seconds_per_item"]) for record in predictions
        )
        / len(predictions),
        "peak_memory_bytes": max(
            int(record["peak_memory_bytes"]) for record in predictions
        ),
    }
    evaluated_path = PROJECT_ROOT / str(
        config["outputs"]["evaluated_records"]
    )
    pairs_path = PROJECT_ROOT / str(
        config["outputs"]["paired_results"]
    )
    summary_path = PROJECT_ROOT / str(config["outputs"]["summary"])
    write_jsonl(evaluated_path, evaluated)
    write_jsonl(pairs_path, pairs)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
