#!/usr/bin/env python3
"""Run the leakage-controlled EXP-004 selector comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from go_no_go import prepare_selector_data, run_go_no_go  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/exp004_go_no_go.yaml",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp004/qwen2_5_vl_3b_go_no_go_summary.json",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp004/qwen2_5_vl_3b_go_no_go_oof.jsonl",
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
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    input_path = PROJECT_ROOT / config["input"]["utility_records"]
    records = read_jsonl(input_path)
    expected_count = int(config["input"]["expected_pair_count"])
    if len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} records, found {len(records)}"
        )

    feature_sets = config["feature_sets"]
    data = prepare_selector_data(records, feature_sets)
    validation = config["validation"]
    criteria = config["go_no_go"]
    summary, predictions = run_go_no_go(
        data,
        feature_sets,
        outer_folds=int(validation["outer_folds"]),
        repeats=int(validation["repeats"]),
        inner_folds=int(validation["inner_folds"]),
        c_grid=[float(value) for value in validation["logistic_c_grid"]],
        bootstrap_resamples=int(validation["bootstrap_resamples"]),
        seed=int(config["experiment"]["seed"]),
        primary_feature_set=str(criteria["primary_feature_set"]),
        comparator_feature_set=str(criteria["comparator_feature_set"]),
        minimum_mean_auroc_gain=float(criteria["minimum_mean_auroc_gain"]),
        minimum_mean_policy_gain_over_always_intervention=float(
            criteria["minimum_mean_policy_gain_over_always_intervention"]
        ),
        minimum_repeat_auroc_win_fraction=float(
            criteria["minimum_repeat_auroc_win_fraction"]
        ),
    )
    summary["experiment"] = config["experiment"]
    summary["target"] = config["target"]
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.predictions_output, predictions)
    concise = {
        "sample": summary["sample"],
        "policy_references": summary["policy_references"],
        "feature_set_results": {
            name: {
                "mean_auroc": result["repeat_metric_distribution"]["auroc"][
                    "mean"
                ],
                "mean_auprc": result["repeat_metric_distribution"]["auprc"][
                    "mean"
                ],
                "mean_policy_utility": result[
                    "repeat_metric_distribution"
                ]["policy_utility"]["mean"],
                "mean_selection_rate": result[
                    "repeat_metric_distribution"
                ]["selection_rate"]["mean"],
                "mean_benefit_capture": result[
                    "repeat_metric_distribution"
                ]["intervention_benefit_capture_rate"]["mean"],
                "mean_native_only_protection": result[
                    "repeat_metric_distribution"
                ]["native_only_protection_rate"]["mean"],
            }
            for name, result in summary["feature_set_results"].items()
        },
        "go_no_go": summary["go_no_go"],
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

