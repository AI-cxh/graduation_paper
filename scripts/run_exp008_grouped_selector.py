#!/usr/bin/env python3
"""Run grouped and cross-scope selector validation for EXP-008."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from haloquest_selector import (  # noqa: E402
    prepare_haloquest_selector_data,
    run_exp008,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/exp008_grouped_selector.yaml",
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


def _validate_counts(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    expected_count = int(config["input"]["expected_sample_count"])
    if len(records) != expected_count:
        raise ValueError(
            f"EXP-008 sample count changed: {len(records)} != {expected_count}"
        )
    for field, expected_key, value_key in (
        ("action_effect", "expected_effect_counts", "effect"),
        ("dataset_scope", "expected_scope_counts", "scope"),
    ):
        expected = {
            str(key): int(value)
            for key, value in config["input"][expected_key].items()
        }
        actual = {
            value: sum(str(record[field]) == value for record in records)
            for value in expected
        }
        if actual != expected:
            raise ValueError(
                f"EXP-008 {value_key} counts changed: {actual} != {expected}"
            )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records = read_jsonl(PROJECT_ROOT / str(config["input"]["features"]))
    _validate_counts(records, config)
    feature_sets = {
        str(name): [str(field) for field in fields]
        for name, fields in config["feature_sets"].items()
    }
    effect_utilities = {
        str(effect): float(value)
        for effect, value in config["target"]["policy_utility"].items()
    }
    data = prepare_haloquest_selector_data(
        records,
        feature_sets,
        effect_utilities,
    )
    validation = config["validation"]
    criteria = config["go_no_go"]
    summary, predictions = run_exp008(
        data,
        feature_sets,
        scopes=[str(scope) for scope in validation["scopes"]],
        transfer_directions=config["cross_scope_transfer"]["directions"],
        outer_folds=int(validation["outer_folds"]),
        repeats=int(validation["repeats"]),
        inner_folds=int(validation["inner_folds"]),
        c_value=float(validation["logistic_c"]),
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
    summary["feature_sets"] = feature_sets
    summary["cross_scope_protocol"] = config["cross_scope_transfer"]
    summary_path = PROJECT_ROOT / str(config["outputs"]["summary"])
    predictions_path = PROJECT_ROOT / str(config["outputs"]["predictions"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(predictions_path, predictions)

    concise = {
        "sample": summary["sample"],
        "within_scope": {
            scope: {
                "policy_references": result["policy_references"],
                "models": {
                    name: {
                        metric: model["repeat_metric_distribution"][metric][
                            "mean"
                        ]
                        for metric in (
                            "auroc",
                            "auprc",
                            "policy_utility",
                            "selection_rate",
                            "help_capture_rate",
                            "harm_avoidance_rate",
                        )
                    }
                    for name, model in result["feature_set_results"].items()
                },
            }
            for scope, result in summary["within_scope"].items()
        },
        "cross_scope_transfer": {
            direction: {
                name: result["metrics"]
                for name, result in models.items()
            }
            for direction, models in summary["cross_scope_transfer"].items()
        },
        "go_no_go": summary["go_no_go"],
        "summary_output": str(summary_path),
        "predictions_output": str(predictions_path),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
