#!/usr/bin/env python3
"""Run EXP-004 with AI-adjudicated semantic action utility."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_semantic_review import validate_action_review_records  # noqa: E402
from go_no_go import prepare_selector_data, run_go_no_go  # noqa: E402


NATIVE_UTILITY_FIELD = "native_ai_semantic_utility"
INTERVENTION_UTILITY_FIELD = "intervention_ai_semantic_utility"
TARGET_NAME = "intervention_only_ai_semantic_success"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/exp004_ai_semantic_sensitivity.yaml",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp004/"
        "qwen2_5_vl_3b_ai_semantic_sensitivity_summary.json",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp004/"
        "qwen2_5_vl_3b_ai_semantic_sensitivity_oof.jsonl",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_semantic_records(
    utility_records: list[dict[str, Any]],
    review_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Align features and keep only predeclared, semantically decided rows."""

    validated_review = validate_action_review_records(review_records)
    review = {int(record["image_id"]): record for record in validated_review}
    utility = {int(record["image_id"]): record for record in utility_records}
    if review.keys() != utility.keys():
        raise ValueError("Review and utility image IDs are not aligned")
    kept = []
    excluded = []
    for image_id in sorted(review):
        labels = review[image_id]
        native = str(labels["ai_native_conflict_correct"]).strip().lower()
        intervention = str(
            labels["ai_intervention_conflict_correct"]
        ).strip().lower()
        if native not in {"yes", "no"} or intervention not in {"yes", "no"}:
            excluded.append(
                {
                    "image_id": image_id,
                    "native_label": native,
                    "intervention_label": intervention,
                    "reason": "uncertain_or_blank_conflict_semantic_label",
                }
            )
            continue
        record = dict(utility[image_id])
        record[NATIVE_UTILITY_FIELD] = native == "yes"
        record[INTERVENTION_UTILITY_FIELD] = intervention == "yes"
        kept.append(record)
    return kept, excluded


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    utility_path = PROJECT_ROOT / config["input"]["utility_records"]
    review_path = PROJECT_ROOT / config["input"]["action_review"]
    utility = read_jsonl(utility_path)
    review = read_csv(review_path)
    expected = int(config["input"]["expected_input_count"])
    if len(utility) != expected or len(review) != expected:
        raise ValueError(
            f"Expected {expected} aligned inputs, found "
            f"utility={len(utility)} review={len(review)}"
        )
    records, excluded = build_semantic_records(utility, review)
    feature_sets = config["feature_sets"]
    data = prepare_selector_data(
        records,
        feature_sets,
        native_utility_field=NATIVE_UTILITY_FIELD,
        intervention_utility_field=INTERVENTION_UTILITY_FIELD,
    )
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
        target_name=TARGET_NAME,
        protocol="exp004-ai-semantic-repeated-nested-cv-v1",
        scope=(
            "Sensitivity analysis using complete AI pre-review labels, not "
            "human annotations. Uncertain conflict-semantic rows are excluded "
            "by a frozen rule before modeling. Bootstrap intervals remain "
            "descriptive for this small, model-selected sample."
        ),
        oracle_reference_name="oracle_ai_semantic_utility",
        native_utility_output_name=NATIVE_UTILITY_FIELD,
        intervention_utility_output_name=INTERVENTION_UTILITY_FIELD,
    )
    summary["experiment"] = config["experiment"]
    summary["adjudication"] = {
        **config["adjudication"],
        "input_count": expected,
        "included_count": len(records),
        "coverage": len(records) / expected,
        "excluded_count": len(excluded),
        "excluded_records": excluded,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.predictions_output, predictions)
    concise = {
        "sample": summary["sample"],
        "adjudication": summary["adjudication"],
        "policy_references": summary["policy_references"],
        "feature_set_results": {
            name: {
                metric: result["repeat_metric_distribution"][metric]["mean"]
                for metric in ("auroc", "auprc", "policy_utility")
            }
            for name, result in summary["feature_set_results"].items()
        },
        "go_no_go": summary["go_no_go"],
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
