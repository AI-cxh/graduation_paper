#!/usr/bin/env python3
"""Apply EXP-008 selectors to VQAv2 without fitting on VQAv2 labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from haloquest_selector import (  # noqa: E402
    fit_frozen_grouped_selector,
    prepare_haloquest_selector_data,
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


def bootstrap_metric(
    values: np.ndarray,
    *,
    target: np.ndarray,
    metric: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        indices = generator.integers(0, len(values), size=len(values))
        if metric == "mean":
            estimate = float(values[indices].mean())
        elif metric == "auroc":
            sampled_target = target[indices]
            if len(np.unique(sampled_target)) != 2:
                continue
            estimate = float(roc_auc_score(sampled_target, values[indices]))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        estimates.append(estimate)
    point = (
        float(values.mean())
        if metric == "mean"
        else float(roc_auc_score(target, values))
    )
    return {
        "point_estimate": point,
        "bootstrap_95_percent_interval": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "valid_resamples": len(estimates),
    }


def subset_metrics(
    *,
    indices: np.ndarray,
    target: np.ndarray,
    effects: np.ndarray,
    native_scores: np.ndarray,
    intervention_scores: np.ndarray,
    probabilities: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    policy_scores = np.where(
        selected, intervention_scores, native_scores
    )
    harmful = effects == "harms"
    helpful = effects == "helps"
    return {
        "count": len(indices),
        "effect_counts": dict(
            sorted(Counter(effects[indices].tolist()).items())
        ),
        "auroc_help_vs_rest": (
            float(roc_auc_score(target[indices], probabilities[indices]))
            if len(np.unique(target[indices])) == 2
            else None
        ),
        "auprc_help_vs_rest": (
            float(
                average_precision_score(
                    target[indices], probabilities[indices]
                )
            )
            if target[indices].any()
            else None
        ),
        "selection_rate": float(selected[indices].mean()),
        "help_capture_rate": (
            float(selected[np.intersect1d(indices, np.flatnonzero(helpful))].mean())
            if helpful[indices].any()
            else None
        ),
        "harm_avoidance_rate": (
            float(
                (~selected[
                    np.intersect1d(indices, np.flatnonzero(harmful))
                ]).mean()
            )
            if harmful[indices].any()
            else None
        ),
        "mean_policy_vqa_soft_accuracy": float(policy_scores[indices].mean()),
        "policy_minus_always_native": float(
            (policy_scores[indices] - native_scores[indices]).mean()
        ),
        "policy_minus_always_intervention": float(
            (
                policy_scores[indices]
                - intervention_scores[indices]
            ).mean()
        ),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frozen_config = config["frozen_selector"]
    feature_sets = {
        str(name): [str(field) for field in fields]
        for name, fields in frozen_config["feature_sets"].items()
    }
    utilities = {
        str(effect): float(value)
        for effect, value in frozen_config["utility"].items()
    }
    training_records = read_jsonl(
        PROJECT_ROOT / str(frozen_config["training_features"])
    )
    expected_training = int(frozen_config["training_expected_count"])
    if len(training_records) != expected_training:
        raise ValueError(
            f"EXP-008 training feature count changed: {len(training_records)}"
        )
    training = prepare_haloquest_selector_data(
        training_records,
        feature_sets,
        utilities,
    )
    target_records = read_jsonl(
        PROJECT_ROOT / str(config["outputs"]["native_answer_features"])
    )
    expected_target = sum(
        int(value) for value in config["sample"]["quotas"].values()
    )
    if len(target_records) != expected_target:
        raise ValueError(
            f"EXP-009 target feature count changed: {len(target_records)}"
        )
    target_records.sort(key=lambda record: int(record["question_id"]))
    question_ids = np.asarray(
        [int(record["question_id"]) for record in target_records]
    )
    effects = np.asarray(
        [str(record["action_effect"]) for record in target_records],
        dtype=str,
    )
    answer_types = np.asarray(
        [str(record["answer_type"]) for record in target_records], dtype=str
    )
    target = effects == "helps"
    native_scores = np.asarray(
        [
            float(record["native_vqa_soft_accuracy"])
            for record in target_records
        ]
    )
    intervention_scores = np.asarray(
        [
            float(record["intervention_vqa_soft_accuracy"])
            for record in target_records
        ]
    )
    prediction_records = [
        {
            "question_id": int(question_ids[index]),
            "image_sha256": str(target_records[index]["image_sha256"]),
            "answer_type": str(answer_types[index]),
            "action_effect": str(effects[index]),
            "native_vqa_soft_accuracy": float(native_scores[index]),
            "intervention_vqa_soft_accuracy": float(
                intervention_scores[index]
            ),
            "models": {},
        }
        for index in range(len(target_records))
    ]
    model_results = {}
    bootstrap_resamples = int(config["evaluation"]["bootstrap_resamples"])
    seed = int(config["experiment"]["seed"])
    all_indices = np.arange(len(target_records))
    for model_index, (name, fields) in enumerate(feature_sets.items()):
        frozen = fit_frozen_grouped_selector(
            training.features[name],
            training.target,
            training.intervention_utility,
            training.groups,
            folds=int(frozen_config["inner_folds"]),
            repeats=int(
                frozen_config["repeated_oof_threshold_repeats"]
            ),
            c_value=float(frozen_config["logistic_c"]),
            seed=seed + model_index * 10_000,
        )
        target_matrix = np.asarray(
            [
                [float(record[field]) for field in fields]
                for record in target_records
            ],
            dtype=float,
        )
        probabilities = frozen.predict_probability(target_matrix)
        selected = probabilities >= frozen.threshold
        policy_scores = np.where(
            selected, intervention_scores, native_scores
        )
        metrics = subset_metrics(
            indices=all_indices,
            target=target,
            effects=effects,
            native_scores=native_scores,
            intervention_scores=intervention_scores,
            probabilities=probabilities,
            selected=selected,
        )
        metrics["auroc_bootstrap"] = bootstrap_metric(
            probabilities,
            target=target,
            metric="auroc",
            resamples=bootstrap_resamples,
            seed=seed + 700_000 + model_index * 10,
        )
        metrics["policy_minus_native_bootstrap"] = bootstrap_metric(
            policy_scores - native_scores,
            target=target,
            metric="mean",
            resamples=bootstrap_resamples,
            seed=seed + 700_001 + model_index * 10,
        )
        metrics["policy_minus_intervention_bootstrap"] = bootstrap_metric(
            policy_scores - intervention_scores,
            target=target,
            metric="mean",
            resamples=bootstrap_resamples,
            seed=seed + 700_002 + model_index * 10,
        )
        model_results[name] = {
            "feature_names": fields,
            "training": {
                "sample_count": len(training.target),
                "threshold": frozen.threshold,
                "mean_oof_auroc": float(
                    roc_auc_score(
                        training.target,
                        frozen.mean_training_oof_probability,
                    )
                ),
                "mean_oof_auprc": float(
                    average_precision_score(
                        training.target,
                        frozen.mean_training_oof_probability,
                    )
                ),
                "standardized_coefficients": {
                    field: float(value)
                    for field, value in zip(
                        fields, frozen.model.coef_[0], strict=True
                    )
                },
            },
            "vqav2": metrics,
            "by_answer_type": {
                answer_type: subset_metrics(
                    indices=np.flatnonzero(answer_types == answer_type),
                    target=target,
                    effects=effects,
                    native_scores=native_scores,
                    intervention_scores=intervention_scores,
                    probabilities=probabilities,
                    selected=selected,
                )
                for answer_type in ("yes/no", "number", "other")
            },
        }
        for index in range(len(target_records)):
            prediction_records[index]["models"][name] = {
                "probability": float(probabilities[index]),
                "threshold": frozen.threshold,
                "selected_intervention": bool(selected[index]),
                "policy_vqa_soft_accuracy": float(policy_scores[index]),
            }

    summary = {
        "protocol": "exp009-exp008-frozen-selector-transfer-v1",
        "policy_metric": "unconditionally normalized VQAv2 soft accuracy",
        "evidence_boundary": (
            "Selectors and thresholds use only EXP-008 HaloQuest training "
            "features and labels. VQAv2 action effects are evaluation-only. "
            "The policy metric normalizes every prediction and reference to "
            "exclude capitalization-only changes from action effects."
        ),
        "sample": {
            "count": len(target_records),
            "effect_counts": dict(sorted(Counter(effects.tolist()).items())),
            "answer_type_counts": dict(
                sorted(Counter(answer_types.tolist()).items())
            ),
        },
        "policy_references": {
            "always_native_vqa_soft_accuracy": float(native_scores.mean()),
            "always_intervention_vqa_soft_accuracy": float(
                intervention_scores.mean()
            ),
            "oracle_vqa_soft_accuracy": float(
                np.maximum(native_scores, intervention_scores).mean()
            ),
        },
        "frozen_training_protocol": frozen_config,
        "model_results": model_results,
    }
    predictions_path = PROJECT_ROOT / str(
        config["outputs"]["frozen_selector_predictions"]
    )
    summary_path = PROJECT_ROOT / str(
        config["outputs"]["frozen_selector_summary"]
    )
    write_jsonl(predictions_path, prediction_records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    concise = {
        "policy_references": summary["policy_references"],
        "model_results": {
            name: {
                "training_threshold": result["training"]["threshold"],
                **{
                    key: value
                    for key, value in result["vqav2"].items()
                    if key
                    in {
                        "auroc_help_vs_rest",
                        "auprc_help_vs_rest",
                        "selection_rate",
                        "help_capture_rate",
                        "harm_avoidance_rate",
                        "mean_policy_vqa_soft_accuracy",
                        "policy_minus_always_native",
                        "policy_minus_always_intervention",
                    }
                },
            }
            for name, result in model_results.items()
        },
        "summary": str(summary_path),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
