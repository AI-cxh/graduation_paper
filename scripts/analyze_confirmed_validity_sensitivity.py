#!/usr/bin/env python3
"""Recompute EXP-001/002 diagnostics after confirmed validity exclusions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from candidate_scoring import (  # noqa: E402
    align_scores_with_free_generation,
    summarize_reference_scores,
)
from confirmed_sensitivity import confirmed_dataset_exclusions  # noqa: E402
from exp001_evaluation import summarize_records  # noqa: E402
from reliability_features import summarize_reliability_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prereview",
        type=Path,
        default=PROJECT_ROOT / "annotations/exp001_smoke_ai_prereview.csv",
    )
    parser.add_argument(
        "--generation",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_smoke_scored.jsonl",
    )
    parser.add_argument(
        "--candidate-scores",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_reference_scores.jsonl",
    )
    parser.add_argument(
        "--reliability-features",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp002/qwen2_5_vl_3b_reliability_features.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/sensitivity/"
        "confirmed_validity_exclusion_summary.json",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _headline(
    generation_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    reliability_summary: dict[str, Any],
) -> dict[str, Any]:
    paired = generation_summary["paired_conflict_stance"]
    candidate = candidate_summary["paired_deltas"][
        "conflict_visual_delta_mean_logprob"
    ]
    feature_auc = reliability_summary["rule_proxy_alignment"]["feature_auroc"]
    return {
        "pair_count": paired["pair_count"],
        "multimodal_explicit_rejection_rate": paired[
            "multimodal_explicit_target_rejection_rate"
        ],
        "text_only_explicit_rejection_rate": paired[
            "text_only_explicit_target_rejection_rate"
        ],
        "paired_rejection_rate_difference": paired[
            "multimodal_minus_text_only_rejection_rate"
        ],
        "exact_mcnemar_p_value": paired["exact_mcnemar_p_value"],
        "mean_conflict_visual_delta": candidate["mean"],
        "positive_conflict_visual_delta_count": candidate["positive_count"],
        "candidate_exact_sign_test_p_value": candidate[
            "exact_sign_test_p_value"
        ],
        "auroc_contribution_for_multimodal_rejection": feature_auc[
            "contribution_identity"
        ]["multimodal_explicit_rejection"],
        "auroc_worst_case_contribution_for_multimodal_rejection": feature_auc[
            "worst_case_contribution"
        ]["multimodal_explicit_rejection"],
        "auroc_negative_instability_for_multimodal_rejection": feature_auc[
            "negative_instability_std"
        ]["multimodal_explicit_rejection"],
    }


def _summaries(
    generation: list[dict[str, Any]],
    candidate_scores: list[dict[str, Any]],
    reliability_features: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    generation_summary = summarize_records(generation)
    candidate_pairs, candidate_summary = summarize_reference_scores(
        candidate_scores
    )
    candidate_summary["free_generation_alignment"] = (
        align_scores_with_free_generation(candidate_pairs, generation)
    )
    reliability_summary = summarize_reliability_features(
        reliability_features,
        generation,
    )
    return generation_summary, candidate_summary, reliability_summary


def main() -> None:
    args = parse_args()
    with args.prereview.open(encoding="utf-8", newline="") as handle:
        prereview = list(csv.DictReader(handle))
    exclusions = confirmed_dataset_exclusions(prereview)
    generation = read_jsonl(args.generation)
    candidate_scores = read_jsonl(args.candidate_scores)
    reliability_features = read_jsonl(args.reliability_features)

    full = _summaries(generation, candidate_scores, reliability_features)
    filtered_generation = [
        record
        for record in generation
        if int(record["image_id"]) not in exclusions
    ]
    filtered_candidate_scores = [
        record
        for record in candidate_scores
        if int(record["image_id"]) not in exclusions
    ]
    filtered_reliability = [
        record
        for record in reliability_features
        if int(record["image_id"]) not in exclusions
    ]
    filtered = _summaries(
        filtered_generation,
        filtered_candidate_scores,
        filtered_reliability,
    )
    summary = {
        "protocol": "confirmed-validity-exclusion-sensitivity-v1",
        "scope": (
            "Sensitivity analysis excluding only dataset-validity issues "
            "explicitly confirmed by the user. Outcome labels remain "
            "rule-based proxies, not complete human semantic accuracy."
        ),
        "excluded_pair_count": len(exclusions),
        "excluded_image_ids": list(exclusions),
        "exclusion_reasons": {
            str(image_id): note for image_id, note in exclusions.items()
        },
        "full_50_pair_headline": _headline(*full),
        "filtered_headline": _headline(*filtered),
        "filtered_generation_summary": filtered[0],
        "filtered_candidate_summary": filtered[1],
        "filtered_reliability_summary": filtered[2],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "excluded_pair_count": len(exclusions),
                "included_pair_count": 50 - len(exclusions),
                "full": summary["full_50_pair_headline"],
                "filtered": summary["filtered_headline"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
