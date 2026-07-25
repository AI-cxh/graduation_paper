"""Paired rule-proxy evaluation for EXP-003 interventions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import comb
from statistics import mean
from typing import Any


def _exact_mcnemar_p_value(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(
        comb(discordant, index)
        for index in range(min(first_only, second_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def _binary_transition(
    baseline: Sequence[bool],
    intervention: Sequence[bool],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for baseline_value, intervention_value in zip(
        baseline,
        intervention,
        strict=True,
    ):
        if baseline_value and intervention_value:
            counts["both_success"] += 1
        elif baseline_value:
            counts["baseline_only_success"] += 1
        elif intervention_value:
            counts["intervention_only_success"] += 1
        else:
            counts["neither_success"] += 1
    count = len(baseline)
    return {
        "count": count,
        "baseline_success_rate": sum(baseline) / count,
        "intervention_success_rate": sum(intervention) / count,
        "intervention_minus_baseline_rate": (
            sum(intervention) - sum(baseline)
        )
        / count,
        "transition_counts": {
            key: counts[key]
            for key in (
                "both_success",
                "baseline_only_success",
                "intervention_only_success",
                "neither_success",
            )
        },
        "exact_mcnemar_p_value": _exact_mcnemar_p_value(
            counts["baseline_only_success"],
            counts["intervention_only_success"],
        ),
    }


def _mean_metric(
    records: Sequence[Mapping[str, Any]],
    field: str,
) -> float:
    return mean(float(record[field]) for record in records)


def compare_intervention_outputs(
    baseline_records: Iterable[Mapping[str, Any]],
    intervention_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare native and intervened generations on identical valid pairs."""

    baseline = {
        (int(record["image_id"]), str(record["condition"])): record
        for record in baseline_records
        if record["condition"] in {"multimodal_conflict", "multimodal_clean"}
    }
    intervention = {
        (int(record["image_id"]), str(record["condition"])): record
        for record in intervention_records
        if record["condition"] in {"multimodal_conflict", "multimodal_clean"}
    }
    if baseline.keys() != intervention.keys():
        raise ValueError("Baseline and intervention records are not pair-aligned")
    if not baseline:
        raise ValueError("No aligned intervention records supplied")

    image_ids = sorted({key[0] for key in baseline})
    conflict_baseline = [
        baseline[(image_id, "multimodal_conflict")] for image_id in image_ids
    ]
    conflict_intervention = [
        intervention[(image_id, "multimodal_conflict")] for image_id in image_ids
    ]
    clean_baseline = [
        baseline[(image_id, "multimodal_clean")] for image_id in image_ids
    ]
    clean_intervention = [
        intervention[(image_id, "multimodal_clean")] for image_id in image_ids
    ]
    rejection = _binary_transition(
        [
            record["stance"] == "reject_premise"
            for record in conflict_baseline
        ],
        [
            record["stance"] == "reject_premise"
            for record in conflict_intervention
        ],
    )
    clean_containment = _binary_transition(
        [bool(record["reference_contained"]) for record in clean_baseline],
        [bool(record["reference_contained"]) for record in clean_intervention],
    )
    by_type = {}
    conflict_types = sorted(
        {str(record["conflict_type"]) for record in conflict_baseline}
    )
    for conflict_type in conflict_types:
        selected = [
            index
            for index, record in enumerate(conflict_baseline)
            if record["conflict_type"] == conflict_type
        ]
        by_type[conflict_type] = _binary_transition(
            [
                conflict_baseline[index]["stance"] == "reject_premise"
                for index in selected
            ],
            [
                conflict_intervention[index]["stance"] == "reject_premise"
                for index in selected
            ],
        )
    return {
        "protocol": "exp003-intervention-rule-proxy-comparison-v1",
        "scope": (
            "Conflict success is explicit target-linked premise rejection; "
            "clean preservation uses lexical diagnostics. Neither is complete "
            "human semantic accuracy."
        ),
        "pair_count": len(image_ids),
        "conflict_rejection_proxy": rejection,
        "conflict_rejection_proxy_by_type": by_type,
        "clean_reference_containment_proxy": clean_containment,
        "clean_mean_metrics": {
            "baseline_token_f1": _mean_metric(clean_baseline, "token_f1"),
            "intervention_token_f1": _mean_metric(
                clean_intervention,
                "token_f1",
            ),
            "baseline_rouge_l_f1": _mean_metric(
                clean_baseline,
                "rouge_l_f1",
            ),
            "intervention_rouge_l_f1": _mean_metric(
                clean_intervention,
                "rouge_l_f1",
            ),
        },
        "action_heterogeneity_proxy": {
            "intervention_helps_conflict_count": rejection[
                "transition_counts"
            ]["intervention_only_success"],
            "intervention_harms_conflict_count": rejection[
                "transition_counts"
            ]["baseline_only_success"],
            "intervention_helps_clean_containment_count": clean_containment[
                "transition_counts"
            ]["intervention_only_success"],
            "intervention_harms_clean_containment_count": clean_containment[
                "transition_counts"
            ]["baseline_only_success"],
        },
    }


def _binary_roc_auc(
    scores: Sequence[float],
    labels: Sequence[bool],
) -> float | None:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [
        score for score, label in zip(scores, labels, strict=True) if not label
    ]
    if not positives or not negatives:
        return None
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            favorable += float(positive > negative)
            favorable += 0.5 * float(positive == negative)
    return favorable / (len(positives) * len(negatives))


def align_intervention_utility_with_features(
    baseline_records: Iterable[Mapping[str, Any]],
    intervention_records: Iterable[Mapping[str, Any]],
    reliability_features: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Align rule-proxy action utility with contribution/reliability features."""

    baseline = {
        (int(record["image_id"]), str(record["condition"])): record
        for record in baseline_records
        if record["condition"] in {"multimodal_conflict", "multimodal_clean"}
    }
    intervention = {
        (int(record["image_id"]), str(record["condition"])): record
        for record in intervention_records
        if record["condition"] in {"multimodal_conflict", "multimodal_clean"}
    }
    features = {
        int(record["image_id"]): record for record in reliability_features
    }
    if baseline.keys() != intervention.keys():
        raise ValueError("Baseline and intervention records are not pair-aligned")
    image_ids = sorted({key[0] for key in baseline})
    if set(image_ids) != set(features):
        raise ValueError("Reliability features are not aligned to intervention pairs")

    records = []
    for image_id in image_ids:
        base_conflict = baseline[(image_id, "multimodal_conflict")]
        int_conflict = intervention[(image_id, "multimodal_conflict")]
        base_clean = baseline[(image_id, "multimodal_clean")]
        int_clean = intervention[(image_id, "multimodal_clean")]
        baseline_rejects = base_conflict["stance"] == "reject_premise"
        intervention_rejects = int_conflict["stance"] == "reject_premise"
        if baseline_rejects and intervention_rejects:
            transition = "both_success"
        elif baseline_rejects:
            transition = "baseline_only_success"
        elif intervention_rejects:
            transition = "intervention_only_success"
        else:
            transition = "neither_success"
        feature = features[image_id]
        records.append(
            {
                "image_id": image_id,
                "conflict_type": str(base_conflict["conflict_type"]),
                "conflict_rule_proxy_transition": transition,
                "baseline_conflict_rejection_proxy": baseline_rejects,
                "intervention_conflict_rejection_proxy": intervention_rejects,
                "baseline_clean_reference_contained": bool(
                    base_clean["reference_contained"]
                ),
                "intervention_clean_reference_contained": bool(
                    int_clean["reference_contained"]
                ),
                "conflict_identity_visual_delta": float(
                    feature["conflict_identity_visual_delta"]
                ),
                "conflict_minimum_visual_delta": float(
                    feature["conflict_minimum_visual_delta"]
                ),
                "conflict_population_std_visual_delta": float(
                    feature["conflict_population_std_visual_delta"]
                ),
                "conflict_worst_drop_from_identity": float(
                    feature["conflict_worst_drop_from_identity"]
                ),
            }
        )

    score_definitions = {
        "negative_native_contribution": (
            "conflict_identity_visual_delta",
            -1.0,
        ),
        "negative_worst_case_contribution": (
            "conflict_minimum_visual_delta",
            -1.0,
        ),
        "instability_std": (
            "conflict_population_std_visual_delta",
            1.0,
        ),
        "worst_drop": ("conflict_worst_drop_from_identity", 1.0),
    }
    help_labels = [
        record["conflict_rule_proxy_transition"] == "intervention_only_success"
        for record in records
    ]
    harm_labels = [
        record["conflict_rule_proxy_transition"] == "baseline_only_success"
        for record in records
    ]
    feature_auc = {}
    for name, (field, direction) in score_definitions.items():
        scores = [direction * float(record[field]) for record in records]
        feature_auc[name] = {
            "auroc_for_intervention_only_success": _binary_roc_auc(
                scores,
                help_labels,
            ),
            "auroc_for_baseline_only_success": _binary_roc_auc(
                scores,
                harm_labels,
            ),
        }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(
            str(record["conflict_rule_proxy_transition"]),
            [],
        ).append(record)
    summary = {
        "scope": (
            "Exploratory association between pre-intervention contribution/"
            "stability features and rule-proxy action transitions. Directions "
            "are hypothesis-fixed; no selector is trained."
        ),
        "pair_count": len(records),
        "transition_counts": dict(
            sorted(
                Counter(
                    str(record["conflict_rule_proxy_transition"])
                    for record in records
                ).items()
            )
        ),
        "feature_means_by_transition": {
            transition: {
                field: mean(float(record[field]) for record in group)
                for field in (
                    "conflict_identity_visual_delta",
                    "conflict_minimum_visual_delta",
                    "conflict_population_std_visual_delta",
                    "conflict_worst_drop_from_identity",
                )
            }
            for transition, group in sorted(grouped.items())
        },
        "feature_auroc": feature_auc,
    }
    return records, summary
