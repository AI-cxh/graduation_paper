"""Deterministic diagnostics for HaloQuest non-false-premise controls."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Any

from exp001_evaluation import (
    classify_conflict_response,
    lexical_metrics,
    normalize_text,
)


PROTOCOL_VERSION = "exp006-haloquest-control-local-diagnostics-v1"


def split_reference_variants(reference: str) -> list[str]:
    """Split HaloQuest's semicolon-delimited acceptable responses."""

    variants = [part.strip() for part in str(reference).split(";") if part.strip()]
    return variants or [str(reference).strip()]


def evaluate_control_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Attach best-over-reference lexical diagnostics and rejection stance."""

    required = {
        "haloquest_id",
        "action",
        "hallucination_type",
        "groundtruth_response",
        "generated_text",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"HaloQuest control prediction missing: {sorted(missing)}")
    prediction = str(record["generated_text"]).strip()
    variants = split_reference_variants(str(record["groundtruth_response"]))
    metrics = [lexical_metrics(reference, prediction) for reference in variants]
    stance = classify_conflict_response(
        str(record["groundtruth_response"]),
        prediction,
    )
    result = dict(record)
    result.update(
        {
            "control_evaluation_protocol": PROTOCOL_VERSION,
            "reference_variants": variants,
            "best_normalized_exact_match": any(
                bool(metric["normalized_exact_match"]) for metric in metrics
            ),
            "best_reference_contained": any(
                bool(metric["reference_contained"]) for metric in metrics
            ),
            "best_token_f1": max(float(metric["token_f1"]) for metric in metrics),
            "best_rouge_l_f1": max(
                float(metric["rouge_l_f1"]) for metric in metrics
            ),
            "rejection_stance": stance["stance"],
            "rejection_rule": stance["stance_rule"],
            "rejection_evidence": stance["stance_evidence"],
        }
    )
    return result


def pair_control_actions(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Align action outputs and expose lexical-change audit candidates."""

    by_id: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        haloquest_id = int(record["haloquest_id"])
        action = str(record["action"])
        if action in by_id.setdefault(haloquest_id, {}):
            raise ValueError(f"Duplicate control action: {(haloquest_id, action)}")
        by_id[haloquest_id][action] = record

    expected = {"native_prompt", "premise_verification"}
    paired = []
    for haloquest_id, actions in sorted(by_id.items()):
        if set(actions) != expected:
            raise ValueError(
                f"Incomplete control actions for {haloquest_id}: {sorted(actions)}"
            )
        native = actions["native_prompt"]
        intervention = actions["premise_verification"]
        response_similarity = lexical_metrics(
            str(native["generated_text"]),
            str(intervention["generated_text"]),
        )["token_f1"]
        lexical_delta = float(intervention["best_token_f1"]) - float(
            native["best_token_f1"]
        )
        native_rejects = native["rejection_stance"] == "reject_premise"
        intervention_rejects = (
            intervention["rejection_stance"] == "reject_premise"
        )
        new_rejection = bool(intervention_rejects and not native_rejects)
        audit_candidate = bool(
            response_similarity < 0.8
            or abs(lexical_delta) >= 0.1
            or new_rejection
        )
        paired.append(
            {
                "haloquest_id": haloquest_id,
                "image_type": str(native["image_type"]),
                "hallucination_type": str(native["hallucination_type"]),
                "question": str(native["question"]),
                "groundtruth_response": str(native["groundtruth_response"]),
                "native_response": str(native["generated_text"]),
                "intervention_response": str(intervention["generated_text"]),
                "native_best_token_f1": float(native["best_token_f1"]),
                "intervention_best_token_f1": float(
                    intervention["best_token_f1"]
                ),
                "lexical_delta": lexical_delta,
                "response_token_f1": float(response_similarity),
                "normalized_response_equal": (
                    normalize_text(str(native["generated_text"]))
                    == normalize_text(str(intervention["generated_text"]))
                ),
                "native_rejection_stance": str(native["rejection_stance"]),
                "intervention_rejection_stance": str(
                    intervention["rejection_stance"]
                ),
                "new_target_linked_rejection": new_rejection,
                "audit_candidate": audit_candidate,
            }
        )
    return paired


def _category_summary(
    records: list[Mapping[str, Any]],
    pairs: list[Mapping[str, Any]],
    category: str,
) -> dict[str, Any]:
    selected_records = [
        record for record in records if record["hallucination_type"] == category
    ]
    selected_pairs = [
        pair for pair in pairs if pair["hallucination_type"] == category
    ]
    by_action = {
        action: [record for record in selected_records if record["action"] == action]
        for action in ("native_prompt", "premise_verification")
    }
    lexical_wins = Counter(
        (
            "intervention"
            if pair["lexical_delta"] > 0.1
            else "native"
            if pair["lexical_delta"] < -0.1
            else "tie"
        )
        for pair in selected_pairs
    )

    def average(values: Iterable[float | bool]) -> float | None:
        values = list(values)
        return mean(values) if values else None

    return {
        "question_count": len(selected_pairs),
        "actions": {
            action: {
                "mean_best_token_f1": average(
                    float(record["best_token_f1"])
                    for record in action_records
                ),
                "mean_best_rouge_l_f1": average(
                    float(record["best_rouge_l_f1"])
                    for record in action_records
                ),
                "exact_match_rate": average(
                    bool(record["best_normalized_exact_match"])
                    for record in action_records
                ),
            }
            for action, action_records in by_action.items()
        },
        "lexical_effect_counts_margin_0_1": dict(sorted(lexical_wins.items())),
        "normalized_response_equal_count": sum(
            bool(pair["normalized_response_equal"]) for pair in selected_pairs
        ),
        "new_target_linked_rejection_count": sum(
            bool(pair["new_target_linked_rejection"]) for pair in selected_pairs
        ),
        "audit_candidate_count": sum(
            bool(pair["audit_candidate"]) for pair in selected_pairs
        ),
    }


def summarize_control_evaluation(
    records: Iterable[Mapping[str, Any]],
    pairs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize diagnostics while preserving their non-semantic status."""

    records = list(records)
    pairs = list(pairs)
    return {
        "protocol": PROTOCOL_VERSION,
        "official_haloquest_autoeval_run": False,
        "evidence_boundary": (
            "Best-over-reference lexical metrics and response-change flags are "
            "diagnostics, not semantic accuracy or action utility."
        ),
        "question_count": len(pairs),
        "generation_count": len(records),
        "normalized_response_equal_count": sum(
            bool(pair["normalized_response_equal"]) for pair in pairs
        ),
        "new_target_linked_rejection_count": sum(
            bool(pair["new_target_linked_rejection"]) for pair in pairs
        ),
        "audit_candidate_count": sum(
            bool(pair["audit_candidate"]) for pair in pairs
        ),
        "by_hallucination_type": {
            category: _category_summary(records, pairs, category)
            for category in ("visual challenge", "insufficient context")
        },
    }
