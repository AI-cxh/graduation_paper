"""Validation and summaries for the EXP-001 human semantic review."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean
from typing import Any


SEMANTIC_LABELS = {"yes", "no", "uncertain", "n/a"}
STANCE_LABELS = {
    "reject_premise",
    "abstain",
    "answer_premise",
    "other",
    "n/a",
}
CONDITIONS = {
    "multimodal_conflict",
    "text_only_conflict",
    "multimodal_clean",
}


def _label(record: Mapping[str, Any], field: str) -> str:
    return str(record.get(field, "")).strip().lower()


def validate_review_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize labels and reject invalid or internally inconsistent rows."""

    normalized = []
    seen: set[tuple[int, str]] = set()
    for row_number, source in enumerate(records, start=2):
        record = dict(source)
        try:
            image_id = int(record["image_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Row {row_number}: invalid image_id") from error
        condition = str(record.get("condition", "")).strip()
        if condition not in CONDITIONS:
            raise ValueError(
                f"Row {row_number}: unexpected condition {condition!r}"
            )
        key = (image_id, condition)
        if key in seen:
            raise ValueError(f"Row {row_number}: duplicate key {key}")
        seen.add(key)

        semantic = _label(record, "human_semantic_correct")
        stance = _label(record, "human_stance")
        if semantic and semantic not in SEMANTIC_LABELS:
            raise ValueError(
                f"Row {row_number}: invalid human_semantic_correct={semantic!r}"
            )
        if stance and stance not in STANCE_LABELS:
            raise ValueError(
                f"Row {row_number}: invalid human_stance={stance!r}"
            )
        if condition == "text_only_conflict" and semantic not in {"", "n/a"}:
            raise ValueError(
                f"Row {row_number}: text_only_conflict semantic label must be n/a"
            )
        if condition == "multimodal_clean" and stance not in {"", "n/a"}:
            raise ValueError(
                f"Row {row_number}: multimodal_clean stance must be n/a"
            )
        if condition == "multimodal_conflict":
            if semantic == "n/a":
                raise ValueError(
                    f"Row {row_number}: multimodal_conflict semantic label "
                    "cannot be n/a"
                )
            if stance == "n/a":
                raise ValueError(
                    f"Row {row_number}: multimodal_conflict stance cannot be n/a"
                )
        if condition == "text_only_conflict" and stance == "n/a":
            raise ValueError(
                f"Row {row_number}: text_only_conflict stance cannot be n/a"
            )
        if condition == "multimodal_clean" and semantic == "n/a":
            raise ValueError(
                f"Row {row_number}: multimodal_clean semantic label cannot be n/a"
            )

        record["image_id"] = image_id
        record["condition"] = condition
        record["human_semantic_correct"] = semantic
        record["human_stance"] = stance
        normalized.append(record)
    if not normalized:
        raise ValueError("Review table is empty")
    return normalized


def _required_fields_complete(record: Mapping[str, Any]) -> bool:
    return bool(
        _label(record, "human_semantic_correct")
        and _label(record, "human_stance")
    )


def _safe_binary_metrics(
    predicted: Sequence[bool],
    actual: Sequence[bool],
) -> dict[str, Any]:
    true_positive = sum(p and a for p, a in zip(predicted, actual, strict=True))
    false_positive = sum(p and not a for p, a in zip(predicted, actual, strict=True))
    false_negative = sum(
        not p and a for p, a in zip(predicted, actual, strict=True)
    )
    true_negative = len(actual) - true_positive - false_positive - false_negative
    return {
        "count": len(actual),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        ),
        "recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        ),
        "accuracy": (
            (true_positive + true_negative) / len(actual) if actual else None
        ),
    }


def _binary_roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
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


def _semantic_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for condition in ("multimodal_conflict", "multimodal_clean"):
        labels = [
            _label(record, "human_semantic_correct")
            for record in records
            if record["condition"] == condition
            and _label(record, "human_semantic_correct")
        ]
        counts = Counter(labels)
        decided = counts["yes"] + counts["no"]
        result[condition] = {
            "labeled_count": len(labels),
            "label_counts": dict(sorted(counts.items())),
            "decided_count": decided,
            "semantic_correct_rate_excluding_uncertain": (
                counts["yes"] / decided if decided else None
            ),
            "uncertain_rate_among_labeled": (
                counts["uncertain"] / len(labels) if labels else None
            ),
        }
    return result


def _human_stance_transitions(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lookup = {
        (int(record["image_id"]), str(record["condition"])): record
        for record in records
    }
    counts: Counter[str] = Counter()
    for image_id in sorted({key[0] for key in lookup}):
        multimodal = lookup.get((image_id, "multimodal_conflict"))
        text_only = lookup.get((image_id, "text_only_conflict"))
        if not multimodal or not text_only:
            continue
        multimodal_stance = _label(multimodal, "human_stance")
        text_only_stance = _label(text_only, "human_stance")
        if not multimodal_stance or not text_only_stance:
            continue
        multimodal_rejects = multimodal_stance == "reject_premise"
        text_only_rejects = text_only_stance == "reject_premise"
        if multimodal_rejects and text_only_rejects:
            counts["both_reject"] += 1
        elif multimodal_rejects:
            counts["multimodal_only_reject"] += 1
        elif text_only_rejects:
            counts["text_only_only_reject"] += 1
        else:
            counts["neither_reject"] += 1
    return {
        "complete_pair_count": sum(counts.values()),
        "transition_counts": dict(sorted(counts.items())),
    }


def _candidate_alignment(
    records: Sequence[Mapping[str, Any]],
    candidate_pairs: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if candidate_pairs is None:
        return {"available": False, "reason": "candidate_pair_file_not_supplied"}
    pair_lookup = {
        int(pair["image_id"]): pair for pair in candidate_pairs
    }
    multimodal = [
        record
        for record in records
        if record["condition"] == "multimodal_conflict"
        and int(record["image_id"]) in pair_lookup
    ]
    result: dict[str, Any] = {
        "available": True,
        "matched_multimodal_conflict_rows": len(multimodal),
    }
    definitions = {
        "human_semantic_correct": {"yes"},
        "human_reject_premise": {"reject_premise"},
    }
    for name, positive_labels in definitions.items():
        field = (
            "human_semantic_correct"
            if name == "human_semantic_correct"
            else "human_stance"
        )
        eligible = []
        for record in multimodal:
            label = _label(record, field)
            if field == "human_semantic_correct" and label not in {"yes", "no"}:
                continue
            if field == "human_stance" and not label:
                continue
            eligible.append(record)
        scores = [
            float(
                pair_lookup[int(record["image_id"])][
                    "conflict_visual_delta_mean_logprob"
                ]
            )
            for record in eligible
        ]
        labels = [_label(record, field) in positive_labels for record in eligible]
        result[name] = {
            "count": len(eligible),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "auroc": _binary_roc_auc(scores, labels),
            "mean_score_positive": (
                mean(
                    score
                    for score, label in zip(scores, labels, strict=True)
                    if label
                )
                if any(labels)
                else None
            ),
            "mean_score_negative": (
                mean(
                    score
                    for score, label in zip(scores, labels, strict=True)
                    if not label
                )
                if labels and not all(labels)
                else None
            ),
        }
    return result


def summarize_semantic_review(
    records: Iterable[Mapping[str, Any]],
    candidate_pairs: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate the review table and summarize completed human labels."""

    records = validate_review_records(records)
    condition_counts = Counter(str(record["condition"]) for record in records)
    complete = sum(_required_fields_complete(record) for record in records)
    stance_records = [
        record
        for record in records
        if record["condition"] in {"multimodal_conflict", "text_only_conflict"}
        and _label(record, "human_stance")
    ]
    predicted = [
        str(record.get("stance", "")) == "reject_premise"
        for record in stance_records
    ]
    actual = [
        _label(record, "human_stance") == "reject_premise"
        for record in stance_records
    ]
    return {
        "protocol": "exp001-human-semantic-review-v1",
        "status": "complete" if complete == len(records) else "incomplete",
        "row_count": len(records),
        "condition_counts": dict(sorted(condition_counts.items())),
        "fully_labeled_row_count": complete,
        "incomplete_row_count": len(records) - complete,
        "completion_rate": complete / len(records),
        "semantic_labels": _semantic_summary(records),
        "rule_rejection_vs_human_stance": _safe_binary_metrics(
            predicted,
            actual,
        ),
        "human_stance_transitions": _human_stance_transitions(records),
        "candidate_score_alignment": _candidate_alignment(
            records,
            candidate_pairs,
        ),
        "interpretation_boundary": (
            "Metrics use only completed human fields. Null values mean that "
            "there are too few labels or only one class; they must not be "
            "reported as zero performance."
        ),
    }
