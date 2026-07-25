"""Validation and summaries for paired EXP-003 action semantic review."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


SEMANTIC_LABELS = {"yes", "no", "uncertain"}
EFFECT_LABELS = {
    "intervention_helps",
    "intervention_harms",
    "semantic_tie",
    "uncertain",
}
ACTION_LABELS = {
    "choose_intervention",
    "choose_native",
    "either",
    "uncertain",
}
CONFIDENCE_LABELS = {"high", "medium", "low"}
CONFIRMATION_LABELS = {"accept", "edit"}

AI_FIELDS = (
    "ai_native_conflict_correct",
    "ai_intervention_conflict_correct",
    "ai_conflict_action_effect",
    "ai_native_clean_correct",
    "ai_intervention_clean_correct",
    "ai_clean_action_effect",
    "ai_overall_action",
    "ai_confidence",
)
HUMAN_FIELDS = (
    "human_native_conflict_correct",
    "human_intervention_conflict_correct",
    "human_conflict_action_effect",
    "human_native_clean_correct",
    "human_intervention_clean_correct",
    "human_clean_action_effect",
    "human_overall_action",
    "human_confidence",
)


def _label(record: Mapping[str, Any], field: str) -> str:
    return str(record.get(field, "")).strip().lower()


def infer_action_effect(native: str, intervention: str) -> str:
    """Infer a correctness transition from two semantic labels."""

    if native == "uncertain" or intervention == "uncertain":
        return "uncertain"
    if native == "no" and intervention == "yes":
        return "intervention_helps"
    if native == "yes" and intervention == "no":
        return "intervention_harms"
    return "semantic_tie"


def infer_overall_action(
    conflict_effect: str,
    clean_effect: str,
) -> str:
    """Combine conflict correction with clean preservation lexicographically."""

    if "uncertain" in {conflict_effect, clean_effect}:
        return "uncertain"
    if conflict_effect == "intervention_helps":
        return (
            "uncertain"
            if clean_effect == "intervention_harms"
            else "choose_intervention"
        )
    if conflict_effect == "intervention_harms":
        return "choose_native"
    if clean_effect == "intervention_helps":
        return "choose_intervention"
    if clean_effect == "intervention_harms":
        return "choose_native"
    return "either"


def _validate_label(
    record: Mapping[str, Any],
    field: str,
    allowed: set[str],
    row_number: int,
) -> str:
    value = _label(record, field)
    if value and value not in allowed:
        raise ValueError(f"Row {row_number}: invalid {field}={value!r}")
    return value


def _validate_label_group(
    record: dict[str, Any],
    *,
    prefix: str,
    row_number: int,
    require_complete: bool,
    allow_partial: bool,
) -> None:
    native_conflict = _validate_label(
        record,
        f"{prefix}_native_conflict_correct",
        SEMANTIC_LABELS,
        row_number,
    )
    intervention_conflict = _validate_label(
        record,
        f"{prefix}_intervention_conflict_correct",
        SEMANTIC_LABELS,
        row_number,
    )
    conflict_effect = _validate_label(
        record,
        f"{prefix}_conflict_action_effect",
        EFFECT_LABELS,
        row_number,
    )
    native_clean = _validate_label(
        record,
        f"{prefix}_native_clean_correct",
        SEMANTIC_LABELS,
        row_number,
    )
    intervention_clean = _validate_label(
        record,
        f"{prefix}_intervention_clean_correct",
        SEMANTIC_LABELS,
        row_number,
    )
    clean_effect = _validate_label(
        record,
        f"{prefix}_clean_action_effect",
        EFFECT_LABELS,
        row_number,
    )
    _validate_label(
        record,
        f"{prefix}_overall_action",
        ACTION_LABELS,
        row_number,
    )
    _validate_label(
        record,
        f"{prefix}_confidence",
        CONFIDENCE_LABELS,
        row_number,
    )
    values = [
        native_conflict,
        intervention_conflict,
        conflict_effect,
        native_clean,
        intervention_clean,
        clean_effect,
        _label(record, f"{prefix}_overall_action"),
        _label(record, f"{prefix}_confidence"),
    ]
    if require_complete and not all(values):
        raise ValueError(
            f"Row {row_number}: {prefix} labels must be complete"
        )
    if any(values) and not all(values) and not allow_partial:
        raise ValueError(
            f"Row {row_number}: partially completed {prefix} label group"
        )
    if all((native_conflict, intervention_conflict, conflict_effect)):
        expected_conflict = infer_action_effect(
            native_conflict,
            intervention_conflict,
        )
        if conflict_effect != expected_conflict:
            raise ValueError(
                f"Row {row_number}: conflict effect {conflict_effect!r} "
                f"does not match correctness transition {expected_conflict!r}"
            )
    if all((native_clean, intervention_clean, clean_effect)):
        expected_clean = infer_action_effect(native_clean, intervention_clean)
        if clean_effect != expected_clean:
            raise ValueError(
                f"Row {row_number}: clean effect {clean_effect!r} "
                f"does not match correctness transition {expected_clean!r}"
            )
    if all(values):
        expected_overall = infer_overall_action(
            conflict_effect,
            clean_effect,
        )
        overall = _label(record, f"{prefix}_overall_action")
        if overall != expected_overall:
            raise ValueError(
                f"Row {row_number}: overall action {overall!r} does not "
                f"match branch effects {expected_overall!r}"
            )
    for field in (
        f"{prefix}_native_conflict_correct",
        f"{prefix}_intervention_conflict_correct",
        f"{prefix}_conflict_action_effect",
        f"{prefix}_native_clean_correct",
        f"{prefix}_intervention_clean_correct",
        f"{prefix}_clean_action_effect",
        f"{prefix}_overall_action",
        f"{prefix}_confidence",
    ):
        record[field] = _label(record, field)


def validate_action_review_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize labels and reject partial or inconsistent action reviews."""

    normalized = []
    seen: set[int] = set()
    for row_number, source in enumerate(records, start=2):
        record = dict(source)
        try:
            image_id = int(record["image_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Row {row_number}: invalid image_id") from error
        if image_id in seen:
            raise ValueError(f"Row {row_number}: duplicate image_id={image_id}")
        seen.add(image_id)
        _validate_label_group(
            record,
            prefix="ai",
            row_number=row_number,
            require_complete=False,
            allow_partial=True,
        )
        _validate_label_group(
            record,
            prefix="human",
            row_number=row_number,
            require_complete=False,
            allow_partial=False,
        )
        confirmation = _validate_label(
            record,
            "human_confirmation",
            CONFIRMATION_LABELS,
            row_number,
        )
        if confirmation == "accept" and not all(
            _label(record, field) for field in AI_FIELDS
        ):
            raise ValueError(
                f"Row {row_number}: cannot accept incomplete AI labels"
            )
        if confirmation == "edit" and not all(
            _label(record, field) for field in HUMAN_FIELDS
        ):
            raise ValueError(
                f"Row {row_number}: edit requires complete human labels"
            )
        if confirmation == "accept" and any(
            _label(record, field) for field in HUMAN_FIELDS
        ):
            raise ValueError(
                f"Row {row_number}: accepted AI row must not contain edits"
            )
        record["image_id"] = image_id
        record["human_confirmation"] = confirmation
        normalized.append(record)
    if not normalized:
        raise ValueError("Action review table is empty")
    return normalized


def resolved_human_labels(
    record: Mapping[str, Any],
) -> dict[str, str] | None:
    """Resolve explicit accept/edit confirmation without treating AI as human."""

    confirmation = _label(record, "human_confirmation")
    if confirmation == "accept":
        return {
            field.removeprefix("ai_"): _label(record, field)
            for field in AI_FIELDS
        }
    if confirmation == "edit":
        return {
            field.removeprefix("human_"): _label(record, field)
            for field in HUMAN_FIELDS
        }
    return None


def confirm_action_prereview_by_confidence(
    records: Iterable[Mapping[str, Any]],
    confidences: set[str],
    note: str,
) -> tuple[list[dict[str, Any]], int]:
    """Accept selected complete AI rows after explicit user review."""

    selected = {value.strip().lower() for value in confidences}
    if not selected or not selected <= CONFIDENCE_LABELS:
        raise ValueError(f"Invalid confidence selection: {confidences}")
    normalized = validate_action_review_records(records)
    changed = 0
    for record in normalized:
        if _label(record, "ai_confidence") not in selected:
            continue
        confirmation = _label(record, "human_confirmation")
        if confirmation == "edit":
            raise ValueError("Cannot overwrite an existing human edit")
        if not confirmation:
            if not all(_label(record, field) for field in AI_FIELDS):
                raise ValueError("Cannot accept an incomplete AI pre-review")
            record["human_confirmation"] = "accept"
            record["human_notes"] = note
            changed += 1
    validate_action_review_records(normalized)
    return normalized, changed


def _semantic_rates(
    resolved: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    fields = (
        "native_conflict_correct",
        "intervention_conflict_correct",
        "native_clean_correct",
        "intervention_clean_correct",
    )
    result = {}
    for field in fields:
        counts = Counter(labels[field] for labels in resolved)
        decided = counts["yes"] + counts["no"]
        result[field] = {
            "label_counts": dict(sorted(counts.items())),
            "decided_count": decided,
            "correct_rate_excluding_uncertain": (
                counts["yes"] / decided if decided else None
            ),
        }
    return result


def summarize_action_semantic_review(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize AI pre-review separately from confirmed human decisions."""

    normalized = validate_action_review_records(records)
    ai_complete = [
        record for record in normalized if all(_label(record, field) for field in AI_FIELDS)
    ]
    resolved = [
        labels
        for record in normalized
        if (labels := resolved_human_labels(record)) is not None
    ]
    confirmations = Counter(
        _label(record, "human_confirmation") or "unconfirmed"
        for record in normalized
    )
    summary: dict[str, Any] = {
        "protocol": "exp003-paired-action-semantic-review-v1",
        "row_count": len(normalized),
        "ai_complete_count": len(ai_complete),
        "human_confirmed_count": len(resolved),
        "human_confirmation_counts": dict(sorted(confirmations.items())),
        "status": "complete" if len(resolved) == len(normalized) else "incomplete",
        "scope": (
            "AI labels are pre-review suggestions only. Metrics under "
            "human_confirmed_summary use only explicit accept/edit decisions."
        ),
    }
    ai_confidence = Counter(_label(record, "ai_confidence") for record in ai_complete)
    ai_overall = Counter(_label(record, "ai_overall_action") for record in ai_complete)
    summary["ai_prereview"] = {
        "confidence_counts": dict(sorted(ai_confidence.items())),
        "overall_action_counts": dict(sorted(ai_overall.items())),
    }
    if not resolved:
        summary["human_confirmed_summary"] = None
        return summary
    summary["human_confirmed_summary"] = {
        "semantic_rates": _semantic_rates(resolved),
        "conflict_action_effect_counts": dict(
            sorted(Counter(labels["conflict_action_effect"] for labels in resolved).items())
        ),
        "clean_action_effect_counts": dict(
            sorted(Counter(labels["clean_action_effect"] for labels in resolved).items())
        ),
        "overall_action_counts": dict(
            sorted(Counter(labels["overall_action"] for labels in resolved).items())
        ),
        "confidence_counts": dict(
            sorted(Counter(labels["confidence"] for labels in resolved).items())
        ),
    }
    return summary
