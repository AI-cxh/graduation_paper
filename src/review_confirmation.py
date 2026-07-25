"""Apply auditable human confirmations to AI pre-review suggestions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from semantic_review import validate_review_records


CONFIRMATION_VALUES = {"accept", "edit"}


def confirm_by_confidence(
    records: Iterable[Mapping[str, Any]],
    confidences: set[str],
    note: str,
) -> tuple[list[dict[str, Any]], int]:
    """Mark only selected confidence groups as explicitly accepted."""

    normalized_confidences = {value.strip().lower() for value in confidences}
    if not normalized_confidences or not normalized_confidences <= {
        "low",
        "medium",
        "high",
    }:
        raise ValueError(f"Invalid confidence selection: {confidences}")
    results = []
    changed = 0
    for source in records:
        record = dict(source)
        confidence = str(record.get("ai_confidence", "")).strip().lower()
        confirmation = str(record.get("human_confirmation", "")).strip().lower()
        if confirmation and confirmation not in CONFIRMATION_VALUES:
            raise ValueError(f"Invalid human_confirmation={confirmation!r}")
        if confidence in normalized_confidences:
            if confirmation == "edit":
                raise ValueError(
                    "Cannot overwrite an existing human edit with accept"
                )
            if not confirmation:
                record["human_confirmation"] = "accept"
                record["human_notes"] = note
                changed += 1
        results.append(record)
    return results, changed


def _resolved_labels(record: Mapping[str, Any]) -> tuple[str, str]:
    confirmation = str(record.get("human_confirmation", "")).strip().lower()
    if confirmation == "accept":
        return (
            str(record["ai_semantic_correct"]).strip().lower(),
            str(record["ai_stance"]).strip().lower(),
        )
    if confirmation == "edit":
        semantic = str(record.get("human_correction_semantic", "")).strip().lower()
        stance = str(record.get("human_correction_stance", "")).strip().lower()
        if not semantic or not stance:
            raise ValueError(
                "Edited confirmations require both human correction labels"
            )
        return semantic, stance
    raise ValueError(f"Cannot resolve unconfirmed record: {confirmation!r}")


def merge_confirmed_prereview(
    formal_records: Iterable[Mapping[str, Any]],
    prereview_records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Copy only explicit accept/edit decisions into formal human fields."""

    prereview = {
        (int(record["image_id"]), str(record["condition"])): record
        for record in prereview_records
    }
    formal = [dict(record) for record in formal_records]
    formal_keys = {
        (int(record["image_id"]), str(record["condition"])) for record in formal
    }
    if formal_keys != set(prereview):
        raise ValueError("Formal and pre-review keys do not match")

    changed = 0
    for record in formal:
        key = (int(record["image_id"]), str(record["condition"]))
        source = prereview[key]
        confirmation = str(source.get("human_confirmation", "")).strip().lower()
        if not confirmation:
            continue
        semantic, stance = _resolved_labels(source)
        existing_semantic = str(
            record.get("human_semantic_correct", "")
        ).strip().lower()
        existing_stance = str(record.get("human_stance", "")).strip().lower()
        if existing_semantic and existing_semantic != semantic:
            raise ValueError(f"Conflicting formal semantic label for {key}")
        if existing_stance and existing_stance != stance:
            raise ValueError(f"Conflicting formal stance label for {key}")
        record["human_semantic_correct"] = semantic
        record["human_stance"] = stance
        source_note = str(source.get("human_notes", "")).strip()
        record["human_notes"] = (
            f"{source_note} AI pre-review confidence="
            f"{source.get('ai_confidence', '')}; confirmation={confirmation}."
        ).strip()
        if not existing_semantic or not existing_stance:
            changed += 1
    validate_review_records(formal)
    return formal, changed
