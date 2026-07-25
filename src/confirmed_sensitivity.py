"""Helpers for sensitivity analyses driven by confirmed review issues."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def confirmed_dataset_exclusions(
    prereview_records: Iterable[Mapping[str, object]],
) -> dict[int, str]:
    """Return confirmed multimodal-conflict dataset-validity exclusions."""

    exclusions: dict[int, str] = {}
    for record in prereview_records:
        if record.get("condition") != "multimodal_conflict":
            continue
        if record.get("ai_issue_type") != "dataset_validity":
            continue
        confirmation = str(record.get("human_confirmation", "")).strip().lower()
        if confirmation not in {"accept", "edit"}:
            raise ValueError(
                f"Dataset-validity issue for image_id={record['image_id']} "
                "has not been human-confirmed"
            )
        semantic = (
            str(record.get("ai_semantic_correct", "")).strip().lower()
            if confirmation == "accept"
            else str(record.get("human_correction_semantic", "")).strip().lower()
        )
        if semantic != "uncertain":
            raise ValueError(
                f"Confirmed dataset-validity exclusion for "
                f"image_id={record['image_id']} must resolve to uncertain"
            )
        image_id = int(record["image_id"])
        exclusions[image_id] = str(record.get("ai_notes", "")).strip()
    if not exclusions:
        raise ValueError("No confirmed dataset-validity exclusions found")
    return dict(sorted(exclusions.items()))
