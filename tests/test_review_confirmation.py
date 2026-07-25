from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from review_confirmation import (  # noqa: E402
    confirm_by_confidence,
    merge_confirmed_prereview,
)


def _formal(image_id: int, condition: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "condition": condition,
        "human_semantic_correct": "",
        "human_stance": "",
        "human_notes": "",
    }


def _prereview(
    image_id: int,
    condition: str,
    confidence: str,
    semantic: str,
    stance: str,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "condition": condition,
        "ai_confidence": confidence,
        "ai_semantic_correct": semantic,
        "ai_stance": stance,
        "human_confirmation": "",
        "human_correction_semantic": "",
        "human_correction_stance": "",
        "human_notes": "",
    }


def test_only_requested_confidences_are_confirmed_and_merged() -> None:
    prereview = [
        _prereview(
            1,
            "multimodal_conflict",
            "low",
            "uncertain",
            "answer_premise",
        ),
        _prereview(1, "text_only_conflict", "medium", "n/a", "abstain"),
        _prereview(1, "multimodal_clean", "high", "yes", "n/a"),
    ]
    confirmed, changed = confirm_by_confidence(
        prereview,
        {"low", "medium"},
        "confirmed",
    )
    assert changed == 2
    assert [row["human_confirmation"] for row in confirmed] == [
        "accept",
        "accept",
        "",
    ]
    formal = [
        _formal(1, "multimodal_conflict"),
        _formal(1, "text_only_conflict"),
        _formal(1, "multimodal_clean"),
    ]
    merged, merged_count = merge_confirmed_prereview(formal, confirmed)
    assert merged_count == 2
    assert merged[0]["human_semantic_correct"] == "uncertain"
    assert merged[1]["human_stance"] == "abstain"
    assert merged[2]["human_semantic_correct"] == ""


def test_existing_edit_is_not_overwritten_by_bulk_accept() -> None:
    row = _prereview(
        1,
        "multimodal_conflict",
        "low",
        "yes",
        "reject_premise",
    )
    row["human_confirmation"] = "edit"
    with pytest.raises(ValueError, match="overwrite"):
        confirm_by_confidence([row], {"low"}, "confirmed")


def test_edit_requires_correction_labels() -> None:
    prereview = [
        _prereview(
            1,
            "multimodal_conflict",
            "low",
            "yes",
            "reject_premise",
        )
    ]
    prereview[0]["human_confirmation"] = "edit"
    with pytest.raises(ValueError, match="require both"):
        merge_confirmed_prereview(
            [_formal(1, "multimodal_conflict")],
            prereview,
        )
