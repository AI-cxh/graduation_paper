from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_semantic_review import (  # noqa: E402
    confirm_action_prereview_by_confidence,
    infer_action_effect,
    infer_overall_action,
    summarize_action_semantic_review,
    validate_action_review_records,
)


def _row(
    image_id: int,
    *,
    confirmation: str = "",
    prefix: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "image_id": image_id,
        "human_confirmation": confirmation,
    }
    for label_prefix in ("ai", "human"):
        record.update(
            {
                f"{label_prefix}_native_conflict_correct": "",
                f"{label_prefix}_intervention_conflict_correct": "",
                f"{label_prefix}_conflict_action_effect": "",
                f"{label_prefix}_native_clean_correct": "",
                f"{label_prefix}_intervention_clean_correct": "",
                f"{label_prefix}_clean_action_effect": "",
                f"{label_prefix}_overall_action": "",
                f"{label_prefix}_confidence": "",
            }
        )
    if prefix:
        record.update(
            {
                f"{prefix}_native_conflict_correct": "no",
                f"{prefix}_intervention_conflict_correct": "yes",
                f"{prefix}_conflict_action_effect": "intervention_helps",
                f"{prefix}_native_clean_correct": "yes",
                f"{prefix}_intervention_clean_correct": "yes",
                f"{prefix}_clean_action_effect": "semantic_tie",
                f"{prefix}_overall_action": "choose_intervention",
                f"{prefix}_confidence": "high",
            }
        )
    return record


@pytest.mark.parametrize(
    ("native", "intervention", "effect"),
    [
        ("no", "yes", "intervention_helps"),
        ("yes", "no", "intervention_harms"),
        ("yes", "yes", "semantic_tie"),
        ("no", "no", "semantic_tie"),
        ("uncertain", "yes", "uncertain"),
    ],
)
def test_infer_action_effect(
    native: str,
    intervention: str,
    effect: str,
) -> None:
    assert infer_action_effect(native, intervention) == effect


@pytest.mark.parametrize(
    ("conflict", "clean", "action"),
    [
        ("intervention_helps", "semantic_tie", "choose_intervention"),
        ("intervention_helps", "intervention_harms", "uncertain"),
        ("intervention_harms", "intervention_helps", "choose_native"),
        ("semantic_tie", "intervention_harms", "choose_native"),
        ("semantic_tie", "semantic_tie", "either"),
        ("uncertain", "semantic_tie", "uncertain"),
    ],
)
def test_infer_overall_action(
    conflict: str,
    clean: str,
    action: str,
) -> None:
    assert infer_overall_action(conflict, clean) == action


def test_ai_prereview_is_not_counted_as_human_confirmation() -> None:
    summary = summarize_action_semantic_review([_row(1, prefix="ai")])
    assert summary["ai_complete_count"] == 1
    assert summary["human_confirmed_count"] == 0
    assert summary["human_confirmed_summary"] is None


def test_accept_resolves_ai_labels() -> None:
    summary = summarize_action_semantic_review(
        [_row(1, prefix="ai", confirmation="accept")]
    )
    assert summary["status"] == "complete"
    human = summary["human_confirmed_summary"]
    assert human["conflict_action_effect_counts"] == {
        "intervention_helps": 1
    }
    assert human["semantic_rates"]["intervention_conflict_correct"][
        "correct_rate_excluding_uncertain"
    ] == pytest.approx(1.0)


def test_edit_requires_all_human_labels() -> None:
    with pytest.raises(ValueError, match="edit requires"):
        validate_action_review_records(
            [_row(1, prefix="ai", confirmation="edit")]
        )


def test_inconsistent_effect_is_rejected() -> None:
    record = _row(1, prefix="ai")
    record["ai_conflict_action_effect"] = "intervention_harms"
    with pytest.raises(ValueError, match="does not match"):
        validate_action_review_records([record])


def test_duplicate_image_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_action_review_records([_row(1), _row(1)])


def test_confirmation_only_accepts_selected_confidence() -> None:
    high = _row(1, prefix="ai")
    medium = _row(2, prefix="ai")
    medium["ai_confidence"] = "medium"
    updated, changed = confirm_action_prereview_by_confidence(
        [high, medium],
        {"medium"},
        "User reviewed this row.",
    )
    assert changed == 1
    assert updated[0]["human_confirmation"] == ""
    assert updated[1]["human_confirmation"] == "accept"
    assert updated[1]["human_notes"] == "User reviewed this row."
