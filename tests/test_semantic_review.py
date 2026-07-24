from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_review import (  # noqa: E402
    summarize_semantic_review,
    validate_review_records,
)


def _row(
    image_id: int,
    condition: str,
    semantic: str,
    human_stance: str,
    automatic_stance: str = "no_explicit_rejection",
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "condition": condition,
        "human_semantic_correct": semantic,
        "human_stance": human_stance,
        "stance": automatic_stance,
    }


def test_incomplete_review_reports_progress_without_fake_metrics() -> None:
    summary = summarize_semantic_review(
        [
            _row(1, "multimodal_conflict", "", ""),
            _row(1, "text_only_conflict", "", ""),
            _row(1, "multimodal_clean", "", ""),
        ]
    )
    assert summary["status"] == "incomplete"
    assert summary["fully_labeled_row_count"] == 0
    assert summary["semantic_labels"]["multimodal_conflict"][
        "semantic_correct_rate_excluding_uncertain"
    ] is None
    assert summary["candidate_score_alignment"]["available"] is False


def test_complete_review_metrics_and_candidate_alignment() -> None:
    rows = [
        _row(1, "multimodal_conflict", "yes", "reject_premise", "reject_premise"),
        _row(1, "text_only_conflict", "n/a", "answer_premise"),
        _row(1, "multimodal_clean", "yes", "n/a"),
        _row(2, "multimodal_conflict", "no", "answer_premise", "reject_premise"),
        _row(2, "text_only_conflict", "n/a", "reject_premise"),
        _row(2, "multimodal_clean", "uncertain", "n/a"),
    ]
    pairs = [
        {"image_id": 1, "conflict_visual_delta_mean_logprob": 1.0},
        {"image_id": 2, "conflict_visual_delta_mean_logprob": -1.0},
    ]
    summary = summarize_semantic_review(rows, pairs)
    assert summary["status"] == "complete"
    assert summary["semantic_labels"]["multimodal_conflict"][
        "semantic_correct_rate_excluding_uncertain"
    ] == pytest.approx(0.5)
    rule = summary["rule_rejection_vs_human_stance"]
    assert rule["true_positive"] == 1
    assert rule["false_positive"] == 1
    assert summary["human_stance_transitions"]["transition_counts"] == {
        "multimodal_only_reject": 1,
        "text_only_only_reject": 1,
    }
    assert summary["candidate_score_alignment"]["human_semantic_correct"][
        "auroc"
    ] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("condition", "semantic", "stance", "message"),
    [
        ("text_only_conflict", "yes", "answer_premise", "must be n/a"),
        ("multimodal_clean", "yes", "answer_premise", "stance must be n/a"),
        ("multimodal_conflict", "n/a", "reject_premise", "cannot be n/a"),
        ("multimodal_conflict", "yes", "n/a", "cannot be n/a"),
    ],
)
def test_invalid_condition_specific_labels_are_rejected(
    condition: str,
    semantic: str,
    stance: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_review_records([_row(1, condition, semantic, stance)])


def test_duplicate_review_key_is_rejected() -> None:
    row = _row(1, "multimodal_clean", "yes", "n/a")
    with pytest.raises(ValueError, match="duplicate"):
        validate_review_records([row, row])
