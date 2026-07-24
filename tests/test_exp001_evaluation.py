from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from exp001_evaluation import (  # noqa: E402
    classify_conflict_response,
    evaluate_record,
    lexical_metrics,
    normalize_text,
    summarize_records,
)


def _record(
    *,
    condition: str = "multimodal_conflict",
    reference: str = "The image does not contain any bicycles.",
    prediction: str = "There are no bicycles visible in the image.",
) -> dict[str, object]:
    return {
        "image_id": 7,
        "condition": condition,
        "conflict_type": "object",
        "question": "How many bicycles are shown?",
        "reference_answer": reference,
        "generated_text": prediction,
    }


def test_normalize_text_handles_list_marker_articles_and_numbers() -> None:
    assert (
        normalize_text("1. The image does not contain two chairs!")
        == "image does not contain 2 chairs"
    )


def test_lexical_metrics_are_auditable() -> None:
    metrics = lexical_metrics("Two red cars.", "There are two red cars.")
    assert metrics["normalized_exact_match"] is False
    assert metrics["reference_contained"] is True
    assert metrics["token_recall"] == pytest.approx(1.0)
    assert metrics["token_precision"] == pytest.approx(3 / 5)
    assert metrics["token_f1"] == pytest.approx(0.75)
    assert metrics["rouge_l_f1"] == pytest.approx(0.75)


def test_target_linked_rejection_is_separate_from_unrelated_rejection() -> None:
    linked = classify_conflict_response(
        "The image does not contain any bicycles.",
        "There are no bicycles visible in the image.",
    )
    unrelated = classify_conflict_response(
        "The image does not contain any bicycles.",
        "There are no clouds visible in the image.",
    )
    assert linked["stance"] == "reject_premise"
    assert linked["reference_content_overlap"] == ["bicycle"]
    assert unrelated["stance"] == "ambiguous_rejection"


def test_missing_image_abstention_is_not_counted_as_premise_rejection() -> None:
    result = classify_conflict_response(
        "The image does not contain a chair.",
        "I cannot see the image. Please upload it.",
    )
    assert result["stance"] == "abstain_no_visual_input"


@pytest.mark.parametrize(
    ("reference", "prediction"),
    [
        (
            "The image does not contain a bus.",
            "The image shows a train, not a bus.",
        ),
        (
            "The image does not show the cat sitting on the chair.",
            "The cat is not sitting on the chair.",
        ),
        (
            "The image does not contain a woman's hat.",
            "The woman is not wearing a hat.",
        ),
    ],
)
def test_target_local_negation_is_a_rejection(
    reference: str,
    prediction: str,
) -> None:
    result = classify_conflict_response(reference, prediction)
    assert result["stance"] == "reject_premise"
    assert result["stance_rule"] == "target_local_negation"


def test_uncertainty_near_target_is_not_a_rejection() -> None:
    result = classify_conflict_response(
        "The image does not contain a book.",
        "I am not sure where the book is.",
    )
    assert result["stance"] == "no_explicit_rejection"


def test_answer_without_rejection_is_flagged_for_review() -> None:
    evaluated = evaluate_record(
        _record(prediction="There are three bicycles near the building.")
    )
    assert evaluated["stance"] == "no_explicit_rejection"
    assert evaluated["manual_review_required"] is True


def test_clean_exact_or_contained_answer_does_not_require_review() -> None:
    exact = evaluate_record(
        _record(
            condition="multimodal_clean",
            reference="Skis.",
            prediction="Skis.",
        )
    )
    contained = evaluate_record(
        _record(
            condition="multimodal_clean",
            reference="Skis.",
            prediction="The airplane is equipped with skis.",
        )
    )
    assert exact["manual_review_required"] is False
    assert contained["manual_review_required"] is False


def test_clean_uncertain_containment_requires_review() -> None:
    evaluated = evaluate_record(
        _record(
            condition="multimodal_clean",
            reference="Flowers.",
            prediction="I am not sure; possibly flowers or a plant.",
        )
    )
    assert evaluated["reference_contained"] is True
    assert evaluated["prediction_uncertainty_flag"] is True
    assert evaluated["manual_review_required"] is True


def test_missing_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        evaluate_record({"image_id": 1})


def test_summary_reports_rates_without_calling_them_accuracy() -> None:
    records = [
        evaluate_record(_record()),
        evaluate_record(
            {
                **_record(),
                "image_id": 8,
                "generated_text": "There are two bicycles.",
            }
        ),
    ]
    summary = summarize_records(records)
    group = summary["by_condition"]["multimodal_conflict"]
    assert group["count"] == 2
    assert group["explicit_target_rejection_rate"] == pytest.approx(0.5)
    assert group["manual_review_count"] == 1
    assert not any("accuracy" in key for key in group)
    assert summary["paired_conflict_stance"]["available"] is False


def test_paired_conflict_summary_reports_directional_changes() -> None:
    records = []
    for image_id, multimodal_rejects, text_only_rejects in [
        (1, True, False),
        (2, True, True),
        (3, False, True),
        (4, False, False),
    ]:
        for condition, rejects in [
            ("multimodal_conflict", multimodal_rejects),
            ("text_only_conflict", text_only_rejects),
        ]:
            records.append(
                evaluate_record(
                    {
                        **_record(
                            condition=condition,
                            prediction=(
                                "There are no bicycles visible in the image."
                                if rejects
                                else "There are two bicycles."
                            ),
                        ),
                        "image_id": image_id,
                    }
                )
            )
    paired = summarize_records(records)["paired_conflict_stance"]
    assert paired["available"] is True
    assert paired["transition_counts"] == {
        "both_reject": 1,
        "multimodal_only_reject": 1,
        "text_only_only_reject": 1,
        "neither_reject": 1,
    }
    assert paired["multimodal_minus_text_only_rejection_rate"] == 0.0
    assert paired["exact_mcnemar_p_value"] == 1.0
