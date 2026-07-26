from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vqav2_control import (  # noqa: E402
    normalize_vqa_answer,
    normalized_vqa_soft_accuracy,
    pair_vqav2_actions,
    select_balanced_vqav2_sample,
    vqa_soft_accuracy,
)


def test_normalize_vqa_answer_matches_official_examples() -> None:
    assert normalize_vqa_answer("The two cats.") == "2 cats"
    assert normalize_vqa_answer("Dont, guess!") == "don't guess"


def test_vqa_soft_accuracy_uses_leave_one_annotator_out_rule() -> None:
    answers = [
        {"answer_id": index, "answer": "blue" if index < 3 else "red"}
        for index in range(10)
    ]
    assert vqa_soft_accuracy("blue", answers) == pytest.approx(0.9)
    assert vqa_soft_accuracy("red", answers) == pytest.approx(1.0)
    assert vqa_soft_accuracy("green", answers) == pytest.approx(0.0)


def test_normalized_score_removes_unanimous_reference_case_artifact() -> None:
    answers = [
        {"answer_id": index, "answer": "yes"} for index in range(10)
    ]
    assert vqa_soft_accuracy("Yes", answers) == pytest.approx(0.0)
    assert normalized_vqa_soft_accuracy("Yes", answers) == pytest.approx(1.0)


def test_balanced_selection_is_deterministic_and_unique_image() -> None:
    questions = []
    annotations = []
    question_id = 1
    for answer_type in ("yes/no", "number", "other"):
        for index in range(6):
            questions.append(
                {
                    "question_id": question_id,
                    "image_id": index,
                    "question": f"Question {question_id}?",
                }
            )
            answer = (
                "yes"
                if answer_type == "yes/no"
                else "2"
                if answer_type == "number"
                else "blue"
            )
            annotations.append(
                {
                    "question_id": question_id,
                    "image_id": index,
                    "answer_type": answer_type,
                    "question_type": "what",
                    "multiple_choice_answer": answer,
                    "answers": [
                        {
                            "answer_id": answer_id,
                            "answer": answer,
                            "answer_confidence": "yes",
                        }
                        for answer_id in range(10)
                    ],
                }
            )
            question_id += 1
    selected = select_balanced_vqav2_sample(
        questions,
        annotations,
        quotas={"yes/no": 2, "number": 2, "other": 2},
        minimum_consensus=6,
        seed=7,
    )
    assert len(selected) == 6
    assert len({record["image_id"] for record in selected}) == 6
    repeated = select_balanced_vqav2_sample(
        questions,
        annotations,
        quotas={"yes/no": 2, "number": 2, "other": 2},
        minimum_consensus=6,
        seed=7,
    )
    assert [row["question_id"] for row in selected] == [
        row["question_id"] for row in repeated
    ]


def test_pair_actions_uses_soft_score_direction() -> None:
    common = {
        "question_id": 1,
        "image_id": 2,
        "image_sha256": "abc",
        "answer_type": "other",
        "question_type": "what",
        "question": "What color?",
        "multiple_choice_answer": "blue",
        "generic_refusal_proxy": False,
        "normalized_vqa_soft_accuracy": 0.0,
        "vqa_soft_accuracy": 0.0,
    }
    pairs = pair_vqav2_actions(
        [
            {
                **common,
                "action": "native_prompt",
                "generated_text": "red",
                "vqa_soft_accuracy": 0.0,
                "normalized_vqa_soft_accuracy": 0.0,
            },
            {
                **common,
                "action": "premise_verification",
                "generated_text": "blue",
                "vqa_soft_accuracy": 1.0,
                "normalized_vqa_soft_accuracy": 1.0,
            },
        ],
        effect_epsilon=1e-12,
    )
    assert pairs[0]["effect"] == "helps"
    assert pairs[0]["intervention_minus_native"] == pytest.approx(1.0)
