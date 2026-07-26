from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from haloquest_evaluation import (  # noqa: E402
    evaluate_haloquest_record,
    pair_action_records,
    summarize_haloquest_evaluation,
)


def _record(action: str, response: str) -> dict[str, object]:
    return {
        "haloquest_id": 1,
        "action": action,
        "question": "What color is the cat's collar?",
        "groundtruth_response": "The cat does not have a collar.",
        "generated_text": response,
        "image_type": "generated",
    }


def test_target_linked_rejection_is_local_correct() -> None:
    evaluated = evaluate_haloquest_record(
        _record("native_prompt", "The cat is not wearing a collar.")
    )
    assert evaluated["local_verdict"] == "correct"
    assert evaluated["local_utility"] == 1


def test_unsupported_attribute_answer_is_local_incorrect() -> None:
    evaluated = evaluate_haloquest_record(
        _record("native_prompt", "The collar is blue.")
    )
    assert evaluated["local_verdict"] == "incorrect"
    assert evaluated["local_utility"] == 0


def test_generic_information_unavailable_is_not_high_confidence_correct() -> None:
    evaluated = evaluate_haloquest_record(
        _record(
            "native_prompt",
            "The image does not provide information about the collar color.",
        )
    )
    assert evaluated["local_verdict"] == "uncertain"


def test_zero_is_correct_for_false_premise_count_question() -> None:
    record = _record("native_prompt", "0")
    record["question"] = "How many collars does the cat wear?"
    evaluated = evaluate_haloquest_record(record)
    assert evaluated["local_verdict"] == "correct"


def test_pairing_reports_help_and_summary_delta() -> None:
    evaluated = [
        evaluate_haloquest_record(
            _record("native_prompt", "The collar is blue.")
        ),
        evaluate_haloquest_record(
            _record(
                "premise_verification",
                "The cat is not wearing a collar.",
            )
        ),
    ]
    pairs = pair_action_records(evaluated)
    summary = summarize_haloquest_evaluation(evaluated, pairs)
    assert pairs[0]["effect"] == "helps"
    assert summary["paired"]["intervention_minus_native"] == 1.0
