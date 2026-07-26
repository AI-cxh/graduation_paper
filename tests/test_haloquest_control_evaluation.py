from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from haloquest_control_evaluation import (  # noqa: E402
    evaluate_control_record,
    pair_control_actions,
    split_reference_variants,
    summarize_control_evaluation,
)


def _record(action: str, response: str) -> dict[str, object]:
    return {
        "haloquest_id": 1,
        "action": action,
        "question": "What color are the ears?",
        "groundtruth_response": "The ears are gray.; Gray",
        "generated_text": response,
        "image_type": "generated",
        "hallucination_type": "visual challenge",
    }


def test_reference_variants_ignore_empty_segments() -> None:
    assert split_reference_variants("First; Second;") == ["First", "Second"]


def test_control_record_uses_best_reference_variant() -> None:
    result = evaluate_control_record(_record("native_prompt", "Gray"))
    assert result["best_normalized_exact_match"] is True
    assert result["best_token_f1"] == 1.0


def test_pair_flags_large_answer_change_for_audit() -> None:
    evaluated = [
        evaluate_control_record(_record("native_prompt", "The ears are gray.")),
        evaluate_control_record(
            _record(
                "premise_verification",
                "The image does not show any ears.",
            )
        ),
    ]
    pairs = pair_control_actions(evaluated)
    summary = summarize_control_evaluation(evaluated, pairs)
    assert pairs[0]["audit_candidate"] is True
    assert pairs[0]["lexical_delta"] < 0
    assert summary["audit_candidate_count"] == 1

