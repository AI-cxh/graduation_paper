from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from candidate_scoring import (  # noqa: E402
    align_scores_with_free_generation,
    build_pair_deltas,
    candidate_logprob_statistics,
    summarize_delta_values,
    summarize_reference_scores,
)


def test_candidate_logprob_alignment() -> None:
    input_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.zeros((1, 4, 5))
    logits[0, 1, 2] = 2.0
    logits[0, 2, 3] = 1.0
    result = candidate_logprob_statistics(logits, input_ids, prompt_length=2)
    expected = torch.log_softmax(logits[0, 1:3], dim=-1)[
        torch.arange(2), torch.tensor([2, 3])
    ]
    assert result["candidate_token_count"] == 2
    assert result["candidate_sum_logprob"] == pytest.approx(float(expected.sum()))
    assert result["candidate_mean_logprob"] == pytest.approx(
        float(expected.mean())
    )
    assert result["candidate_token_ids"] == [2, 3]


@pytest.mark.parametrize("prompt_length", [0, 4])
def test_candidate_logprob_rejects_invalid_prompt_length(
    prompt_length: int,
) -> None:
    with pytest.raises(ValueError, match="prompt_length"):
        candidate_logprob_statistics(
            torch.zeros((1, 4, 5)),
            torch.tensor([[0, 1, 2, 3]]),
            prompt_length,
        )


def _score_record(
    image_id: int,
    condition: str,
    score: float,
    *,
    conflict_type: str = "object",
) -> dict[str, object]:
    is_conflict = "conflict" in condition
    return {
        "image_id": image_id,
        "condition": condition,
        "conflict_type": conflict_type,
        "candidate_answer": "Reject." if is_conflict else "Answer.",
        "candidate_token_count": 2,
        "candidate_mean_logprob": score,
        "candidate_sum_logprob": 2 * score,
        "candidate_perplexity": 1.0,
    }


def test_build_pair_deltas() -> None:
    records = [
        _score_record(1, "multimodal_conflict", -1.0),
        _score_record(1, "text_only_conflict", -2.0),
        _score_record(1, "multimodal_clean", -0.5),
        _score_record(1, "text_only_clean", -1.0),
    ]
    pair = build_pair_deltas(records)[0]
    assert pair["conflict_visual_delta_mean_logprob"] == pytest.approx(1.0)
    assert pair["clean_visual_delta_mean_logprob"] == pytest.approx(0.5)
    assert pair["conflict_minus_clean_visual_delta_mean_logprob"] == pytest.approx(
        0.5
    )


def test_missing_score_condition_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing score conditions"):
        build_pair_deltas(
            [
                _score_record(1, "multimodal_conflict", -1.0),
                _score_record(1, "text_only_conflict", -2.0),
            ]
        )


def test_delta_summary_uses_exact_sign_test() -> None:
    summary = summarize_delta_values([1.0, 2.0, 3.0, -1.0])
    assert summary["positive_count"] == 3
    assert summary["negative_count"] == 1
    assert summary["positive_rate"] == pytest.approx(0.75)
    assert summary["exact_sign_test_p_value"] == pytest.approx(0.625)


def test_reference_score_summary() -> None:
    records = []
    for image_id, conflict_delta, clean_delta in [
        (1, 1.0, 0.5),
        (2, -0.5, 0.25),
    ]:
        records.extend(
            [
                _score_record(image_id, "multimodal_conflict", conflict_delta),
                _score_record(image_id, "text_only_conflict", 0.0),
                _score_record(image_id, "multimodal_clean", clean_delta),
                _score_record(image_id, "text_only_clean", 0.0),
            ]
        )
    pairs, summary = summarize_reference_scores(records)
    assert len(pairs) == 2
    assert summary["record_count"] == 8
    assert summary["pair_count"] == 2
    conflict = summary["paired_deltas"][
        "conflict_visual_delta_mean_logprob"
    ]
    assert conflict["mean"] == pytest.approx(0.25)
    assert conflict["positive_count"] == 1


def test_align_scores_with_free_generation() -> None:
    records = []
    generation = []
    definitions = [
        (1, 1.0, True, False),
        (2, 0.5, True, True),
        (3, -0.5, False, True),
        (4, -1.0, False, False),
    ]
    for image_id, conflict_delta, multimodal_rejects, text_only_rejects in definitions:
        records.extend(
            [
                _score_record(image_id, "multimodal_conflict", conflict_delta),
                _score_record(image_id, "text_only_conflict", 0.0),
                _score_record(image_id, "multimodal_clean", 0.0),
                _score_record(image_id, "text_only_clean", 0.0),
            ]
        )
        generation.extend(
            [
                {
                    "image_id": image_id,
                    "condition": "multimodal_conflict",
                    "stance": (
                        "reject_premise"
                        if multimodal_rejects
                        else "no_explicit_rejection"
                    ),
                },
                {
                    "image_id": image_id,
                    "condition": "text_only_conflict",
                    "stance": (
                        "reject_premise"
                        if text_only_rejects
                        else "no_explicit_rejection"
                    ),
                },
            ]
        )
    pairs = build_pair_deltas(records)
    alignment = align_scores_with_free_generation(pairs, generation)
    assert alignment["pair_count"] == 4
    assert alignment["auroc_for_multimodal_explicit_rejection"] == 1.0
    assert (
        alignment["by_free_generation_transition"]["multimodal_only_reject"][
            "mean_conflict_visual_delta_mean_logprob"
        ]
        == 1.0
    )
