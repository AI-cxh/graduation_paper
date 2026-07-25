from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intervention_evaluation import (  # noqa: E402
    align_intervention_utility_with_features,
    compare_intervention_outputs,
)
from intervention_prompts import apply_prompt_intervention  # noqa: E402


def test_premise_verification_prompt_preserves_question() -> None:
    result = apply_prompt_intervention("Where is the cat?", "premise_verification")
    assert "Check the image" in result
    assert result.endswith("Question: Where is the cat?")
    assert apply_prompt_intervention("Q?", "native_prompt") == "Q?"


def _record(
    image_id: int,
    condition: str,
    *,
    rejects: bool = False,
    contained: bool = False,
    token_f1: float = 0.0,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "condition": condition,
        "conflict_type": "object",
        "stance": "reject_premise" if rejects else "no_explicit_rejection",
        "reference_contained": contained,
        "token_f1": token_f1,
        "rouge_l_f1": token_f1,
    }


def test_intervention_comparison_tracks_help_and_harm() -> None:
    baseline = [
        _record(1, "multimodal_conflict", rejects=False),
        _record(1, "multimodal_clean", contained=True, token_f1=0.5),
        _record(2, "multimodal_conflict", rejects=True),
        _record(2, "multimodal_clean", contained=False, token_f1=0.0),
    ]
    intervention = [
        _record(1, "multimodal_conflict", rejects=True),
        _record(1, "multimodal_clean", contained=False, token_f1=0.25),
        _record(2, "multimodal_conflict", rejects=False),
        _record(2, "multimodal_clean", contained=True, token_f1=0.5),
    ]
    summary = compare_intervention_outputs(baseline, intervention)
    assert summary["pair_count"] == 2
    conflict = summary["conflict_rejection_proxy"]
    assert conflict["transition_counts"]["intervention_only_success"] == 1
    assert conflict["transition_counts"]["baseline_only_success"] == 1
    assert conflict["exact_mcnemar_p_value"] == pytest.approx(1.0)
    heterogeneity = summary["action_heterogeneity_proxy"]
    assert heterogeneity["intervention_helps_clean_containment_count"] == 1
    assert heterogeneity["intervention_harms_clean_containment_count"] == 1


def test_misaligned_intervention_records_are_rejected() -> None:
    with pytest.raises(ValueError, match="not pair-aligned"):
        compare_intervention_outputs(
            [_record(1, "multimodal_conflict")],
            [_record(2, "multimodal_conflict")],
        )


def test_feature_alignment_uses_hypothesis_fixed_directions() -> None:
    baseline = []
    intervention = []
    features = []
    for image_id, baseline_rejects, intervention_rejects, contribution in [
        (1, False, True, -1.0),
        (2, True, True, 0.0),
        (3, False, False, 1.0),
    ]:
        baseline.extend(
            [
                _record(
                    image_id,
                    "multimodal_conflict",
                    rejects=baseline_rejects,
                ),
                _record(image_id, "multimodal_clean"),
            ]
        )
        intervention.extend(
            [
                _record(
                    image_id,
                    "multimodal_conflict",
                    rejects=intervention_rejects,
                ),
                _record(image_id, "multimodal_clean"),
            ]
        )
        features.append(
            {
                "image_id": image_id,
                "conflict_identity_visual_delta": contribution,
                "conflict_minimum_visual_delta": contribution,
                "conflict_population_std_visual_delta": -contribution,
                "conflict_worst_drop_from_identity": -contribution,
            }
        )
    records, summary = align_intervention_utility_with_features(
        baseline,
        intervention,
        features,
    )
    assert len(records) == 3
    assert summary["feature_auroc"]["negative_native_contribution"][
        "auroc_for_intervention_only_success"
    ] == pytest.approx(1.0)
