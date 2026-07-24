from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reliability_features import (  # noqa: E402
    PERTURBATIONS,
    apply_image_perturbation,
    build_reliability_features,
    summarize_reliability_features,
)


def _base(
    image_id: int,
    condition: str,
    score: float,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "condition": condition,
        "conflict_type": "object",
        "candidate_answer": "Reject" if "conflict" in condition else "Answer",
        "candidate_mean_logprob": score,
    }


def _perturbed(
    image_id: int,
    condition: str,
    perturbation: str,
    score: float,
) -> dict[str, object]:
    record = _base(image_id, condition, score)
    record["perturbation"] = perturbation
    return record


def _complete_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base = [
        _base(1, "multimodal_conflict", -1.0),
        _base(1, "text_only_conflict", -2.0),
        _base(1, "multimodal_clean", -0.5),
        _base(1, "text_only_clean", -1.0),
    ]
    perturbations = [
        _perturbed(1, "multimodal_conflict", "jpeg_q75", -1.2),
        _perturbed(1, "multimodal_conflict", "gaussian_blur_r1", -2.2),
        _perturbed(1, "multimodal_clean", "jpeg_q75", -0.6),
        _perturbed(1, "multimodal_clean", "gaussian_blur_r1", -0.7),
    ]
    return base, perturbations


@pytest.mark.parametrize("perturbation", PERTURBATIONS)
def test_image_perturbations_are_deterministic_and_size_preserving(
    perturbation: str,
) -> None:
    image = Image.new("RGB", (12, 8), color=(20, 100, 200))
    first = apply_image_perturbation(image, perturbation)
    second = apply_image_perturbation(image, perturbation)
    assert first.mode == "RGB"
    assert first.size == image.size
    assert first.tobytes() == second.tobytes()


def test_reliability_features_include_worst_case_sign_failure() -> None:
    base, perturbations = _complete_records()
    feature = build_reliability_features(base, perturbations)[0]
    assert feature["conflict_identity_visual_delta"] == pytest.approx(1.0)
    assert feature["conflict_minimum_visual_delta"] == pytest.approx(-0.2)
    assert feature["conflict_worst_drop_from_identity"] == pytest.approx(1.2)
    assert feature["conflict_positive_variant_fraction"] == pytest.approx(2 / 3)
    summary = summarize_reliability_features([feature])
    diagnostic = summary["contribution_reliability_diagnostic"]
    assert diagnostic["positive_identity_contribution_count"] == 1
    assert diagnostic["positive_identity_but_nonpositive_worst_case_count"] == 1


def test_missing_perturbation_is_rejected() -> None:
    base, perturbations = _complete_records()
    with pytest.raises(ValueError, match="missing perturbation scores"):
        build_reliability_features(base, perturbations[:-1])


def test_rule_proxy_alignment_reports_feature_auroc() -> None:
    base, perturbations = _complete_records()
    second_base = [
        {**record, "image_id": 2, "candidate_mean_logprob": 0.0}
        for record in base
    ]
    second_perturbations = [
        {**record, "image_id": 2, "candidate_mean_logprob": 0.0}
        for record in perturbations
    ]
    features = build_reliability_features(
        base + second_base,
        perturbations + second_perturbations,
    )
    generation = [
        {"image_id": 1, "condition": "multimodal_conflict", "stance": "reject_premise"},
        {
            "image_id": 1,
            "condition": "text_only_conflict",
            "stance": "no_explicit_rejection",
        },
        {
            "image_id": 2,
            "condition": "multimodal_conflict",
            "stance": "no_explicit_rejection",
        },
        {
            "image_id": 2,
            "condition": "text_only_conflict",
            "stance": "no_explicit_rejection",
        },
    ]
    summary = summarize_reliability_features(features, generation)
    assert summary["rule_proxy_alignment"]["feature_auroc"][
        "contribution_identity"
    ]["multimodal_explicit_rejection"] == pytest.approx(1.0)
