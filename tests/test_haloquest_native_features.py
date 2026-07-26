from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from haloquest_native_features import (  # noqa: E402
    SCORE_CONDITIONS,
    bootstrap_binary_roc_auc,
    build_audited_samples,
    build_native_answer_features,
    summarize_native_answer_features,
)


def test_stratified_bootstrap_auc_is_deterministic() -> None:
    result = bootstrap_binary_roc_auc(
        [0.9, 0.8, 0.2, 0.1],
        [True, True, False, False],
        resamples=20,
        seed=7,
    )
    assert result is not None
    assert result["observed"] == pytest.approx(1.0)
    assert result["stratified_95_percent_interval"] == pytest.approx(
        [1.0, 1.0]
    )


def test_build_audited_samples_excludes_uncertain() -> None:
    manifest = [
        {
            "haloquest_id": 1,
            "download_status": "available",
            "hallucination_type": "false premises",
            "image_type": "generated",
            "local_image_path": "image.png",
            "image_sha256": "abc",
            "question": "Question?",
        },
        {
            "haloquest_id": 2,
            "download_status": "available",
            "hallucination_type": "false premises",
            "image_type": "generated",
            "local_image_path": "image2.png",
            "image_sha256": "def",
            "question": "Question 2?",
        },
    ]
    predictions = [
        {
            "haloquest_id": row_id,
            "action": "native_prompt",
            "generated_text": "Native answer",
            "generated_token_count": 2,
            "mean_transition_logprob": -0.2,
        }
        for row_id in (1, 2)
    ]
    audit = {
        "effect_groups": {
            "helps": [1],
            "harms": [],
            "tie": [],
            "uncertain": [2],
        }
    }
    samples = build_audited_samples(
        scope="false_premise",
        manifest_records=manifest,
        prediction_records=predictions,
        audit=audit,
        accepted_effects={"helps", "harms", "tie"},
    )
    assert [sample["haloquest_id"] for sample in samples] == [1]
    assert samples[0]["action_effect"] == "helps"


def _score(condition: str, value: float) -> dict[str, object]:
    return {
        "haloquest_id": 1,
        "condition": condition,
        "candidate_answer": "Native answer",
        "candidate_token_ids": [10, 11],
        "candidate_mean_logprob": value,
    }


def test_build_features_uses_native_candidate_without_reference() -> None:
    sample = {
        "haloquest_id": 1,
        "dataset_scope": "false_premise",
        "hallucination_type": "false premises",
        "image_type": "generated",
        "local_image_path": "image.png",
        "image_sha256": "abc",
        "question": "Question?",
        "candidate_answer": "Native answer",
        "native_generated_token_count": 2,
        "native_mean_transition_logprob": -0.2,
        "action_effect": "helps",
    }
    values = {
        "multimodal_identity": -0.2,
        "text_only": -0.5,
        "multimodal_jpeg_q75": -0.3,
        "multimodal_gaussian_blur_r1": -0.4,
    }
    features = build_native_answer_features(
        [sample],
        [_score(condition, values[condition]) for condition in SCORE_CONDITIONS],
    )
    assert features[0]["identity_visual_delta"] == pytest.approx(0.3)
    assert features[0]["minimum_visual_delta"] == pytest.approx(0.1)
    assert features[0]["population_std_visual_delta"] > 0
    assert "groundtruth_response" not in features[0]


def test_summary_reports_effect_counts_without_training_selector() -> None:
    sample = {
        "haloquest_id": 1,
        "dataset_scope": "false_premise",
        "hallucination_type": "false premises",
        "image_type": "generated",
        "local_image_path": "image.png",
        "image_sha256": "abc",
        "question": "Question?",
        "candidate_answer": "Native answer",
        "native_generated_token_count": 2,
        "native_mean_transition_logprob": -0.2,
        "action_effect": "helps",
    }
    values = {
        "multimodal_identity": -0.2,
        "text_only": -0.5,
        "multimodal_jpeg_q75": -0.3,
        "multimodal_gaussian_blur_r1": -0.4,
    }
    features = build_native_answer_features(
        [sample],
        [_score(condition, values[condition]) for condition in SCORE_CONDITIONS],
    )
    summary = summarize_native_answer_features(features)
    assert summary["effect_counts"] == {"helps": 1}
    assert "overall_feature_analysis" in summary
    consistency = summary["generation_identity_score_consistency"]
    assert consistency["mean_absolute_difference"] == pytest.approx(0.0)
    assert consistency["pearson_correlation"] is None
