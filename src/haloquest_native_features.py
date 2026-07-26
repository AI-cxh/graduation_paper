"""Native-answer contribution and reliability features for HaloQuest."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import sqrt
from random import Random
from statistics import mean, median, pstdev
from typing import Any


PROTOCOL_VERSION = "exp007-native-answer-support-stability-v1"
SCORE_CONDITIONS = (
    "multimodal_identity",
    "text_only",
    "multimodal_jpeg_q75",
    "multimodal_gaussian_blur_r1",
)
FEATURE_KEYS = (
    "native_mean_transition_logprob",
    "native_generated_token_count",
    "identity_candidate_mean_logprob",
    "text_only_candidate_mean_logprob",
    "identity_visual_delta",
    "mean_visual_delta",
    "minimum_visual_delta",
    "population_std_visual_delta",
    "range_visual_delta",
    "worst_drop_from_identity",
    "maximum_absolute_change_from_identity",
    "positive_variant_fraction",
    "identity_sign_consistency_fraction",
)


def build_audited_samples(
    *,
    scope: str,
    manifest_records: Iterable[Mapping[str, Any]],
    prediction_records: Iterable[Mapping[str, Any]],
    audit: Mapping[str, Any],
    accepted_effects: set[str],
) -> list[dict[str, Any]]:
    """Align available images, native generations, and frozen AI effects."""

    manifest = {
        int(record["haloquest_id"]): record
        for record in manifest_records
        if record["download_status"] == "available"
    }
    native = {
        int(record["haloquest_id"]): record
        for record in prediction_records
        if record["action"] == "native_prompt"
    }
    effect_by_id: dict[int, str] = {}
    for effect, values in audit["effect_groups"].items():
        for value in values:
            haloquest_id = int(value)
            if haloquest_id in effect_by_id:
                raise ValueError(f"Duplicate audited ID: {haloquest_id}")
            effect_by_id[haloquest_id] = str(effect)

    samples = []
    for haloquest_id, effect in sorted(effect_by_id.items()):
        if effect not in accepted_effects:
            continue
        if haloquest_id not in manifest:
            raise ValueError(f"Audited ID missing available manifest: {haloquest_id}")
        if haloquest_id not in native:
            raise ValueError(f"Audited ID missing native prediction: {haloquest_id}")
        manifest_record = manifest[haloquest_id]
        prediction = native[haloquest_id]
        candidate = str(prediction["generated_text"]).strip()
        if not candidate:
            raise ValueError(f"Empty native candidate: {haloquest_id}")
        samples.append(
            {
                "haloquest_id": haloquest_id,
                "dataset_scope": scope,
                "hallucination_type": str(
                    manifest_record["hallucination_type"]
                ),
                "image_type": str(manifest_record["image_type"]),
                "local_image_path": str(manifest_record["local_image_path"]),
                "image_sha256": str(manifest_record["image_sha256"]),
                "question": str(manifest_record["question"]),
                "candidate_answer": candidate,
                "native_generated_token_count": int(
                    prediction["generated_token_count"]
                ),
                "native_mean_transition_logprob": float(
                    prediction["mean_transition_logprob"]
                ),
                "action_effect": effect,
            }
        )
    return samples


def build_native_answer_features(
    samples: Iterable[Mapping[str, Any]],
    score_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Align four score conditions into no-reference contribution features."""

    sample_by_id = {int(sample["haloquest_id"]): sample for sample in samples}
    scores: dict[tuple[int, str], Mapping[str, Any]] = {}
    for record in score_records:
        key = (int(record["haloquest_id"]), str(record["condition"]))
        if key in scores:
            raise ValueError(f"Duplicate native-answer score: {key}")
        scores[key] = record

    results = []
    for haloquest_id, sample in sorted(sample_by_id.items()):
        required = {
            (haloquest_id, condition) for condition in SCORE_CONDITIONS
        }
        missing = required.difference(scores)
        if missing:
            raise ValueError(
                f"HaloQuest ID {haloquest_id} missing scores: {sorted(missing)}"
            )
        aligned = {
            condition: scores[(haloquest_id, condition)]
            for condition in SCORE_CONDITIONS
        }
        candidates = {
            str(record["candidate_answer"]) for record in aligned.values()
        }
        if candidates != {str(sample["candidate_answer"])}:
            raise ValueError(f"Candidate mismatch for HaloQuest ID {haloquest_id}")
        token_ids = {
            tuple(int(token) for token in record["candidate_token_ids"])
            for record in aligned.values()
        }
        if len(token_ids) != 1:
            raise ValueError(
                f"Candidate token mismatch for HaloQuest ID {haloquest_id}"
            )

        identity = float(
            aligned["multimodal_identity"]["candidate_mean_logprob"]
        )
        text_only = float(aligned["text_only"]["candidate_mean_logprob"])
        variant_scores = {
            "identity": identity,
            "jpeg_q75": float(
                aligned["multimodal_jpeg_q75"]["candidate_mean_logprob"]
            ),
            "gaussian_blur_r1": float(
                aligned["multimodal_gaussian_blur_r1"][
                    "candidate_mean_logprob"
                ]
            ),
        }
        deltas = {
            variant: score - text_only
            for variant, score in variant_scores.items()
        }
        values = list(deltas.values())
        identity_delta = deltas["identity"]
        identity_sign = (identity_delta > 0) - (identity_delta < 0)
        matching_signs = sum(
            ((value > 0) - (value < 0)) == identity_sign
            for value in values
        )
        result = dict(sample)
        result.update(
            {
                "protocol": PROTOCOL_VERSION,
                "candidate_token_count": len(next(iter(token_ids))),
                "identity_candidate_mean_logprob": identity,
                "text_only_candidate_mean_logprob": text_only,
                "variant_candidate_mean_logprobs": variant_scores,
                "variant_visual_deltas": deltas,
                "identity_visual_delta": identity_delta,
                "mean_visual_delta": mean(values),
                "minimum_visual_delta": min(values),
                "maximum_visual_delta": max(values),
                "population_std_visual_delta": pstdev(values),
                "range_visual_delta": max(values) - min(values),
                "worst_drop_from_identity": identity_delta - min(values),
                "maximum_absolute_change_from_identity": max(
                    abs(value - identity_delta) for value in values
                ),
                "positive_variant_fraction": sum(
                    value > 0 for value in values
                )
                / len(values),
                "identity_sign_consistency_fraction": (
                    matching_signs / len(values)
                ),
            }
        )
        results.append(result)
    return results


def _describe(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "population_std": pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _binary_roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = sorted(
        score for score, label in zip(scores, labels, strict=True) if not label
    )
    if not positives or not negatives:
        return None
    favorable = 0.0
    for positive in positives:
        favorable += bisect_left(negatives, positive)
        favorable += 0.5 * (
            bisect_right(negatives, positive)
            - bisect_left(negatives, positive)
        )
    return favorable / (len(positives) * len(negatives))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_binary_roc_auc(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any] | None:
    """Return a deterministic class-stratified descriptive AUROC interval."""

    positives = [
        float(score)
        for score, label in zip(scores, labels, strict=True)
        if label
    ]
    negatives = [
        float(score)
        for score, label in zip(scores, labels, strict=True)
        if not label
    ]
    observed = _binary_roc_auc(scores, labels)
    if observed is None:
        return None
    if resamples <= 0:
        return {
            "observed": observed,
            "resamples": 0,
            "stratified_95_percent_interval": None,
        }
    rng = Random(seed)
    bootstrapped = []
    for _ in range(resamples):
        positive_sample = [
            positives[rng.randrange(len(positives))]
            for _ in range(len(positives))
        ]
        negative_sample = [
            negatives[rng.randrange(len(negatives))]
            for _ in range(len(negatives))
        ]
        value = _binary_roc_auc(
            positive_sample + negative_sample,
            [True] * len(positive_sample) + [False] * len(negative_sample),
        )
        if value is None:
            raise AssertionError("Stratified bootstrap lost a class")
        bootstrapped.append(value)
    return {
        "observed": observed,
        "resamples": resamples,
        "stratified_95_percent_interval": [
            _quantile(bootstrapped, 0.025),
            _quantile(bootstrapped, 0.975),
        ],
    }


def _pearson_correlation(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    if len(left) != len(right):
        raise ValueError("Correlation inputs must have the same length")
    if len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    denominator = sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def _feature_analysis(
    features: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    effects = [str(feature["action_effect"]) for feature in features]
    return {
        key: {
            "overall": _describe([float(feature[key]) for feature in features]),
            "by_effect": {
                effect: _describe(
                    [
                        float(feature[key])
                        for feature in features
                        if feature["action_effect"] == effect
                    ]
                )
                for effect in ("helps", "harms", "tie")
                if any(feature["action_effect"] == effect for feature in features)
            },
            "auroc_help_vs_rest": _binary_roc_auc(
                [float(feature[key]) for feature in features],
                [effect == "helps" for effect in effects],
            ),
            "auroc_harm_vs_rest": _binary_roc_auc(
                [float(feature[key]) for feature in features],
                [effect == "harms" for effect in effects],
            ),
            "auroc_help_vs_rest_bootstrap": bootstrap_binary_roc_auc(
                [float(feature[key]) for feature in features],
                [effect == "helps" for effect in effects],
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + feature_index * 2,
            ),
            "auroc_harm_vs_rest_bootstrap": bootstrap_binary_roc_auc(
                [float(feature[key]) for feature in features],
                [effect == "harms" for effect in effects],
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + feature_index * 2 + 1,
            ),
        }
        for feature_index, key in enumerate(FEATURE_KEYS)
    }


def summarize_native_answer_features(
    features: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 0,
    bootstrap_seed: int = 20260726,
) -> dict[str, Any]:
    """Summarize exploratory associations without training a selector."""

    if not features:
        raise ValueError("No EXP-007 features supplied")
    scopes = sorted({str(feature["dataset_scope"]) for feature in features})
    generation_scores = [
        float(feature["native_mean_transition_logprob"]) for feature in features
    ]
    identity_scores = [
        float(feature["identity_candidate_mean_logprob"]) for feature in features
    ]
    generation_identity_differences = [
        generation - identity
        for generation, identity in zip(
            generation_scores, identity_scores, strict=True
        )
    ]
    return {
        "protocol": PROTOCOL_VERSION,
        "scope": (
            "Exploratory no-reference feature diagnostic on AI-audited, "
            "action-change-enriched samples. The candidate is the first-pass "
            "native response; official reference answers are labels only and "
            "are not used in any feature."
        ),
        "sample_count": len(features),
        "effect_counts": dict(
            sorted(Counter(str(item["action_effect"]) for item in features).items())
        ),
        "dataset_scope_counts": dict(
            sorted(Counter(str(item["dataset_scope"]) for item in features).items())
        ),
        "generation_identity_score_consistency": {
            "purpose": (
                "Checks whether first-pass generation transition scores can "
                "replace a separate identity-image teacher-forced pass."
            ),
            "mean_absolute_difference": mean(
                abs(value) for value in generation_identity_differences
            ),
            "maximum_absolute_difference": max(
                abs(value) for value in generation_identity_differences
            ),
            "pearson_correlation": _pearson_correlation(
                generation_scores, identity_scores
            ),
        },
        "bootstrap": {
            "method": "class-stratified sample bootstrap",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "interpretation": (
                "Descriptive uncertainty only; feature directions and samples "
                "were not held out for confirmatory inference."
            ),
        },
        "overall_feature_analysis": _feature_analysis(
            features,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "by_dataset_scope": {
            scope: {
                "sample_count": sum(
                    feature["dataset_scope"] == scope for feature in features
                ),
                "effect_counts": dict(
                    sorted(
                        Counter(
                            str(feature["action_effect"])
                            for feature in features
                            if feature["dataset_scope"] == scope
                        ).items()
                    )
                ),
                "feature_analysis": _feature_analysis(
                    [
                        feature
                        for feature in features
                        if feature["dataset_scope"] == scope
                    ],
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed + 1000 * (scope_index + 1),
                ),
            }
            for scope_index, scope in enumerate(scopes)
        },
    }
