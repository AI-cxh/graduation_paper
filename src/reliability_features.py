"""Perturbation-stability reliability features for EXP-002."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from io import BytesIO
from statistics import mean, median, pstdev
from typing import Any

from PIL import Image, ImageFilter


RELIABILITY_PROTOCOL_VERSION = "exp002-perturbation-stability-v1"
PERTURBATIONS = ("jpeg_q75", "gaussian_blur_r1")
MULTIMODAL_CONDITIONS = ("multimodal_conflict", "multimodal_clean")


def apply_image_perturbation(
    image: Image.Image,
    perturbation: str,
) -> Image.Image:
    """Apply one deterministic, mild perturbation without changing image size."""

    image = image.convert("RGB")
    if perturbation == "jpeg_q75":
        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=75,
            subsampling=2,
            optimize=False,
            progressive=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    if perturbation == "gaussian_blur_r1":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    raise ValueError(f"Unknown perturbation: {perturbation}")


def _variant_features(
    identity_score: float,
    perturbed_scores: Mapping[str, float],
    text_only_score: float,
) -> dict[str, Any]:
    missing = set(PERTURBATIONS).difference(perturbed_scores)
    if missing:
        raise ValueError(f"Missing perturbations: {sorted(missing)}")
    deltas = {
        "identity": identity_score - text_only_score,
        **{
            perturbation: float(perturbed_scores[perturbation]) - text_only_score
            for perturbation in PERTURBATIONS
        },
    }
    values = list(deltas.values())
    identity_delta = deltas["identity"]
    identity_sign = (identity_delta > 0) - (identity_delta < 0)
    matching_signs = sum(
        ((value > 0) - (value < 0)) == identity_sign for value in values
    )
    return {
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
        "positive_variant_fraction": sum(value > 0 for value in values)
        / len(values),
        "identity_sign_consistency_fraction": matching_signs / len(values),
    }


def build_reliability_features(
    base_records: Iterable[Mapping[str, Any]],
    perturbation_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Align base and perturbed scores into contribution/stability features."""

    base: dict[tuple[int, str], Mapping[str, Any]] = {}
    for record in base_records:
        key = (int(record["image_id"]), str(record["condition"]))
        if key in base:
            raise ValueError(f"Duplicate base score key: {key}")
        base[key] = record

    perturbed: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for record in perturbation_records:
        condition = str(record["condition"])
        perturbation = str(record["perturbation"])
        if condition not in MULTIMODAL_CONDITIONS:
            raise ValueError(f"Unexpected perturbed condition: {condition}")
        if perturbation not in PERTURBATIONS:
            raise ValueError(f"Unexpected perturbation: {perturbation}")
        key = (int(record["image_id"]), condition, perturbation)
        if key in perturbed:
            raise ValueError(f"Duplicate perturbed score key: {key}")
        perturbed[key] = record

    image_ids = sorted({key[0] for key in perturbed})
    if not image_ids:
        raise ValueError("No perturbation score records supplied")
    results = []
    for image_id in image_ids:
        required_base = {
            (image_id, "multimodal_conflict"),
            (image_id, "text_only_conflict"),
            (image_id, "multimodal_clean"),
            (image_id, "text_only_clean"),
        }
        missing_base = required_base.difference(base)
        if missing_base:
            raise ValueError(
                f"image_id={image_id} missing base scores: {sorted(missing_base)}"
            )
        required_perturbations = {
            (image_id, condition, perturbation)
            for condition in MULTIMODAL_CONDITIONS
            for perturbation in PERTURBATIONS
        }
        missing_perturbations = required_perturbations.difference(perturbed)
        if missing_perturbations:
            raise ValueError(
                f"image_id={image_id} missing perturbation scores: "
                f"{sorted(missing_perturbations)}"
            )

        common = base[(image_id, "multimodal_conflict")]
        result: dict[str, Any] = {
            "image_id": image_id,
            "conflict_type": str(common["conflict_type"]),
            "protocol": RELIABILITY_PROTOCOL_VERSION,
        }
        for branch in ("conflict", "clean"):
            multimodal = base[(image_id, f"multimodal_{branch}")]
            text_only = base[(image_id, f"text_only_{branch}")]
            if multimodal["candidate_answer"] != text_only["candidate_answer"]:
                raise ValueError(
                    f"image_id={image_id} {branch} base candidate mismatch"
                )
            perturbation_scores = {}
            for perturbation in PERTURBATIONS:
                record = perturbed[
                    (image_id, f"multimodal_{branch}", perturbation)
                ]
                if record["candidate_answer"] != multimodal["candidate_answer"]:
                    raise ValueError(
                        f"image_id={image_id} {branch} candidate mismatch "
                        f"for {perturbation}"
                    )
                if (
                    "candidate_token_ids" in record
                    and "candidate_token_ids" in multimodal
                    and record["candidate_token_ids"]
                    != multimodal["candidate_token_ids"]
                ):
                    raise ValueError(
                        f"image_id={image_id} {branch} candidate token mismatch "
                        f"for {perturbation}"
                    )
                perturbation_scores[perturbation] = float(
                    record["candidate_mean_logprob"]
                )
            features = _variant_features(
                float(multimodal["candidate_mean_logprob"]),
                perturbation_scores,
                float(text_only["candidate_mean_logprob"]),
            )
            result.update(
                {f"{branch}_{key}": value for key, value in features.items()}
            )
        result["conflict_minus_clean_identity_visual_delta"] = (
            result["conflict_identity_visual_delta"]
            - result["clean_identity_visual_delta"]
        )
        result["conflict_minus_clean_minimum_visual_delta"] = (
            result["conflict_minimum_visual_delta"]
            - result["clean_minimum_visual_delta"]
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
    negatives = [
        score for score, label in zip(scores, labels, strict=True) if not label
    ]
    if not positives or not negatives:
        return None
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            favorable += float(positive > negative)
            favorable += 0.5 * float(positive == negative)
    return favorable / (len(positives) * len(negatives))


def _pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    mean_x = mean(values_x)
    mean_y = mean(values_y)
    numerator = sum(
        (value_x - mean_x) * (value_y - mean_y)
        for value_x, value_y in zip(values_x, values_y, strict=True)
    )
    denominator_x = sum((value - mean_x) ** 2 for value in values_x) ** 0.5
    denominator_y = sum((value - mean_y) ** 2 for value in values_y) ** 0.5
    denominator = denominator_x * denominator_y
    return numerator / denominator if denominator else None


def _feature_summary(features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "conflict_identity_visual_delta",
        "conflict_mean_visual_delta",
        "conflict_minimum_visual_delta",
        "conflict_population_std_visual_delta",
        "conflict_worst_drop_from_identity",
        "conflict_positive_variant_fraction",
        "conflict_identity_sign_consistency_fraction",
        "clean_identity_visual_delta",
        "clean_mean_visual_delta",
        "clean_minimum_visual_delta",
        "clean_population_std_visual_delta",
        "clean_worst_drop_from_identity",
        "clean_positive_variant_fraction",
        "clean_identity_sign_consistency_fraction",
        "conflict_minus_clean_identity_visual_delta",
        "conflict_minus_clean_minimum_visual_delta",
    )
    summary = {
        key: _describe([float(feature[key]) for feature in features])
        for key in keys
    }
    for branch in ("conflict", "clean"):
        summary[f"{branch}_visual_delta_by_variant"] = {
            variant: _describe(
                [
                    float(feature[f"{branch}_variant_visual_deltas"][variant])
                    for feature in features
                ]
            )
            for variant in ("identity", *PERTURBATIONS)
        }
    return summary


def _rule_alignment(
    features: Sequence[Mapping[str, Any]],
    evaluated_generation: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    generation = {
        (int(record["image_id"]), str(record["condition"])): record
        for record in evaluated_generation
        if record["condition"] in {"multimodal_conflict", "text_only_conflict"}
    }
    aligned = []
    transitions: Counter[str] = Counter()
    for feature in features:
        image_id = int(feature["image_id"])
        multimodal = generation.get((image_id, "multimodal_conflict"))
        text_only = generation.get((image_id, "text_only_conflict"))
        if multimodal is None or text_only is None:
            raise ValueError(f"Missing generation stance for image_id={image_id}")
        multimodal_rejects = multimodal["stance"] == "reject_premise"
        text_only_rejects = text_only["stance"] == "reject_premise"
        if multimodal_rejects and text_only_rejects:
            transition = "both_reject"
        elif multimodal_rejects:
            transition = "multimodal_only_reject"
        elif text_only_rejects:
            transition = "text_only_only_reject"
        else:
            transition = "neither_reject"
        transitions[transition] += 1
        aligned.append(
            (feature, multimodal_rejects, multimodal_rejects and not text_only_rejects)
        )

    score_definitions = {
        "contribution_identity": "conflict_identity_visual_delta",
        "worst_case_contribution": "conflict_minimum_visual_delta",
        "negative_instability_std": "conflict_population_std_visual_delta",
        "negative_worst_drop": "conflict_worst_drop_from_identity",
        "sign_consistency": "conflict_identity_sign_consistency_fraction",
        "conflict_minus_clean_worst_case": (
            "conflict_minus_clean_minimum_visual_delta"
        ),
    }
    result: dict[str, Any] = {
        "pair_count": len(aligned),
        "transition_counts": dict(sorted(transitions.items())),
        "feature_auroc": {},
    }
    for name, key in score_definitions.items():
        scores = [float(item[0][key]) for item in aligned]
        if name.startswith("negative_"):
            scores = [-score for score in scores]
        result["feature_auroc"][name] = {
            "multimodal_explicit_rejection": _binary_roc_auc(
                scores,
                [item[1] for item in aligned],
            ),
            "multimodal_only_rejection_transition": _binary_roc_auc(
                scores,
                [item[2] for item in aligned],
            ),
        }
    return result


def summarize_reliability_features(
    features: Sequence[Mapping[str, Any]],
    evaluated_generation: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate stability features without treating proxy labels as accuracy."""

    if not features:
        raise ValueError("No reliability features supplied")
    conflict_types = sorted(
        {str(feature["conflict_type"]) for feature in features}
    )
    identity = [
        float(feature["conflict_identity_visual_delta"]) for feature in features
    ]
    instability = [
        float(feature["conflict_population_std_visual_delta"])
        for feature in features
    ]
    positive_identity = sum(value > 0 for value in identity)
    sign_failure = sum(
        float(feature["conflict_identity_visual_delta"]) > 0
        and float(feature["conflict_minimum_visual_delta"]) <= 0
        for feature in features
    )
    summary: dict[str, Any] = {
        "protocol": RELIABILITY_PROTOCOL_VERSION,
        "scope": (
            "Reliability is represented by stability of identical reference "
            "candidate visual deltas under deterministic mild image "
            "perturbations. These features are diagnostics, not calibrated "
            "probabilities or semantic accuracy."
        ),
        "pair_count": len(features),
        "conflict_type_counts": dict(
            sorted(Counter(str(item["conflict_type"]) for item in features).items())
        ),
        "overall": _feature_summary(features),
        "by_conflict_type": {
            conflict_type: _feature_summary(
                [
                    feature
                    for feature in features
                    if feature["conflict_type"] == conflict_type
                ]
            )
            for conflict_type in conflict_types
        },
        "contribution_reliability_diagnostic": {
            "positive_identity_contribution_count": positive_identity,
            "positive_identity_but_nonpositive_worst_case_count": sign_failure,
            "rate_among_positive_identity": (
                sign_failure / positive_identity if positive_identity else None
            ),
            "pearson_identity_contribution_vs_instability_std": _pearson(
                identity,
                instability,
            ),
        },
    }
    if evaluated_generation is not None:
        summary["rule_proxy_alignment"] = _rule_alignment(
            features,
            evaluated_generation,
        )
    return summary
