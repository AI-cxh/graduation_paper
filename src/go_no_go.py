"""Leakage-controlled small-sample selector evaluation for EXP-004."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SelectorData:
    """Aligned features, target, and action utilities."""

    image_ids: NDArray[np.int64]
    conflict_types: tuple[str, ...]
    features: dict[str, FloatArray]
    target: BoolArray
    native_utility: FloatArray
    intervention_utility: FloatArray


def prepare_selector_data(
    records: Sequence[Mapping[str, Any]],
    feature_sets: Mapping[str, Sequence[str]],
    *,
    native_utility_field: str = "baseline_conflict_rejection_proxy",
    intervention_utility_field: str = "intervention_conflict_rejection_proxy",
) -> SelectorData:
    """Validate utility records and create aligned model arrays."""

    if not records:
        raise ValueError("No utility records supplied")
    image_ids = np.asarray([int(record["image_id"]) for record in records])
    if len(set(image_ids.tolist())) != len(image_ids):
        raise ValueError("Duplicate image_id in utility records")
    native = np.asarray(
        [bool(record[native_utility_field]) for record in records],
        dtype=float,
    )
    intervention = np.asarray(
        [
            bool(record[intervention_utility_field])
            for record in records
        ],
        dtype=float,
    )
    target = np.logical_and(intervention == 1.0, native == 0.0)
    if target.sum() < 2 or (~target).sum() < 2:
        raise ValueError("Target needs at least two examples in each class")

    matrices: dict[str, FloatArray] = {}
    for name, fields in feature_sets.items():
        if not fields:
            raise ValueError(f"Feature set {name!r} is empty")
        matrix = np.asarray(
            [
                [float(record[field]) for field in fields]
                for record in records
            ],
            dtype=float,
        )
        if not np.isfinite(matrix).all():
            raise ValueError(f"Feature set {name!r} contains non-finite values")
        matrices[name] = matrix

    return SelectorData(
        image_ids=image_ids,
        conflict_types=tuple(str(record["conflict_type"]) for record in records),
        features=matrices,
        target=target,
        native_utility=native,
        intervention_utility=intervention,
    )


def policy_utility(
    select_intervention: BoolArray,
    native_utility: FloatArray,
    intervention_utility: FloatArray,
) -> FloatArray:
    """Return per-example utility under a binary intervention policy."""

    return np.where(
        select_intervention,
        intervention_utility,
        native_utility,
    )


def choose_policy_threshold(
    probabilities: FloatArray,
    native_utility: FloatArray,
    intervention_utility: FloatArray,
) -> float:
    """Choose a threshold by training-fold utility, preferring fewer actions."""

    candidates = np.concatenate(
        (
            np.asarray([0.0]),
            np.unique(probabilities),
            np.asarray([np.nextafter(1.0, 2.0)]),
        )
    )
    scored: list[tuple[float, float, float]] = []
    for threshold in candidates:
        selected = probabilities >= threshold
        utility = policy_utility(
            selected,
            native_utility,
            intervention_utility,
        ).mean()
        scored.append((float(utility), -float(selected.mean()), float(threshold)))
    return max(scored)[2]


def _new_classifier(c_value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="liblinear",
    )


def _inner_oof_probabilities(
    features: FloatArray,
    target: BoolArray,
    *,
    c_value: float,
    folds: int,
    seed: int,
) -> FloatArray:
    probabilities = np.empty(len(target), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_indices, validation_indices in splitter.split(features, target):
        scaler = StandardScaler().fit(features[train_indices])
        classifier = _new_classifier(c_value, seed)
        classifier.fit(
            scaler.transform(features[train_indices]),
            target[train_indices],
        )
        probabilities[validation_indices] = classifier.predict_proba(
            scaler.transform(features[validation_indices])
        )[:, 1]
    return probabilities


def _select_inner_parameters(
    features: FloatArray,
    target: BoolArray,
    native_utility: FloatArray,
    intervention_utility: FloatArray,
    *,
    c_grid: Sequence[float],
    folds: int,
    seed: int,
) -> tuple[float, float]:
    """Tune C by inner OOF AUPRC, then threshold by inner policy utility."""

    class_count = int(min(target.sum(), (~target).sum()))
    effective_folds = min(folds, class_count)
    if effective_folds < 2:
        raise ValueError("Inner training split has too few examples per class")
    candidates: list[tuple[float, float, FloatArray]] = []
    for c_value in c_grid:
        probabilities = _inner_oof_probabilities(
            features,
            target,
            c_value=float(c_value),
            folds=effective_folds,
            seed=seed,
        )
        candidates.append(
            (
                float(average_precision_score(target, probabilities)),
                -float(c_value),
                probabilities,
            )
        )
    _, negative_c, best_probabilities = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    best_c = -negative_c
    threshold = choose_policy_threshold(
        best_probabilities,
        native_utility,
        intervention_utility,
    )
    return best_c, threshold


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _repeat_metrics(
    target: BoolArray,
    probabilities: FloatArray,
    selected: BoolArray,
    native_utility: FloatArray,
    intervention_utility: FloatArray,
) -> dict[str, float]:
    selected_utility = policy_utility(
        selected,
        native_utility,
        intervention_utility,
    )
    positive = target
    harmful = np.logical_and(native_utility == 1.0, intervention_utility == 0.0)
    return {
        "auroc": float(roc_auc_score(target, probabilities)),
        "auprc": float(average_precision_score(target, probabilities)),
        "policy_utility": float(selected_utility.mean()),
        "selection_rate": float(selected.mean()),
        "intervention_benefit_capture_rate": float(selected[positive].mean()),
        "native_only_protection_rate": (
            float((~selected[harmful]).mean()) if harmful.any() else 1.0
        ),
    }


def evaluate_feature_set(
    features: FloatArray,
    target: BoolArray,
    native_utility: FloatArray,
    intervention_utility: FloatArray,
    *,
    feature_names: Sequence[str],
    outer_folds: int,
    repeats: int,
    inner_folds: int,
    c_grid: Sequence[float],
    seed: int,
) -> tuple[dict[str, Any], FloatArray, BoolArray]:
    """Run repeated nested CV, keeping all preprocessing inside training folds."""

    class_count = int(min(target.sum(), (~target).sum()))
    if outer_folds > class_count:
        raise ValueError("outer_folds exceeds the minority-class count")
    all_probabilities = np.empty((repeats, len(target)), dtype=float)
    all_selected = np.empty((repeats, len(target)), dtype=bool)
    repeat_results: list[dict[str, float]] = []
    coefficients: list[list[float]] = []
    fold_parameters: list[dict[str, Any]] = []

    for repeat in range(repeats):
        repeat_seed = seed + repeat * 1000
        splitter = StratifiedKFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        probabilities = np.empty(len(target), dtype=float)
        selected = np.empty(len(target), dtype=bool)
        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(features, target)
        ):
            inner_seed = repeat_seed + fold + 1
            c_value, threshold = _select_inner_parameters(
                features[train_indices],
                target[train_indices],
                native_utility[train_indices],
                intervention_utility[train_indices],
                c_grid=c_grid,
                folds=inner_folds,
                seed=inner_seed,
            )
            scaler = StandardScaler().fit(features[train_indices])
            classifier = _new_classifier(c_value, inner_seed)
            classifier.fit(
                scaler.transform(features[train_indices]),
                target[train_indices],
            )
            fold_probabilities = classifier.predict_proba(
                scaler.transform(features[test_indices])
            )[:, 1]
            probabilities[test_indices] = fold_probabilities
            selected[test_indices] = fold_probabilities >= threshold
            coefficients.append(classifier.coef_[0].astype(float).tolist())
            fold_parameters.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "c": c_value,
                    "threshold": threshold,
                    "train_count": len(train_indices),
                    "test_count": len(test_indices),
                }
            )
        all_probabilities[repeat] = probabilities
        all_selected[repeat] = selected
        repeat_results.append(
            _repeat_metrics(
                target,
                probabilities,
                selected,
                native_utility,
                intervention_utility,
            )
        )

    metric_names = tuple(repeat_results[0])
    coefficient_array = np.asarray(coefficients)
    aggregate_probabilities = all_probabilities.mean(axis=0)
    aggregate_selected = all_selected.mean(axis=0) >= 0.5
    summary = {
        "feature_names": list(feature_names),
        "repeat_metric_distribution": {
            metric: _metric_summary(
                [result[metric] for result in repeat_results]
            )
            for metric in metric_names
        },
        "aggregate_mean_oof_probability": {
            "auroc": float(roc_auc_score(target, aggregate_probabilities)),
            "auprc": float(
                average_precision_score(target, aggregate_probabilities)
            ),
        },
        "standardized_coefficient_distribution": {
            feature_name: _metric_summary(coefficient_array[:, index].tolist())
            for index, feature_name in enumerate(feature_names)
        },
        "fold_parameters": fold_parameters,
        "repeat_metrics": repeat_results,
        "majority_action_policy": {
            "utility": float(
                policy_utility(
                    aggregate_selected,
                    native_utility,
                    intervention_utility,
                ).mean()
            ),
            "selection_rate": float(aggregate_selected.mean()),
        },
    }
    return summary, all_probabilities, all_selected


def _bootstrap_paired_difference(
    first: FloatArray,
    second: FloatArray,
    target: BoolArray,
    *,
    metric: str,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    differences: list[float] = []
    n_samples = len(target)
    for _ in range(resamples):
        indices = generator.integers(0, n_samples, size=n_samples)
        if metric == "auroc":
            sampled_target = target[indices]
            if sampled_target.all() or (~sampled_target).all():
                continue
            difference = roc_auc_score(
                sampled_target,
                first[indices],
            ) - roc_auc_score(sampled_target, second[indices])
        elif metric == "mean":
            difference = float((first[indices] - second[indices]).mean())
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
        differences.append(float(difference))
    array = np.asarray(differences)
    return {
        "point_difference": (
            float(
                roc_auc_score(target, first) - roc_auc_score(target, second)
            )
            if metric == "auroc"
            else float((first - second).mean())
        ),
        "bootstrap_95_percent_interval": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "valid_resamples": len(array),
    }


def _bootstrap_metric(
    values: FloatArray,
    target: BoolArray,
    *,
    metric: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Return a descriptive sample-bootstrap interval for one metric."""

    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    n_samples = len(target)
    for _ in range(resamples):
        indices = generator.integers(0, n_samples, size=n_samples)
        sampled_target = target[indices]
        if metric in {"auroc", "auprc"} and (
            sampled_target.all() or (~sampled_target).all()
        ):
            continue
        if metric == "auroc":
            estimate = roc_auc_score(sampled_target, values[indices])
        elif metric == "auprc":
            estimate = average_precision_score(
                sampled_target,
                values[indices],
            )
        elif metric == "mean":
            estimate = float(values[indices].mean())
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
        estimates.append(float(estimate))
    array = np.asarray(estimates)
    if metric == "auroc":
        point = float(roc_auc_score(target, values))
    elif metric == "auprc":
        point = float(average_precision_score(target, values))
    else:
        point = float(values.mean())
    return {
        "point_estimate": point,
        "bootstrap_95_percent_interval": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "valid_resamples": len(array),
    }


def run_go_no_go(
    data: SelectorData,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    outer_folds: int,
    repeats: int,
    inner_folds: int,
    c_grid: Sequence[float],
    bootstrap_resamples: int,
    seed: int,
    primary_feature_set: str,
    comparator_feature_set: str,
    minimum_mean_auroc_gain: float,
    minimum_mean_policy_gain_over_always_intervention: float,
    minimum_repeat_auroc_win_fraction: float,
    target_name: str = "intervention_only_rule_proxy_success",
    protocol: str = "exp004-repeated-nested-stratified-cv-v1",
    scope: str | None = None,
    oracle_reference_name: str = "oracle_rule_proxy_utility",
    native_utility_output_name: str = "native_rule_proxy_utility",
    intervention_utility_output_name: str = "intervention_rule_proxy_utility",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate all feature sets and apply predeclared conditional-Go criteria."""

    model_summaries: dict[str, Any] = {}
    probabilities: dict[str, FloatArray] = {}
    selections: dict[str, BoolArray] = {}
    for name, fields in feature_sets.items():
        summary, model_probabilities, model_selections = evaluate_feature_set(
            data.features[name],
            data.target,
            data.native_utility,
            data.intervention_utility,
            feature_names=fields,
            outer_folds=outer_folds,
            repeats=repeats,
            inner_folds=inner_folds,
            c_grid=c_grid,
            seed=seed,
        )
        model_summaries[name] = summary
        probabilities[name] = model_probabilities
        selections[name] = model_selections

    for model_index, name in enumerate(feature_sets):
        aggregate_probability = probabilities[name].mean(axis=0)
        expected_policy_utility = policy_utility(
            selections[name],
            data.native_utility[None, :],
            data.intervention_utility[None, :],
        ).mean(axis=0)
        interval_seed = seed + 800_000 + model_index * 10
        model_summaries[name]["descriptive_sample_bootstrap"] = {
            "aggregate_oof_auroc": _bootstrap_metric(
                aggregate_probability,
                data.target,
                metric="auroc",
                resamples=bootstrap_resamples,
                seed=interval_seed,
            ),
            "aggregate_oof_auprc": _bootstrap_metric(
                aggregate_probability,
                data.target,
                metric="auprc",
                resamples=bootstrap_resamples,
                seed=interval_seed + 1,
            ),
            "repeat_averaged_policy_utility": _bootstrap_metric(
                expected_policy_utility,
                data.target,
                metric="mean",
                resamples=bootstrap_resamples,
                seed=interval_seed + 2,
            ),
            "policy_gain_over_always_native": _bootstrap_paired_difference(
                expected_policy_utility,
                data.native_utility,
                data.target,
                metric="mean",
                resamples=bootstrap_resamples,
                seed=interval_seed + 3,
            ),
            "policy_gain_over_always_intervention": (
                _bootstrap_paired_difference(
                    expected_policy_utility,
                    data.intervention_utility,
                    data.target,
                    metric="mean",
                    resamples=bootstrap_resamples,
                    seed=interval_seed + 4,
                )
            ),
            "policy_regret_to_rule_proxy_oracle": (
                _bootstrap_paired_difference(
                    expected_policy_utility,
                    np.maximum(
                        data.native_utility,
                        data.intervention_utility,
                    ),
                    data.target,
                    metric="mean",
                    resamples=bootstrap_resamples,
                    seed=interval_seed + 5,
                )
            ),
        }

    comparisons = {}
    names = list(feature_sets)
    comparison_index = 0
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1 :]:
            first_probability = probabilities[first_name].mean(axis=0)
            second_probability = probabilities[second_name].mean(axis=0)
            first_expected_utility = policy_utility(
                selections[first_name],
                data.native_utility[None, :],
                data.intervention_utility[None, :],
            ).mean(axis=0)
            second_expected_utility = policy_utility(
                selections[second_name],
                data.native_utility[None, :],
                data.intervention_utility[None, :],
            ).mean(axis=0)
            key = f"{first_name}_minus_{second_name}"
            comparisons[key] = {
                "aggregate_oof_auroc": _bootstrap_paired_difference(
                    first_probability,
                    second_probability,
                    data.target,
                    metric="auroc",
                    resamples=bootstrap_resamples,
                    seed=seed + 900_000 + comparison_index,
                ),
                "repeat_averaged_policy_utility": (
                    _bootstrap_paired_difference(
                        first_expected_utility,
                        second_expected_utility,
                        data.target,
                        metric="mean",
                        resamples=bootstrap_resamples,
                        seed=seed + 910_000 + comparison_index,
                    )
                ),
            }
            comparison_index += 1

    primary_metrics = model_summaries[primary_feature_set][
        "repeat_metric_distribution"
    ]
    comparator_repeats = model_summaries[comparator_feature_set][
        "repeat_metrics"
    ]
    primary_repeats = model_summaries[primary_feature_set]["repeat_metrics"]
    mean_auroc_gain = (
        primary_metrics["auroc"]["mean"]
        - model_summaries[comparator_feature_set][
            "repeat_metric_distribution"
        ]["auroc"]["mean"]
    )
    always_intervention = float(data.intervention_utility.mean())
    mean_policy_gain = (
        primary_metrics["policy_utility"]["mean"] - always_intervention
    )
    repeat_win_fraction = float(
        np.mean(
            [
                primary["auroc"] > comparator["auroc"]
                for primary, comparator in zip(
                    primary_repeats,
                    comparator_repeats,
                    strict=True,
                )
            ]
        )
    )
    checks = {
        "mean_auroc_gain": {
            "value": mean_auroc_gain,
            "threshold": minimum_mean_auroc_gain,
            "pass": mean_auroc_gain >= minimum_mean_auroc_gain,
        },
        "mean_policy_gain_over_always_intervention": {
            "value": mean_policy_gain,
            "threshold": minimum_mean_policy_gain_over_always_intervention,
            "pass": (
                mean_policy_gain
                >= minimum_mean_policy_gain_over_always_intervention
            ),
        },
        "repeat_auroc_win_fraction": {
            "value": repeat_win_fraction,
            "threshold": minimum_repeat_auroc_win_fraction,
            "pass": repeat_win_fraction >= minimum_repeat_auroc_win_fraction,
        },
    }
    verdict = "conditional_go" if all(
        check["pass"] for check in checks.values()
    ) else "no_go"

    prediction_records = []
    for index, image_id in enumerate(data.image_ids):
        record: dict[str, Any] = {
            "image_id": int(image_id),
            "conflict_type": data.conflict_types[index],
            target_name: bool(data.target[index]),
            native_utility_output_name: int(data.native_utility[index]),
            intervention_utility_output_name: int(data.intervention_utility[index]),
        }
        for name in names:
            record[name] = {
                "mean_oof_probability": float(
                    probabilities[name][:, index].mean()
                ),
                "oof_probability_standard_deviation": float(
                    probabilities[name][:, index].std(ddof=1)
                ),
                "intervention_selection_rate": float(
                    selections[name][:, index].mean()
                ),
            }
        prediction_records.append(record)

    oracle = np.maximum(data.native_utility, data.intervention_utility)
    default_scope = (
        "Exploratory small-sample evaluation using an automatic explicit-"
        "rejection proxy. Bootstrap intervals are descriptive and do not "
        "remove uncertainty from proxy-label error or repeated-CV dependence."
    )
    summary = {
        "protocol": protocol,
        "scope": scope or default_scope,
        "sample": {
            "count": len(data.target),
            "target_name": target_name,
            "positive_target_count": int(data.target.sum()),
            "positive_intervention_only_count": int(data.target.sum()),
            "negative_count": int((~data.target).sum()),
            "native_only_harm_count": int(
                np.logical_and(
                    data.native_utility == 1.0,
                    data.intervention_utility == 0.0,
                ).sum()
            ),
        },
        "policy_references": {
            "always_native_utility": float(data.native_utility.mean()),
            "always_intervention_utility": always_intervention,
            oracle_reference_name: float(oracle.mean()),
        },
        "validation": {
            "outer_folds": outer_folds,
            "repeats": repeats,
            "inner_folds": inner_folds,
            "c_grid": [float(value) for value in c_grid],
            "bootstrap_resamples": bootstrap_resamples,
            "leakage_controls": [
                "identical outer and inner fold assignments across feature sets",
                "standardization fitted only on each outer training fold",
                "C selected by inner out-of-fold AUPRC",
                "policy threshold selected by inner training-fold utility",
                "outer test fold used only for evaluation",
            ],
        },
        "feature_set_results": model_summaries,
        "paired_bootstrap_comparisons": comparisons,
        "go_no_go": {
            "verdict": verdict,
            "primary_feature_set": primary_feature_set,
            "comparator_feature_set": comparator_feature_set,
            "criteria": checks,
            "meaning": (
                "A conditional Go only authorizes a larger semantic-label and "
                "held-out evaluation; it is not a deployability claim."
            ),
        },
    }
    return summary, prediction_records
