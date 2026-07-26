"""Leakage-controlled selector evaluation for HaloQuest EXP-008."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class HaloQuestSelectorData:
    """Aligned no-reference features, groups, targets, and action utilities."""

    sample_ids: IntArray
    scopes: NDArray[np.str_]
    groups: NDArray[np.str_]
    effects: NDArray[np.str_]
    target: BoolArray
    intervention_utility: FloatArray
    features: dict[str, FloatArray]


@dataclass(frozen=True)
class EvaluationArrays:
    probabilities: FloatArray
    selections: BoolArray


@dataclass(frozen=True)
class FrozenSelector:
    """A full-data selector whose threshold uses repeated grouped OOF data."""

    scaler: StandardScaler
    model: LogisticRegression
    threshold: float
    mean_training_oof_probability: FloatArray

    def predict_probability(self, features: FloatArray) -> FloatArray:
        return self.model.predict_proba(self.scaler.transform(features))[:, 1]


def prepare_haloquest_selector_data(
    records: Sequence[Mapping[str, Any]],
    feature_sets: Mapping[str, Sequence[str]],
    effect_utilities: Mapping[str, float],
) -> HaloQuestSelectorData:
    """Validate EXP-007 records and align EXP-008 arrays."""

    if not records:
        raise ValueError("No EXP-007 features supplied")
    sample_ids = np.asarray(
        [int(record["haloquest_id"]) for record in records], dtype=np.int64
    )
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("Duplicate HaloQuest IDs in EXP-008 input")
    effects = np.asarray(
        [str(record["action_effect"]) for record in records], dtype=str
    )
    unknown_effects = set(effects.tolist()).difference(effect_utilities)
    if unknown_effects:
        raise ValueError(f"Unknown action effects: {sorted(unknown_effects)}")
    scopes = np.asarray(
        [str(record["dataset_scope"]) for record in records], dtype=str
    )
    groups = np.asarray(
        [str(record["image_sha256"]) for record in records], dtype=str
    )
    if any(not value for value in groups):
        raise ValueError("Empty image hash in EXP-008 input")

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
        matrices[str(name)] = matrix

    return HaloQuestSelectorData(
        sample_ids=sample_ids,
        scopes=scopes,
        groups=groups,
        effects=effects,
        target=effects == "helps",
        intervention_utility=np.asarray(
            [float(effect_utilities[effect]) for effect in effects],
            dtype=float,
        ),
        features=matrices,
    )


def choose_policy_threshold(
    probabilities: FloatArray,
    intervention_utility: FloatArray,
) -> float:
    """Maximize training utility, breaking ties toward fewer interventions."""

    candidates = np.concatenate(
        (
            np.asarray([0.0]),
            np.unique(probabilities),
            np.asarray([np.nextafter(1.0, 2.0)]),
        )
    )
    scored = []
    for threshold in candidates:
        selected = probabilities >= threshold
        utility = float((selected * intervention_utility).mean())
        scored.append((utility, -float(selected.mean()), float(threshold)))
    return max(scored)[2]


def _classifier(c_value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="liblinear",
    )


def _validate_split(
    target: BoolArray,
    groups: NDArray[np.str_],
    train_indices: IntArray,
    test_indices: IntArray,
) -> None:
    overlap = set(groups[train_indices]).intersection(groups[test_indices])
    if overlap:
        raise ValueError(f"Grouped split leaked {len(overlap)} image hashes")
    if len(np.unique(target[train_indices])) != 2:
        raise ValueError("Grouped training fold contains only one target class")


def _safe_group_splits(
    features: FloatArray,
    target: BoolArray,
    groups: NDArray[np.str_],
    *,
    folds: int,
    seed: int,
) -> list[tuple[IntArray, IntArray]]:
    """Find deterministic grouped splits whose training folds keep both classes."""

    for attempt in range(100):
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed + attempt,
        )
        splits = list(splitter.split(features, target, groups))
        if all(
            not set(groups[train_indices]).intersection(groups[test_indices])
            and len(np.unique(target[train_indices])) == 2
            for train_indices, test_indices in splits
        ):
            return splits
    raise ValueError(
        "Could not construct grouped folds with both target classes in training"
    )


def _grouped_oof_probabilities(
    features: FloatArray,
    target: BoolArray,
    groups: NDArray[np.str_],
    *,
    folds: int,
    c_value: float,
    seed: int,
) -> FloatArray:
    probabilities = np.empty(len(target), dtype=float)
    for train_indices, test_indices in _safe_group_splits(
        features,
        target,
        groups,
        folds=folds,
        seed=seed,
    ):
        _validate_split(target, groups, train_indices, test_indices)
        scaler = StandardScaler().fit(features[train_indices])
        model = _classifier(c_value, seed)
        model.fit(
            scaler.transform(features[train_indices]),
            target[train_indices],
        )
        probabilities[test_indices] = model.predict_proba(
            scaler.transform(features[test_indices])
        )[:, 1]
    return probabilities


def fit_frozen_grouped_selector(
    features: FloatArray,
    target: BoolArray,
    intervention_utility: FloatArray,
    groups: NDArray[np.str_],
    *,
    folds: int,
    repeats: int,
    c_value: float,
    seed: int,
) -> FrozenSelector:
    """Fit a selector without using any external-test labels or thresholds."""

    repeated_oof = np.asarray(
        [
            _grouped_oof_probabilities(
                features,
                target,
                groups,
                folds=folds,
                c_value=c_value,
                seed=seed + repeat * 1000,
            )
            for repeat in range(repeats)
        ]
    )
    mean_oof = repeated_oof.mean(axis=0)
    threshold = choose_policy_threshold(mean_oof, intervention_utility)
    scaler = StandardScaler().fit(features)
    model = _classifier(c_value, seed)
    model.fit(scaler.transform(features), target)
    return FrozenSelector(
        scaler=scaler,
        model=model,
        threshold=threshold,
        mean_training_oof_probability=mean_oof,
    )


def _metric_distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _policy_metrics(
    target: BoolArray,
    utility: FloatArray,
    probabilities: FloatArray,
    selected: BoolArray,
) -> dict[str, float]:
    helpful = target
    harmful = utility < 0
    return {
        "auroc": float(roc_auc_score(target, probabilities)),
        "auprc": float(average_precision_score(target, probabilities)),
        "policy_utility": float((selected * utility).mean()),
        "selection_rate": float(selected.mean()),
        "help_capture_rate": float(selected[helpful].mean()),
        "harm_avoidance_rate": (
            float((~selected[harmful]).mean()) if harmful.any() else 1.0
        ),
    }


def _group_bootstrap(
    first: FloatArray,
    target: BoolArray,
    groups: NDArray[np.str_],
    *,
    metric: str,
    resamples: int,
    seed: int,
    second: FloatArray | None = None,
) -> dict[str, Any]:
    """Bootstrap image groups, preserving within-image dependence."""

    unique_groups = np.unique(groups)
    group_indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    generator = np.random.default_rng(seed)

    def estimate(indices: IntArray, values: FloatArray) -> float | None:
        if metric == "auroc":
            sampled_target = target[indices]
            if len(np.unique(sampled_target)) != 2:
                return None
            return float(roc_auc_score(sampled_target, values[indices]))
        if metric == "mean":
            return float(values[indices].mean())
        raise ValueError(f"Unknown bootstrap metric: {metric}")

    full_indices = np.arange(len(target), dtype=np.int64)
    first_point = estimate(full_indices, first)
    if first_point is None:
        raise ValueError("Full sample lacks a target class")
    second_point = estimate(full_indices, second) if second is not None else None
    values = []
    for _ in range(resamples):
        sampled_groups = generator.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        indices = np.concatenate(
            [group_indices[group] for group in sampled_groups]
        )
        first_estimate = estimate(indices, first)
        if first_estimate is None:
            continue
        if second is None:
            values.append(first_estimate)
            continue
        second_estimate = estimate(indices, second)
        if second_estimate is not None:
            values.append(first_estimate - second_estimate)
    array = np.asarray(values, dtype=float)
    point = (
        first_point
        if second is None
        else first_point - float(second_point)
    )
    return {
        "point_estimate": point,
        "group_bootstrap_95_percent_interval": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "valid_resamples": len(array),
    }


def evaluate_grouped_feature_set(
    features: FloatArray,
    target: BoolArray,
    utility: FloatArray,
    groups: NDArray[np.str_],
    *,
    feature_names: Sequence[str],
    outer_folds: int,
    repeats: int,
    inner_folds: int,
    c_value: float,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[dict[str, Any], EvaluationArrays]:
    """Run repeated grouped CV with a training-only policy threshold."""

    probabilities = np.empty((repeats, len(target)), dtype=float)
    selections = np.empty((repeats, len(target)), dtype=bool)
    repeat_metrics = []
    fold_records = []
    coefficients = []
    for repeat in range(repeats):
        repeat_seed = seed + repeat * 1000
        repeat_probabilities = np.empty(len(target), dtype=float)
        repeat_selections = np.empty(len(target), dtype=bool)
        for fold, (train_indices, test_indices) in enumerate(
            _safe_group_splits(
                features,
                target,
                groups,
                folds=outer_folds,
                seed=repeat_seed,
            )
        ):
            _validate_split(target, groups, train_indices, test_indices)
            inner_seed = repeat_seed + fold + 1
            inner_probabilities = _grouped_oof_probabilities(
                features[train_indices],
                target[train_indices],
                groups[train_indices],
                folds=inner_folds,
                c_value=c_value,
                seed=inner_seed,
            )
            threshold = choose_policy_threshold(
                inner_probabilities,
                utility[train_indices],
            )
            scaler = StandardScaler().fit(features[train_indices])
            model = _classifier(c_value, inner_seed)
            model.fit(
                scaler.transform(features[train_indices]),
                target[train_indices],
            )
            fold_probabilities = model.predict_proba(
                scaler.transform(features[test_indices])
            )[:, 1]
            repeat_probabilities[test_indices] = fold_probabilities
            repeat_selections[test_indices] = fold_probabilities >= threshold
            coefficients.append(model.coef_[0].astype(float).tolist())
            fold_records.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "threshold": threshold,
                    "train_count": len(train_indices),
                    "test_count": len(test_indices),
                    "group_overlap_count": 0,
                }
            )
        probabilities[repeat] = repeat_probabilities
        selections[repeat] = repeat_selections
        repeat_metrics.append(
            _policy_metrics(
                target,
                utility,
                repeat_probabilities,
                repeat_selections,
            )
        )

    aggregate_probability = probabilities.mean(axis=0)
    expected_policy_utility = selections.mean(axis=0) * utility
    coefficient_array = np.asarray(coefficients, dtype=float)
    summary = {
        "feature_names": list(feature_names),
        "repeat_metric_distribution": {
            name: _metric_distribution(
                [result[name] for result in repeat_metrics]
            )
            for name in repeat_metrics[0]
        },
        "aggregate_oof": {
            "auroc": _group_bootstrap(
                aggregate_probability,
                target,
                groups,
                metric="auroc",
                resamples=bootstrap_resamples,
                seed=seed + 800_000,
            ),
            "expected_policy_utility": _group_bootstrap(
                expected_policy_utility,
                target,
                groups,
                metric="mean",
                resamples=bootstrap_resamples,
                seed=seed + 800_001,
            ),
        },
        "standardized_coefficient_distribution": {
            feature: _metric_distribution(
                coefficient_array[:, index].tolist()
            )
            for index, feature in enumerate(feature_names)
        },
        "fold_records": fold_records,
        "repeat_metrics": repeat_metrics,
    }
    return summary, EvaluationArrays(probabilities, selections)


def _scope_indices(
    data: HaloQuestSelectorData, scope: str
) -> IntArray:
    if scope == "pooled":
        return np.arange(len(data.target), dtype=np.int64)
    indices = np.flatnonzero(data.scopes == scope)
    if not len(indices):
        raise ValueError(f"No samples for scope: {scope}")
    return indices


def evaluate_cross_scope_feature_set(
    data: HaloQuestSelectorData,
    *,
    feature_set: str,
    feature_names: Sequence[str],
    train_scope: str,
    test_scope: str,
    inner_folds: int,
    repeats: int,
    c_value: float,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[dict[str, Any], FloatArray, BoolArray]:
    """Train on source-only images and evaluate the complete target scope."""

    target_indices = _scope_indices(data, test_scope)
    target_groups = set(data.groups[target_indices])
    source_candidates = _scope_indices(data, train_scope)
    train_indices = np.asarray(
        [
            index
            for index in source_candidates
            if data.groups[index] not in target_groups
        ],
        dtype=np.int64,
    )
    excluded_count = len(source_candidates) - len(train_indices)
    if set(data.groups[train_indices]).intersection(data.groups[target_indices]):
        raise AssertionError("Cross-scope image leakage was not removed")
    source_features = data.features[feature_set][train_indices]
    source_target = data.target[train_indices]
    source_groups = data.groups[train_indices]
    source_utility = data.intervention_utility[train_indices]
    inner_probabilities = np.mean(
        [
            _grouped_oof_probabilities(
                source_features,
                source_target,
                source_groups,
                folds=inner_folds,
                c_value=c_value,
                seed=seed + repeat * 1000,
            )
            for repeat in range(repeats)
        ],
        axis=0,
    )
    threshold = choose_policy_threshold(
        inner_probabilities,
        source_utility,
    )
    scaler = StandardScaler().fit(source_features)
    model = _classifier(c_value, seed)
    model.fit(scaler.transform(source_features), source_target)
    target_probabilities = model.predict_proba(
        scaler.transform(data.features[feature_set][target_indices])
    )[:, 1]
    selected = target_probabilities >= threshold
    metrics = _policy_metrics(
        data.target[target_indices],
        data.intervention_utility[target_indices],
        target_probabilities,
        selected,
    )
    summary = {
        "feature_names": list(feature_names),
        "train_scope": train_scope,
        "test_scope": test_scope,
        "source_candidate_count": len(source_candidates),
        "source_train_count": len(train_indices),
        "source_overlap_excluded_count": excluded_count,
        "target_test_count": len(target_indices),
        "train_test_group_overlap_count": 0,
        "threshold_from_repeated_source_group_oof": threshold,
        "metrics": metrics,
        "target_auroc": _group_bootstrap(
            target_probabilities,
            data.target[target_indices],
            data.groups[target_indices],
            metric="auroc",
            resamples=bootstrap_resamples,
            seed=seed + 700_000,
        ),
        "target_policy_utility": _group_bootstrap(
            selected * data.intervention_utility[target_indices],
            data.target[target_indices],
            data.groups[target_indices],
            metric="mean",
            resamples=bootstrap_resamples,
            seed=seed + 700_001,
        ),
        "standardized_coefficients": {
            name: float(value)
            for name, value in zip(
                feature_names, model.coef_[0], strict=True
            )
        },
    }
    return summary, target_probabilities, selected


def run_exp008(
    data: HaloQuestSelectorData,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    scopes: Sequence[str],
    transfer_directions: Sequence[Mapping[str, str]],
    outer_folds: int,
    repeats: int,
    inner_folds: int,
    c_value: float,
    bootstrap_resamples: int,
    seed: int,
    primary_feature_set: str,
    comparator_feature_set: str,
    minimum_mean_auroc_gain: float,
    minimum_mean_policy_gain_over_always_intervention: float,
    minimum_repeat_auroc_win_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run grouped validation, transfer tests, and frozen Go/No-Go checks."""

    within_results: dict[str, Any] = {}
    within_arrays: dict[str, dict[str, EvaluationArrays]] = {}
    prediction_records = [
        {
            "haloquest_id": int(data.sample_ids[index]),
            "dataset_scope": str(data.scopes[index]),
            "image_sha256": str(data.groups[index]),
            "action_effect": str(data.effects[index]),
            "intervention_utility": float(data.intervention_utility[index]),
            "within_scope": {},
            "cross_scope_target": {},
        }
        for index in range(len(data.target))
    ]
    for scope_index, scope in enumerate(scopes):
        indices = _scope_indices(data, scope)
        scope_results = {}
        scope_arrays = {}
        for feature_index, (name, fields) in enumerate(feature_sets.items()):
            result, arrays = evaluate_grouped_feature_set(
                data.features[name][indices],
                data.target[indices],
                data.intervention_utility[indices],
                data.groups[indices],
                feature_names=fields,
                outer_folds=outer_folds,
                repeats=repeats,
                inner_folds=inner_folds,
                c_value=c_value,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + scope_index * 100_000 + feature_index * 10_000,
            )
            scope_results[name] = result
            scope_arrays[name] = arrays
            for local_index, global_index in enumerate(indices):
                prediction_records[global_index]["within_scope"].setdefault(
                    scope, {}
                )[name] = {
                    "mean_oof_probability": float(
                        arrays.probabilities[:, local_index].mean()
                    ),
                    "intervention_selection_rate": float(
                        arrays.selections[:, local_index].mean()
                    ),
                }
        within_results[scope] = {
            "sample_count": len(indices),
            "image_group_count": len(np.unique(data.groups[indices])),
            "effect_counts": dict(
                sorted(Counter(data.effects[indices].tolist()).items())
            ),
            "policy_references": {
                "always_native_utility": 0.0,
                "always_intervention_utility": float(
                    data.intervention_utility[indices].mean()
                ),
                "oracle_utility": float(
                    np.maximum(data.intervention_utility[indices], 0).mean()
                ),
            },
            "feature_set_results": scope_results,
        }
        within_arrays[scope] = scope_arrays

    transfer_results = {}
    for direction_index, direction in enumerate(transfer_directions):
        train_scope = str(direction["train"])
        test_scope = str(direction["test"])
        key = f"{train_scope}_to_{test_scope}"
        target_indices = _scope_indices(data, test_scope)
        direction_results = {}
        for feature_index, (name, fields) in enumerate(feature_sets.items()):
            result, probabilities, selected = evaluate_cross_scope_feature_set(
                data,
                feature_set=name,
                feature_names=fields,
                train_scope=train_scope,
                test_scope=test_scope,
                inner_folds=inner_folds,
                repeats=repeats,
                c_value=c_value,
                bootstrap_resamples=bootstrap_resamples,
                seed=(
                    seed
                    + 500_000
                    + direction_index * 100_000
                    + feature_index * 10_000
                ),
            )
            direction_results[name] = result
            for local_index, global_index in enumerate(target_indices):
                prediction_records[global_index][
                    "cross_scope_target"
                ].setdefault(key, {})[name] = {
                    "probability": float(probabilities[local_index]),
                    "selected": bool(selected[local_index]),
                }
        transfer_results[key] = direction_results

    primary = within_results["pooled"]["feature_set_results"][
        primary_feature_set
    ]
    comparator = within_results["pooled"]["feature_set_results"][
        comparator_feature_set
    ]
    primary_repeats = primary["repeat_metrics"]
    comparator_repeats = comparator["repeat_metrics"]
    mean_auroc_gain = (
        primary["repeat_metric_distribution"]["auroc"]["mean"]
        - comparator["repeat_metric_distribution"]["auroc"]["mean"]
    )
    always_intervention = within_results["pooled"]["policy_references"][
        "always_intervention_utility"
    ]
    mean_policy_gain = (
        primary["repeat_metric_distribution"]["policy_utility"]["mean"]
        - always_intervention
    )
    win_fraction = float(
        np.mean(
            [
                first["auroc"] > second["auroc"]
                for first, second in zip(
                    primary_repeats, comparator_repeats, strict=True
                )
            ]
        )
    )
    pooled_indices = _scope_indices(data, "pooled")
    primary_arrays = within_arrays["pooled"][primary_feature_set]
    comparator_arrays = within_arrays["pooled"][comparator_feature_set]
    paired_comparison = {
        "aggregate_oof_auroc": _group_bootstrap(
            primary_arrays.probabilities.mean(axis=0),
            data.target[pooled_indices],
            data.groups[pooled_indices],
            metric="auroc",
            resamples=bootstrap_resamples,
            seed=seed + 950_000,
            second=comparator_arrays.probabilities.mean(axis=0),
        ),
        "repeat_averaged_policy_utility": _group_bootstrap(
            primary_arrays.selections.mean(axis=0)
            * data.intervention_utility[pooled_indices],
            data.target[pooled_indices],
            data.groups[pooled_indices],
            metric="mean",
            resamples=bootstrap_resamples,
            seed=seed + 950_001,
            second=(
                comparator_arrays.selections.mean(axis=0)
                * data.intervention_utility[pooled_indices]
            ),
        ),
    }
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
            "value": win_fraction,
            "threshold": minimum_repeat_auroc_win_fraction,
            "pass": win_fraction >= minimum_repeat_auroc_win_fraction,
        },
    }
    verdict = (
        "conditional_go"
        if all(check["pass"] for check in checks.values())
        else "no_go"
    )
    summary = {
        "protocol": "exp008-native-answer-grouped-selector-v1",
        "scope": (
            "Exploratory selector validation on action-change-enriched, "
            "AI-audited samples. It does not estimate population action rates "
            "or replace official semantic evaluation."
        ),
        "sample": {
            "count": len(data.target),
            "effect_counts": dict(
                sorted(Counter(data.effects.tolist()).items())
            ),
            "scope_counts": dict(
                sorted(Counter(data.scopes.tolist()).items())
            ),
            "image_group_count": len(np.unique(data.groups)),
        },
        "validation": {
            "outer_folds": outer_folds,
            "repeats": repeats,
            "inner_folds": inner_folds,
            "logistic_c": c_value,
            "bootstrap_resamples": bootstrap_resamples,
            "leakage_controls": [
                "all outer and inner splits grouped by image_sha256",
                "standardization fitted on training folds only",
                "policy thresholds chosen from inner grouped OOF predictions",
                "cross-scope source images overlapping target are excluded",
                "target folds and target scopes never tune a model or threshold",
            ],
        },
        "within_scope": within_results,
        "cross_scope_transfer": transfer_results,
        "primary_paired_comparison": paired_comparison,
        "go_no_go": {
            "verdict": verdict,
            "evaluation_scope": "pooled",
            "primary_feature_set": primary_feature_set,
            "comparator_feature_set": comparator_feature_set,
            "criteria": checks,
            "meaning": (
                "Conditional Go only supports expanding the selector study; "
                "it is not a deployment or semantic-accuracy claim."
            ),
        },
    }
    return summary, prediction_records
