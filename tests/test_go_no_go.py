from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from go_no_go import (  # noqa: E402
    choose_policy_threshold,
    policy_utility,
    prepare_selector_data,
    run_go_no_go,
)


FEATURE_SETS = {
    "contribution_only": ["contribution"],
    "reliability_only": ["instability", "worst_drop"],
    "contribution_plus_reliability": [
        "contribution",
        "instability",
        "worst_drop",
    ],
}


def _synthetic_records(count: int = 60) -> list[dict[str, object]]:
    records = []
    for index in range(count):
        intervention_only = index % 4 == 0
        native_only = index % 20 == 1
        records.append(
            {
                "image_id": index,
                "conflict_type": "object",
                "baseline_conflict_rejection_proxy": native_only,
                "intervention_conflict_rejection_proxy": intervention_only,
                "contribution": float((index * 7) % 13) / 13,
                "instability": 1.0 if intervention_only else 0.05,
                "worst_drop": 0.8 if intervention_only else 0.02,
            }
        )
    return records


def test_prepare_selector_data_defines_intervention_only_target() -> None:
    records = _synthetic_records(20)
    data = prepare_selector_data(records, FEATURE_SETS)
    assert data.target.sum() == 5
    assert data.native_utility.sum() == 1
    assert data.features["contribution_plus_reliability"].shape == (20, 3)


def test_threshold_optimization_can_protect_native_only_case() -> None:
    probabilities = np.asarray([0.9, 0.1, 0.8])
    native = np.asarray([0.0, 1.0, 0.0])
    intervention = np.asarray([1.0, 0.0, 1.0])
    threshold = choose_policy_threshold(
        probabilities,
        native,
        intervention,
    )
    selected = probabilities >= threshold
    assert policy_utility(selected, native, intervention).mean() == 1.0
    assert selected.tolist() == [True, False, True]


def test_nested_evaluation_is_deterministic_and_favors_signal() -> None:
    data = prepare_selector_data(_synthetic_records(), FEATURE_SETS)
    kwargs = {
        "outer_folds": 3,
        "repeats": 2,
        "inner_folds": 2,
        "c_grid": [0.1, 1.0],
        "bootstrap_resamples": 100,
        "seed": 7,
        "primary_feature_set": "contribution_plus_reliability",
        "comparator_feature_set": "contribution_only",
        "minimum_mean_auroc_gain": 0.05,
        "minimum_mean_policy_gain_over_always_intervention": 0.0,
        "minimum_repeat_auroc_win_fraction": 0.5,
    }
    first, predictions = run_go_no_go(data, FEATURE_SETS, **kwargs)
    second, _ = run_go_no_go(data, FEATURE_SETS, **kwargs)
    assert first == second
    assert len(predictions) == 60
    results = first["feature_set_results"]
    reliability_auc = results["reliability_only"][
        "repeat_metric_distribution"
    ]["auroc"]["mean"]
    contribution_auc = results["contribution_only"][
        "repeat_metric_distribution"
    ]["auroc"]["mean"]
    assert reliability_auc > contribution_auc


def test_duplicate_ids_are_rejected() -> None:
    records = _synthetic_records(20)
    records[1]["image_id"] = records[0]["image_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_selector_data(records, FEATURE_SETS)


def test_custom_semantic_utility_fields_are_supported() -> None:
    records = _synthetic_records(20)
    for record in records:
        record["native_semantic"] = record[
            "baseline_conflict_rejection_proxy"
        ]
        record["intervention_semantic"] = record[
            "intervention_conflict_rejection_proxy"
        ]
    data = prepare_selector_data(
        records,
        FEATURE_SETS,
        native_utility_field="native_semantic",
        intervention_utility_field="intervention_semantic",
    )
    assert data.target.sum() == 5


def test_custom_utility_output_names_are_supported() -> None:
    data = prepare_selector_data(_synthetic_records(20), FEATURE_SETS)
    summary, predictions = run_go_no_go(
        data,
        FEATURE_SETS,
        outer_folds=2,
        repeats=2,
        inner_folds=2,
        c_grid=[1.0],
        bootstrap_resamples=20,
        seed=7,
        primary_feature_set="contribution_plus_reliability",
        comparator_feature_set="contribution_only",
        minimum_mean_auroc_gain=0.0,
        minimum_mean_policy_gain_over_always_intervention=0.0,
        minimum_repeat_auroc_win_fraction=0.0,
        native_utility_output_name="native_semantic_utility",
        intervention_utility_output_name="intervention_semantic_utility",
    )
    assert summary["sample"]["count"] == 20
    assert "native_semantic_utility" in predictions[0]
    assert "intervention_semantic_utility" in predictions[0]
    assert "native_rule_proxy_utility" not in predictions[0]
