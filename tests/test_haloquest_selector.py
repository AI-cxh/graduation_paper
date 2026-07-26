from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from haloquest_selector import (  # noqa: E402
    choose_policy_threshold,
    fit_frozen_grouped_selector,
    prepare_haloquest_selector_data,
    run_exp008,
)


FEATURE_SETS = {
    "contribution_only": ["identity_visual_delta"],
    "reliability_only": [
        "population_std_visual_delta",
        "worst_drop_from_identity",
    ],
    "contribution_plus_reliability": [
        "identity_visual_delta",
        "population_std_visual_delta",
        "worst_drop_from_identity",
    ],
}


def _records() -> list[dict[str, object]]:
    records = []
    row_id = 0
    for scope_index, scope in enumerate(("false_premise", "control")):
        for group_index in range(30):
            effect = ("helps", "tie", "harms")[group_index % 3]
            contribution = (
                0.1
                if effect == "helps"
                else 0.8 + group_index / 1000
            )
            records.append(
                {
                    "haloquest_id": row_id,
                    "dataset_scope": scope,
                    "image_sha256": (
                        f"shared-{group_index}"
                        if group_index < 2
                        else f"{scope_index}-{group_index}"
                    ),
                    "action_effect": effect,
                    "identity_visual_delta": contribution,
                    "population_std_visual_delta": group_index / 100,
                    "worst_drop_from_identity": group_index / 200,
                }
            )
            row_id += 1
    return records


def test_choose_threshold_prefers_fewer_equal_utility_actions() -> None:
    threshold = choose_policy_threshold(
        np.asarray([0.9, 0.8, 0.2]),
        np.asarray([1.0, 0.0, -1.0]),
    )
    assert threshold == pytest.approx(0.9)


def test_prepare_rejects_duplicate_ids() -> None:
    records = _records()
    records[1]["haloquest_id"] = records[0]["haloquest_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_haloquest_selector_data(
            records,
            FEATURE_SETS,
            {"helps": 1.0, "harms": -1.0, "tie": 0.0},
        )


def test_fit_frozen_selector_uses_grouped_oof_threshold() -> None:
    data = prepare_haloquest_selector_data(
        _records(),
        FEATURE_SETS,
        {"helps": 1.0, "harms": -1.0, "tie": 0.0},
    )
    frozen = fit_frozen_grouped_selector(
        data.features["contribution_only"],
        data.target,
        data.intervention_utility,
        data.groups,
        folds=3,
        repeats=2,
        c_value=1.0,
        seed=7,
    )
    probabilities = frozen.predict_probability(
        data.features["contribution_only"]
    )
    assert probabilities.shape == (60,)
    assert 0 <= frozen.threshold <= 1


def test_run_exp008_groups_images_and_removes_transfer_overlap() -> None:
    data = prepare_haloquest_selector_data(
        _records(),
        FEATURE_SETS,
        {"helps": 1.0, "harms": -1.0, "tie": 0.0},
    )
    summary, predictions = run_exp008(
        data,
        FEATURE_SETS,
        scopes=["pooled", "false_premise", "control"],
        transfer_directions=[
            {"train": "false_premise", "test": "control"},
            {"train": "control", "test": "false_premise"},
        ],
        outer_folds=3,
        repeats=2,
        inner_folds=2,
        c_value=1.0,
        bootstrap_resamples=20,
        seed=7,
        primary_feature_set="contribution_plus_reliability",
        comparator_feature_set="contribution_only",
        minimum_mean_auroc_gain=0.0,
        minimum_mean_policy_gain_over_always_intervention=-1.0,
        minimum_repeat_auroc_win_fraction=0.0,
    )
    assert len(predictions) == 60
    assert summary["within_scope"]["pooled"]["image_group_count"] == 58
    transfer = summary["cross_scope_transfer"][
        "false_premise_to_control"
    ]["contribution_only"]
    assert transfer["source_overlap_excluded_count"] == 2
    assert transfer["train_test_group_overlap_count"] == 0
    assert transfer["metrics"]["auroc"] > 0.9
