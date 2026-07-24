from __future__ import annotations

import sys
from pathlib import Path

import pytest
from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mmmc_data import (  # noqa: E402
    MMMCPair,
    build_and_validate_pairs,
    conflict_type_counts,
    stratified_sample,
)


def _dataset(conflict_types: list[str]) -> Dataset:
    rows = {
        "image_id": [],
        "question": [],
        "answer": [],
        "conflict_type": [],
        "key_component": [],
        "key_component_relationships": [],
        "key_component_attributes": [],
        "image": [],
    }
    for image_id, conflict_type in enumerate(conflict_types):
        for current_type in (conflict_type, None):
            rows["image_id"].append(image_id)
            rows["question"].append("question")
            rows["answer"].append("answer")
            rows["conflict_type"].append(current_type)
            rows["key_component"].append(None)
            rows["key_component_relationships"].append(None)
            rows["key_component_attributes"].append(None)
            rows["image"].append(None)
    return Dataset.from_dict(rows)


def test_build_and_validate_pairs() -> None:
    pairs = build_and_validate_pairs(_dataset(["object", "relation", "attribute"]))
    assert len(pairs) == 3
    assert [pair.conflict_type for pair in pairs] == ["object", "relation", "attribute"]
    assert pairs[0].conflict_index == 0
    assert pairs[0].clean_index == 1


def test_rejects_unpaired_rows() -> None:
    dataset = _dataset(["object"]).select([0])
    with pytest.raises(ValueError, match="Invalid MMMC pair structure"):
        build_and_validate_pairs(dataset)


def test_stratified_sample_is_deterministic_and_proportional() -> None:
    pairs = [
        MMMCPair(i, "object" if i < 6 else "relation", 2 * i, 2 * i + 1)
        for i in range(10)
    ]
    first = stratified_sample(pairs, sample_size=5, seed=7)
    second = stratified_sample(pairs, sample_size=5, seed=7)
    assert first == second
    assert conflict_type_counts(first) == {"object": 3, "relation": 2}

