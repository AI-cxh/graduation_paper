from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from confirmed_sensitivity import confirmed_dataset_exclusions  # noqa: E402


def _record(confirmation: str, semantic: str = "uncertain") -> dict[str, object]:
    return {
        "image_id": 7,
        "condition": "multimodal_conflict",
        "ai_issue_type": "dataset_validity",
        "human_confirmation": confirmation,
        "ai_semantic_correct": semantic,
        "human_correction_semantic": "",
        "ai_notes": "label mismatch",
    }


def test_confirmed_dataset_validity_issue_is_selected() -> None:
    assert confirmed_dataset_exclusions([_record("accept")]) == {
        7: "label mismatch"
    }


def test_unconfirmed_dataset_validity_issue_is_rejected() -> None:
    with pytest.raises(ValueError, match="not been human-confirmed"):
        confirmed_dataset_exclusions([_record("")])


def test_non_uncertain_resolution_is_rejected() -> None:
    with pytest.raises(ValueError, match="must resolve to uncertain"):
        confirmed_dataset_exclusions([_record("accept", "yes")])
