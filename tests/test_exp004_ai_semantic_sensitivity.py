from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_exp004_ai_semantic_sensitivity import (  # noqa: E402
    INTERVENTION_UTILITY_FIELD,
    NATIVE_UTILITY_FIELD,
    build_semantic_records,
)


def _utility(image_id: int) -> dict[str, object]:
    return {
        "image_id": image_id,
        "conflict_type": "object",
        "conflict_identity_visual_delta": 0.1,
    }


def _review(
    image_id: int,
    native: str,
    intervention: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "image_id": image_id,
        "human_confirmation": "",
    }
    for prefix in ("ai", "human"):
        for field in (
            "native_conflict_correct",
            "intervention_conflict_correct",
            "conflict_action_effect",
            "native_clean_correct",
            "intervention_clean_correct",
            "clean_action_effect",
            "overall_action",
            "confidence",
        ):
            row[f"{prefix}_{field}"] = ""
    row.update(
        {
            "ai_native_conflict_correct": native,
            "ai_intervention_conflict_correct": intervention,
        }
    )
    return row


def test_uncertain_semantic_record_is_excluded_before_modeling() -> None:
    kept, excluded = build_semantic_records(
        [_utility(1), _utility(2)],
        [_review(1, "no", "yes"), _review(2, "yes", "uncertain")],
    )
    assert len(kept) == 1
    assert kept[0][NATIVE_UTILITY_FIELD] is False
    assert kept[0][INTERVENTION_UTILITY_FIELD] is True
    assert excluded == [
        {
            "image_id": 2,
            "native_label": "yes",
            "intervention_label": "uncertain",
            "reason": "uncertain_or_blank_conflict_semantic_label",
        }
    ]
