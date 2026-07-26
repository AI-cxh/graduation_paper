from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from summarize_exp006_control_ai_audit import (  # noqa: E402
    expected_scope_ids,
    summarize_audit,
)


def _pair(
    row_id: int,
    category: str,
    delta: float,
    new_rejection: bool = False,
) -> dict[str, object]:
    return {
        "haloquest_id": row_id,
        "hallucination_type": category,
        "lexical_delta": delta,
        "new_target_linked_rejection": new_rejection,
    }


def _audit(groups: dict[str, list[int]]) -> dict[str, object]:
    return {
        "protocol": "test",
        "reviewer_type": "AI",
        "human_annotation_claim": False,
        "scope_rule": "frozen",
        "effect_groups": groups,
        "remaining_pool_stratified_audit": {},
        "limitations": [],
    }


def test_scope_combines_visual_changes_and_insufficient_native_wins() -> None:
    pairs = [
        _pair(1, "visual challenge", 0.2),
        _pair(2, "visual challenge", 0.0, True),
        _pair(3, "insufficient context", -0.2),
        _pair(4, "insufficient context", 0.2),
    ]
    assert expected_scope_ids(pairs) == {1, 2, 3}


def test_summary_requires_exact_scope() -> None:
    pairs = [
        _pair(1, "visual challenge", 0.2),
        _pair(2, "insufficient context", -0.2),
    ]
    audit = _audit(
        {"helps": [1], "harms": [2], "tie": [], "uncertain": []}
    )
    summary = summarize_audit(pairs, audit)
    assert summary["ai_effect_counts"]["helps"] == 1
    assert summary["net_help_minus_harm_count"] == 0


def test_summary_rejects_incomplete_scope() -> None:
    pairs = [
        _pair(1, "visual challenge", 0.2),
        _pair(2, "insufficient context", -0.2),
    ]
    audit = _audit(
        {"helps": [1], "harms": [], "tie": [], "uncertain": []}
    )
    with pytest.raises(ValueError, match="scope mismatch"):
        summarize_audit(pairs, audit)

