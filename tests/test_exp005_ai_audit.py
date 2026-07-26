from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from summarize_exp005_ai_audit import summarize_audit  # noqa: E402


def test_audit_requires_every_non_tie_pair_once() -> None:
    pairs = [
        {"haloquest_id": 1, "effect": "helps"},
        {"haloquest_id": 2, "effect": "uncertain"},
        {"haloquest_id": 3, "effect": "tie"},
    ]
    audit = {
        "protocol": "test",
        "reviewer_type": "AI",
        "human_annotation_claim": False,
        "scope_rule": "all non-ties",
        "effect_groups": {
            "helps": [1],
            "harms": [],
            "tie": [2],
            "uncertain": [],
        },
        "proxy_tie_stratified_audit": {},
        "limitations": [],
    }
    summary = summarize_audit(pairs, audit)
    assert summary["scope"]["audited_pair_count"] == 2
    assert summary["ai_effect_counts"]["tie"] == 1


def test_audit_rejects_missing_non_tie_pair() -> None:
    pairs = [
        {"haloquest_id": 1, "effect": "helps"},
        {"haloquest_id": 2, "effect": "uncertain"},
    ]
    audit = {
        "protocol": "test",
        "reviewer_type": "AI",
        "human_annotation_claim": False,
        "scope_rule": "all non-ties",
        "effect_groups": {
            "helps": [1],
            "harms": [],
            "tie": [],
            "uncertain": [],
        },
        "proxy_tie_stratified_audit": {},
        "limitations": [],
    }
    with pytest.raises(ValueError, match="scope mismatch"):
        summarize_audit(pairs, audit)

