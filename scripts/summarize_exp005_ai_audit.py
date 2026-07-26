#!/usr/bin/env python3
"""Validate and summarize the EXP-005 AI action-effect audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paired",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp005/qwen2_5_vl_3b_action_utility.jsonl",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT
        / "annotations/exp005_action_effect_ai_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp005/qwen2_5_vl_3b_action_ai_audit_summary.json",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_audit(
    pairs: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Validate complete review of proxy non-ties and summarize decisions."""

    pair_by_id = {int(pair["haloquest_id"]): pair for pair in pairs}
    expected_ids = {
        haloquest_id
        for haloquest_id, pair in pair_by_id.items()
        if pair["effect"] != "tie"
    }
    grouped: dict[str, set[int]] = {}
    for effect, values in audit["effect_groups"].items():
        grouped[str(effect)] = {int(value) for value in values}
    seen: set[int] = set()
    for effect, ids in grouped.items():
        overlap = seen & ids
        if overlap:
            raise ValueError(f"Audit IDs appear in multiple groups: {sorted(overlap)}")
        unknown = ids - set(pair_by_id)
        if unknown:
            raise ValueError(f"Audit contains unknown IDs: {sorted(unknown)}")
        seen.update(ids)
    if seen != expected_ids:
        raise ValueError(
            "Audit scope mismatch: "
            f"missing={sorted(expected_ids - seen)}, "
            f"extra={sorted(seen - expected_ids)}"
        )

    counts = {effect: len(ids) for effect, ids in grouped.items()}
    decisive = counts["helps"] + counts["harms"] + counts["tie"]
    directional = counts["helps"] + counts["harms"]
    proxy_effects = Counter(
        str(pair_by_id[haloquest_id]["effect"]) for haloquest_id in seen
    )
    return {
        "protocol": audit["protocol"],
        "reviewer_type": audit["reviewer_type"],
        "human_annotation_claim": bool(audit["human_annotation_claim"]),
        "scope": {
            "audited_pair_count": len(seen),
            "total_pair_count": len(pairs),
            "audited_fraction": len(seen) / len(pairs),
            "proxy_effect_counts_in_scope": dict(sorted(proxy_effects.items())),
            "scope_rule": audit["scope_rule"],
        },
        "ai_effect_counts": counts,
        "decisive_audit_coverage": decisive / len(seen),
        "directional_effect": {
            "count": directional,
            "helps_fraction": counts["helps"] / directional,
            "harms_fraction": counts["harms"] / directional,
            "helps_to_harms_ratio": (
                counts["helps"] / counts["harms"] if counts["harms"] else None
            ),
            "net_help_minus_harm_count": counts["helps"] - counts["harms"],
        },
        "proxy_tie_stratified_audit": audit["proxy_tie_stratified_audit"],
        "limitations": audit["limitations"],
    }


def main() -> None:
    args = parse_args()
    pairs = read_jsonl(args.paired)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    summary = summarize_audit(pairs, audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

