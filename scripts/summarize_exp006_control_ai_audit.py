#!/usr/bin/env python3
"""Validate and summarize the EXP-006 control action audit."""

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
        / "outputs/metrics/exp006/"
        "qwen2_5_vl_3b_control_action_pairs.jsonl",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT
        / "annotations/exp006_control_action_ai_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp006/"
        "qwen2_5_vl_3b_control_action_ai_audit_summary.json",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def expected_scope_ids(pairs: list[dict[str, Any]]) -> set[int]:
    """Reproduce the frozen audit selection rule."""

    return {
        int(pair["haloquest_id"])
        for pair in pairs
        if (
            pair["hallucination_type"] == "visual challenge"
            and (
                abs(float(pair["lexical_delta"])) > 0.1
                or bool(pair["new_target_linked_rejection"])
            )
        )
        or (
            pair["hallucination_type"] == "insufficient context"
            and float(pair["lexical_delta"]) < -0.1
        )
    }


def summarize_audit(
    pairs: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Validate exact audit scope and summarize effect labels."""

    pair_by_id = {int(pair["haloquest_id"]): pair for pair in pairs}
    expected_ids = expected_scope_ids(pairs)
    grouped = {
        str(effect): {int(value) for value in values}
        for effect, values in audit["effect_groups"].items()
    }
    seen: set[int] = set()
    for ids in grouped.values():
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
    by_category = {}
    for category in ("visual challenge", "insufficient context"):
        category_ids = {
            haloquest_id
            for haloquest_id in seen
            if pair_by_id[haloquest_id]["hallucination_type"] == category
        }
        by_category[category] = {
            "count": len(category_ids),
            "effect_counts": {
                effect: len(ids & category_ids)
                for effect, ids in grouped.items()
            },
        }
    lexical_direction = Counter(
        (
            "intervention"
            if float(pair_by_id[haloquest_id]["lexical_delta"]) > 0.1
            else "native"
            if float(pair_by_id[haloquest_id]["lexical_delta"]) < -0.1
            else "tie"
        )
        for haloquest_id in seen
    )
    return {
        "protocol": audit["protocol"],
        "reviewer_type": audit["reviewer_type"],
        "human_annotation_claim": bool(audit["human_annotation_claim"]),
        "scope": {
            "audited_pair_count": len(seen),
            "total_pair_count": len(pairs),
            "audited_fraction": len(seen) / len(pairs),
            "lexical_direction_counts": dict(sorted(lexical_direction.items())),
            "scope_rule": audit["scope_rule"],
        },
        "ai_effect_counts": counts,
        "net_help_minus_harm_count": counts["helps"] - counts["harms"],
        "by_hallucination_type": by_category,
        "remaining_pool_stratified_audit": audit[
            "remaining_pool_stratified_audit"
        ],
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

