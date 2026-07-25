#!/usr/bin/env python3
"""Build the 43-pair EXP-003 semantic action review table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_semantic_review import validate_action_review_records  # noqa: E402


FIELDNAMES = [
    "image_id",
    "conflict_type",
    "image_path",
    "conflict_question",
    "conflict_reference_answer",
    "native_conflict_answer",
    "intervention_conflict_answer",
    "native_conflict_rule_stance",
    "intervention_conflict_rule_stance",
    "rule_proxy_transition",
    "clean_question",
    "clean_reference_answer",
    "native_clean_answer",
    "intervention_clean_answer",
    "native_clean_reference_contained",
    "intervention_clean_reference_contained",
    "ai_native_conflict_correct",
    "ai_intervention_conflict_correct",
    "ai_conflict_action_effect",
    "ai_native_clean_correct",
    "ai_intervention_clean_correct",
    "ai_clean_action_effect",
    "ai_overall_action",
    "ai_confidence",
    "ai_notes",
    "ai_review_protocol",
    "human_confirmation",
    "human_native_conflict_correct",
    "human_intervention_conflict_correct",
    "human_conflict_action_effect",
    "human_native_clean_correct",
    "human_intervention_clean_correct",
    "human_clean_action_effect",
    "human_overall_action",
    "human_confidence",
    "human_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_smoke_scored.jsonl",
    )
    parser.add_argument(
        "--intervention",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp003/"
        "qwen2_5_vl_3b_premise_verification_scored.jsonl",
    )
    parser.add_argument(
        "--native-prereview",
        type=Path,
        default=PROJECT_ROOT / "annotations/exp001_smoke_ai_prereview.csv",
    )
    parser.add_argument(
        "--transition-prereview",
        type=Path,
        default=PROJECT_ROOT
        / "annotations/exp003_conflict_transition_ai_review.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "annotations/exp003_action_semantic_review_43.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing table only when it has no human confirmations.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        existing = read_csv(args.output)
        confirmed = [
            row for row in existing if row.get("human_confirmation", "").strip()
        ]
        if confirmed:
            raise ValueError(
                f"Refusing to replace {len(confirmed)} human confirmations"
            )
        if not args.force:
            raise FileExistsError(
                f"{args.output} exists; use --force to replace it"
            )

    intervention_records = read_jsonl(args.intervention)
    selected_ids = {int(record["image_id"]) for record in intervention_records}
    native_records = [
        record
        for record in read_jsonl(args.native)
        if int(record["image_id"]) in selected_ids
        and record["condition"] in {
            "multimodal_conflict",
            "multimodal_clean",
        }
    ]
    native = {
        (int(record["image_id"]), str(record["condition"])): record
        for record in native_records
    }
    intervention = {
        (int(record["image_id"]), str(record["condition"])): record
        for record in intervention_records
    }
    if native.keys() != intervention.keys():
        raise ValueError("Native and intervention records are not aligned")
    if len(selected_ids) != 43:
        raise ValueError(f"Expected 43 image IDs, found {len(selected_ids)}")

    native_ai = {
        (int(row["image_id"]), str(row["condition"])): row
        for row in read_csv(args.native_prereview)
        if int(row["image_id"]) in selected_ids
    }
    transition_ai = {
        int(row["image_id"]): row
        for row in read_csv(args.transition_prereview)
    }
    output = []
    for image_id in sorted(selected_ids):
        native_conflict = native[(image_id, "multimodal_conflict")]
        intervention_conflict = intervention[
            (image_id, "multimodal_conflict")
        ]
        native_clean = native[(image_id, "multimodal_clean")]
        intervention_clean = intervention[(image_id, "multimodal_clean")]
        prior_conflict = native_ai[(image_id, "multimodal_conflict")]
        prior_clean = native_ai[(image_id, "multimodal_clean")]
        transition = transition_ai.get(image_id)
        ai_intervention_conflict = (
            transition["intervention_semantic_correct"] if transition else ""
        )
        ai_conflict_effect = (
            transition["semantic_transition"] if transition else ""
        )
        ai_fields_complete = bool(transition)
        output.append(
            {
                "image_id": image_id,
                "conflict_type": native_conflict["conflict_type"],
                "image_path": (
                    f"annotations/exp001_audit_images/{image_id:08d}.png"
                ),
                "conflict_question": native_conflict["question"],
                "conflict_reference_answer": native_conflict["reference_answer"],
                "native_conflict_answer": native_conflict["generated_text"],
                "intervention_conflict_answer": intervention_conflict[
                    "generated_text"
                ],
                "native_conflict_rule_stance": native_conflict["stance"],
                "intervention_conflict_rule_stance": intervention_conflict[
                    "stance"
                ],
                "rule_proxy_transition": (
                    transition["rule_proxy_transition"] if transition else (
                        "both_success"
                        if native_conflict["stance"] == "reject_premise"
                        and intervention_conflict["stance"] == "reject_premise"
                        else "neither_success"
                    )
                ),
                "clean_question": native_clean["question"],
                "clean_reference_answer": native_clean["reference_answer"],
                "native_clean_answer": native_clean["generated_text"],
                "intervention_clean_answer": intervention_clean[
                    "generated_text"
                ],
                "native_clean_reference_contained": native_clean[
                    "reference_contained"
                ],
                "intervention_clean_reference_contained": intervention_clean[
                    "reference_contained"
                ],
                "ai_native_conflict_correct": prior_conflict[
                    "ai_semantic_correct"
                ],
                "ai_intervention_conflict_correct": (
                    ai_intervention_conflict
                ),
                "ai_conflict_action_effect": ai_conflict_effect,
                "ai_native_clean_correct": prior_clean[
                    "ai_semantic_correct"
                ],
                "ai_intervention_clean_correct": "",
                "ai_clean_action_effect": "",
                "ai_overall_action": "",
                "ai_confidence": "",
                "ai_notes": (
                    transition["ai_notes"] if transition else ""
                ),
                "ai_review_protocol": (
                    "exp003-conflict-transition-ai-prereview-v1-partial"
                    if ai_fields_complete
                    else "pending"
                ),
                "human_confirmation": "",
                "human_native_conflict_correct": "",
                "human_intervention_conflict_correct": "",
                "human_conflict_action_effect": "",
                "human_native_clean_correct": "",
                "human_intervention_clean_correct": "",
                "human_clean_action_effect": "",
                "human_overall_action": "",
                "human_confidence": "",
                "human_notes": "",
            }
        )
    validate_action_review_records(output)
    _write_csv(args.output, output)
    print(
        f"wrote={len(output)} partial_ai_conflict={len(transition_ai)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()

