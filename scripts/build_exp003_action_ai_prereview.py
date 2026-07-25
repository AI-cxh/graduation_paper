#!/usr/bin/env python3
"""Complete a transparent AI pre-review for the 43 EXP-003 action pairs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_semantic_review import (  # noqa: E402
    infer_action_effect,
    infer_overall_action,
    validate_action_review_records,
)


PROTOCOL = "exp003-paired-action-ai-prereview-v1"

# The other 14 conflict judgments come from the earlier visual review of rule
# transitions and are already present in the paired table.
INTERVENTION_CONFLICT = {
    150293: "yes",
    713590: "no",
    1592318: "uncertain",
    2394830: "yes",
    2395592: "yes",
    2395656: "no",
    2395947: "no",
    2396016: "no",
    2396109: "yes",
    2396984: "no",
    2399942: "no",
    2399944: "no",
    2400695: "no",
    2401084: "no",
    2403720: "yes",
    2403767: "yes",
    2403954: "no",
    2404745: "yes",
    2404880: "no",
    2406390: "yes",
    2406642: "yes",
    2406717: "yes",
    2406901: "yes",
    2406916: "yes",
    2408152: "yes",
    2409203: "no",
    2409661: "no",
    2410268: "yes",
    2411972: "yes",
}

INTERVENTION_CLEAN = {
    150293: "yes",
    713590: "yes",
    1592318: "uncertain",
    2393697: "yes",
    2394830: "yes",
    2395592: "yes",
    2395656: "no",
    2395947: "yes",
    2396016: "yes",
    2396109: "yes",
    2396910: "yes",
    2396984: "yes",
    2399942: "no",
    2399944: "yes",
    2400695: "yes",
    2401084: "yes",
    2401710: "yes",
    2401850: "yes",
    2401906: "yes",
    2402123: "yes",
    2402350: "yes",
    2402736: "yes",
    2403720: "yes",
    2403767: "no",
    2403851: "yes",
    2403954: "no",
    2404745: "no",
    2404880: "yes",
    2405342: "yes",
    2405568: "yes",
    2406390: "yes",
    2406642: "yes",
    2406717: "no",
    2406901: "no",
    2406916: "no",
    2407206: "yes",
    2408152: "yes",
    2408623: "yes",
    2408970: "no",
    2409203: "yes",
    2409661: "yes",
    2410268: "yes",
    2411972: "no",
}

MEDIUM_CONFIDENCE_IDS = {
    2393697,
    2401850,
    2402350,
    2403851,
    2404745,
    2405568,
    2407206,
    2408970,
    2411972,
}
LOW_CONFIDENCE_IDS = {1592318}

CUSTOM_NOTES = {
    1592318: (
        "冲突干预回答否认具体地点/时间但没有直接处理moment/event；"
        "干净回答Carlisle与参考near the bus粒度不同，保留uncertain。"
    ),
    2395656: "冲突仍回答虚构侧板编号；干净干预回答snow而参考为skis，倾向原生。",
    2401084: "原生正确表示无法确定具体时间，干预直接猜night，属于冲突分支伤害。",
    2402736: "干预把图中的train误称bus；原生明确纠正对象，应该保留原生。",
    2404745: (
        "两种冲突回答都正确；干净分支原生给出拍照目的，干预只说无法判断，"
        "按参考答案倾向原生。"
    ),
    2406916: "冲突两者均正确；干净干预回答suit而参考为skirt，倾向原生。",
    2411972: (
        "冲突两者均正确；干净原生提到French fries，干预只答sandwich，"
        "但答案粒度需人工复核。"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=PROJECT_ROOT
        / "annotations/exp003_action_semantic_review_43.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace AI suggestions, but never human confirmations.",
    )
    return parser.parse_args()


def _default_note(conflict_effect: str, clean_effect: str) -> str:
    return (
        f"冲突分支={conflict_effect}；干净分支={clean_effect}。"
        "判断依据为已筛除问题样本后的官方参考与两动作回答核心语义，"
        "仍需人工结合图像确认。"
    )


def main() -> None:
    args = parse_args()
    with args.review_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != 43:
        raise ValueError(f"Expected 43 rows, found {len(rows)}")
    confirmed = [
        row for row in rows if row.get("human_confirmation", "").strip()
    ]
    if confirmed:
        raise ValueError(
            f"Refusing to alter AI labels after {len(confirmed)} confirmations"
        )
    if not args.force and any(
        row.get("ai_review_protocol", "") == PROTOCOL for row in rows
    ):
        raise FileExistsError("Complete AI pre-review already exists; use --force")

    ids = {int(row["image_id"]) for row in rows}
    existing_transition_ids = ids - set(INTERVENTION_CONFLICT)
    if len(existing_transition_ids) != 14:
        raise ValueError(
            "Expected 14 existing conflict transition judgments, found "
            f"{len(existing_transition_ids)}"
        )
    if ids != set(INTERVENTION_CLEAN):
        raise ValueError("Clean judgments do not exactly match review IDs")

    for row in rows:
        image_id = int(row["image_id"])
        intervention_conflict = (
            INTERVENTION_CONFLICT[image_id]
            if image_id in INTERVENTION_CONFLICT
            else row["ai_intervention_conflict_correct"].strip().lower()
        )
        native_conflict = row["ai_native_conflict_correct"].strip().lower()
        native_clean = row["ai_native_clean_correct"].strip().lower()
        intervention_clean = INTERVENTION_CLEAN[image_id]
        conflict_effect = infer_action_effect(
            native_conflict,
            intervention_conflict,
        )
        clean_effect = infer_action_effect(native_clean, intervention_clean)
        overall = infer_overall_action(conflict_effect, clean_effect)
        confidence = (
            "low"
            if image_id in LOW_CONFIDENCE_IDS
            else "medium"
            if image_id in MEDIUM_CONFIDENCE_IDS
            else "high"
        )
        row.update(
            {
                "ai_intervention_conflict_correct": intervention_conflict,
                "ai_conflict_action_effect": conflict_effect,
                "ai_intervention_clean_correct": intervention_clean,
                "ai_clean_action_effect": clean_effect,
                "ai_overall_action": overall,
                "ai_confidence": confidence,
                "ai_notes": CUSTOM_NOTES.get(
                    image_id,
                    _default_note(conflict_effect, clean_effect),
                ),
                "ai_review_protocol": PROTOCOL,
            }
        )
    validate_action_review_records(rows)
    with args.review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"completed_ai_prereview={len(rows)} high="
        f"{sum(row['ai_confidence'] == 'high' for row in rows)} medium="
        f"{sum(row['ai_confidence'] == 'medium' for row in rows)} low="
        f"{sum(row['ai_confidence'] == 'low' for row in rows)}"
    )


if __name__ == "__main__":
    main()
