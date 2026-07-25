#!/usr/bin/env python3
"""Build a transparent AI pre-review without modifying human label columns."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "exp001-ai-prereview-v1"
AI_REVIEWER = "OpenAI Codex visual review"

# Values: multimodal semantic label, multimodal stance, confidence.
MM_JUDGMENTS = {
    2395592: ("yes", "reject_premise", "high"),
    2395656: ("no", "answer_premise", "high"),
    2404880: ("no", "answer_premise", "high"),
    150293: ("yes", "reject_premise", "high"),
    2399942: ("no", "answer_premise", "high"),
    2408623: ("no", "answer_premise", "high"),
    2394830: ("yes", "reject_premise", "high"),
    2399944: ("no", "answer_premise", "high"),
    2406642: ("yes", "reject_premise", "high"),
    2409203: ("no", "answer_premise", "high"),
    2405435: ("uncertain", "answer_premise", "low"),
    1592318: ("no", "answer_premise", "medium"),
    2402123: ("no", "answer_premise", "high"),
    2393956: ("uncertain", "answer_premise", "low"),
    2406916: ("yes", "reject_premise", "high"),
    2403720: ("yes", "reject_premise", "high"),
    2396016: ("no", "answer_premise", "medium"),
    713590: ("no", "answer_premise", "high"),
    2401850: ("no", "other", "medium"),
    2401710: ("yes", "reject_premise", "high"),
    2411972: ("yes", "reject_premise", "high"),
    2403954: ("no", "answer_premise", "high"),
    2405672: ("uncertain", "reject_premise", "low"),
    2393697: ("no", "answer_premise", "high"),
    2403767: ("yes", "reject_premise", "high"),
    2406901: ("yes", "reject_premise", "high"),
    2396910: ("no", "answer_premise", "high"),
    2401906: ("no", "answer_premise", "high"),
    2402736: ("yes", "reject_premise", "high"),
    2402704: ("uncertain", "answer_premise", "low"),
    2402350: ("no", "answer_premise", "high"),
    2409661: ("no", "answer_premise", "high"),
    2400695: ("no", "answer_premise", "high"),
    2404745: ("yes", "reject_premise", "high"),
    2406717: ("yes", "reject_premise", "high"),
    2408970: ("yes", "reject_premise", "medium"),
    2405342: ("yes", "reject_premise", "high"),
    2396109: ("yes", "reject_premise", "high"),
    2396984: ("no", "answer_premise", "high"),
    2403851: ("yes", "other", "medium"),
    2406390: ("yes", "reject_premise", "high"),
    2395947: ("no", "answer_premise", "high"),
    2401084: ("yes", "abstain", "medium"),
    2410068: ("uncertain", "answer_premise", "low"),
    2403489: ("uncertain", "answer_premise", "low"),
    2405568: ("no", "answer_premise", "high"),
    2410888: ("uncertain", "answer_premise", "low"),
    2410268: ("yes", "reject_premise", "high"),
    2408152: ("yes", "reject_premise", "high"),
    2407206: ("no", "answer_premise", "medium"),
}

# Text-only outputs are stance-only controls; semantic correctness is n/a.
TEXT_ANSWER_PREMISE = {2395592, 2394830, 2403954, 2402704, 2396984, 2410888}
TEXT_REJECT_PREMISE = {2404880}
TEXT_OTHER = {2396016}

# Values: clean semantic label, confidence.
CLEAN_JUDGMENTS = {
    2395592: ("yes", "high"),
    2395656: ("yes", "high"),
    2404880: ("yes", "high"),
    150293: ("yes", "high"),
    2399942: ("no", "high"),
    2408623: ("yes", "high"),
    2394830: ("yes", "high"),
    2399944: ("yes", "high"),
    2406642: ("yes", "high"),
    2409203: ("yes", "high"),
    2405435: ("yes", "high"),
    1592318: ("uncertain", "low"),
    2402123: ("yes", "high"),
    2393956: ("yes", "high"),
    2406916: ("yes", "high"),
    2403720: ("yes", "high"),
    2396016: ("yes", "high"),
    713590: ("yes", "high"),
    2401850: ("yes", "medium"),
    2401710: ("yes", "high"),
    2411972: ("yes", "high"),
    2403954: ("no", "high"),
    2405672: ("yes", "high"),
    2393697: ("yes", "medium"),
    2403767: ("no", "high"),
    2406901: ("no", "high"),
    2396910: ("yes", "high"),
    2401906: ("yes", "high"),
    2402736: ("yes", "high"),
    2402704: ("yes", "high"),
    2402350: ("yes", "medium"),
    2409661: ("yes", "high"),
    2400695: ("yes", "high"),
    2404745: ("yes", "medium"),
    2406717: ("no", "high"),
    2408970: ("no", "high"),
    2405342: ("yes", "high"),
    2396109: ("yes", "high"),
    2396984: ("yes", "high"),
    2403851: ("yes", "high"),
    2406390: ("yes", "high"),
    2395947: ("yes", "high"),
    2401084: ("yes", "high"),
    2410068: ("uncertain", "low"),
    2403489: ("uncertain", "low"),
    2405568: ("uncertain", "low"),
    2410888: ("uncertain", "low"),
    2410268: ("yes", "high"),
    2408152: ("yes", "high"),
    2407206: ("yes", "high"),
}

MM_CUSTOM_NOTES = {
    2405435: "原图清楚存在可称为table/desk的家具，与官方“无table”参考冲突。",
    2393956: "黑熊明显位于画面前景，官方“无前景元素”参考疑似不成立。",
    2405672: "网球运动员明显在前景；模型和官方参考均称无前景对象，样本有效性可疑。",
    2402704: "图中有红色地面及带红色圆形花纹的座椅，官方“无任何红色物体”疑似错误。",
    2410068: "右上角可见剪刀、右下有线缆；模型候选含scissors，官方“右角无物”可疑。",
    2403489: "窗外有低矮山丘，mountain与hill边界主观，不能稳定判定冲突前提。",
    2410888: "图中虽无训练者，但看到马后回答horse trainer在常识问法下合理，官方参考的可答性存疑。",
}
DATASET_VALIDITY_IDS = set(MM_CUSTOM_NOTES)

CLEAN_CUSTOM_NOTES = {
    2399942: "未回答dock，列举beach/road/parking多个互斥地点。",
    1592318: "回答Carlisle而参考为near the Stagecoach bus，粒度不同，需人工决定是否接受。",
    2403954: "回答square-shaped，与参考rectangular及图中长方形餐盘不一致。",
    2403767: "回答short hair，与参考long直接矛盾。",
    2406901: "回答racket在脚边地面，与参考on leg不一致。",
    2406717: "只描述夹克颜色/类型，没有回答wrinkled。",
    2408970: "归因于动物踩踏，与参考when it was wet不一致。",
    2410068: "同时猜测cable/flower/bed；虽包含cable但不是单一明确答案。",
    2403489: "同时猜测window、wall hole和ground hole；只部分覆盖broken window。",
    2405568: "同时猜测flowers/plants/statue，无法确定是否应接受。",
    2410888: "回答horse rider/equestrian而非jockey；语义范围是否可接受需人工决定。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "annotations/exp001_smoke_semantic_review.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "annotations/exp001_smoke_ai_prereview.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an output containing human confirmations.",
    )
    return parser.parse_args()


def text_stance(image_id: int) -> str:
    if image_id in TEXT_ANSWER_PREMISE:
        return "answer_premise"
    if image_id in TEXT_REJECT_PREMISE:
        return "reject_premise"
    if image_id in TEXT_OTHER:
        return "other"
    return "abstain"


def default_mm_note(semantic: str, stance: str) -> str:
    if semantic == "yes" and stance == "reject_premise":
        return "明确指出冲突目标、属性或关系不受图像支持，并给出图像证据。"
    if semantic == "yes":
        return "回答与官方冲突参考语义一致，但不是标准的显式前提拒绝表达。"
    return "接受了错误前提、给出猜测或未解决官方参考指出的前提冲突。"


def default_clean_note(semantic: str) -> str:
    if semantic == "yes":
        return "生成回答与干净参考答案语义一致或给出了兼容的更具体描述。"
    return "生成回答与干净参考答案不一致或没有回答目标问题。"


def main() -> None:
    args = parse_args()
    with args.source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    image_ids = {int(row["image_id"]) for row in rows}
    if len(rows) != 150 or len(image_ids) != 50:
        raise ValueError(
            f"Expected 150 rows and 50 image IDs, got {len(rows)} and "
            f"{len(image_ids)}"
        )
    if image_ids != set(MM_JUDGMENTS) or image_ids != set(CLEAN_JUDGMENTS):
        raise ValueError("Judgment IDs do not exactly match the review table")

    output_rows = []
    for row in rows:
        image_id = int(row["image_id"])
        condition = row["condition"]
        if condition == "multimodal_conflict":
            semantic, stance, confidence = MM_JUDGMENTS[image_id]
            notes = MM_CUSTOM_NOTES.get(
                image_id,
                default_mm_note(semantic, stance),
            )
        elif condition == "text_only_conflict":
            semantic = "n/a"
            stance = text_stance(image_id)
            confidence = "high" if stance == "abstain" else "medium"
            notes = (
                "仅记录无图条件的语言立场，不评价视觉语义正确率；"
                f"该输出归为{stance}。"
            )
        elif condition == "multimodal_clean":
            semantic, confidence = CLEAN_JUDGMENTS[image_id]
            stance = "n/a"
            notes = CLEAN_CUSTOM_NOTES.get(
                image_id,
                default_clean_note(semantic),
            )
        else:
            raise ValueError(f"Unexpected condition: {condition}")
        output_rows.append(
            {
                "image_id": image_id,
                "image_path": (
                    f"annotations/exp001_audit_images/{image_id:08d}.png"
                ),
                "condition": condition,
                "conflict_type": row["conflict_type"],
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "generated_text": row["generated_text"],
                "automatic_stance": row["stance"],
                "manual_review_required": row["manual_review_required"],
                "ai_semantic_correct": semantic,
                "ai_stance": stance,
                "ai_confidence": confidence,
                "ai_issue_type": (
                    "dataset_validity"
                    if condition == "multimodal_conflict"
                    and image_id in DATASET_VALIDITY_IDS
                    else (
                        "answer_ambiguity"
                        if semantic == "uncertain"
                        else ""
                    )
                ),
                "ai_notes": notes,
                "ai_reviewer": AI_REVIEWER,
                "protocol": PROTOCOL,
                "human_confirmation": "",
                "human_correction_semantic": "",
                "human_correction_stance": "",
                "human_notes": "",
            }
        )

    if args.output.is_file() and not args.force:
        with args.output.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
        if any(row.get("human_confirmation", "").strip() for row in existing):
            raise ValueError(
                "Output contains human confirmations; use --force only if "
                "discarding them is intentional"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} AI pre-review rows to {args.output}")


if __name__ == "__main__":
    main()
