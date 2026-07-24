#!/usr/bin/env python3
"""Export the fixed EXP-001 audit subset to images and an annotation CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mmmc_data import load_arrow_split  # noqa: E402


FIELDNAMES = [
    "image_id",
    "conflict_type",
    "image_file",
    "conflict_question",
    "conflict_answer",
    "clean_question",
    "clean_answer",
    "key_component",
    "key_component_relationships",
    "key_component_attributes",
    "conflict_is_real",
    "conflict_answer_correct",
    "clean_answer_correct",
    "image_quality_ok",
    "template_leakage",
    "issue_type",
    "notes",
    "annotator",
    "audit_date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/exp001_sample_ids.jsonl",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/mmmc/test",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "annotations/exp001_audit_200.csv",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "annotations/exp001_audit_images",
    )
    return parser.parse_args()


def _text_or_json(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit_records = [record for record in records if record["in_audit"]]
    audit_records.sort(key=lambda record: (record["conflict_type"], record["image_id"]))

    dataset = load_arrow_split(args.split_dir)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in audit_records:
        conflict = dataset[int(record["conflict_index"])]
        clean = dataset[int(record["clean_index"])]
        if not (
            conflict["image_id"] == clean["image_id"] == int(record["image_id"])
        ):
            raise ValueError(f"Pair mismatch for image_id={record['image_id']}")

        image_name = f"{int(record['image_id']):08d}.png"
        image_path = args.image_dir / image_name
        if not image_path.exists():
            conflict["image"].convert("RGB").save(image_path)

        rows.append(
            {
                "image_id": record["image_id"],
                "conflict_type": record["conflict_type"],
                "image_file": str(image_path.relative_to(PROJECT_ROOT)),
                "conflict_question": conflict["question"],
                "conflict_answer": conflict["answer"],
                "clean_question": clean["question"],
                "clean_answer": clean["answer"],
                "key_component": _text_or_json(conflict["key_component"]),
                "key_component_relationships": _text_or_json(
                    conflict["key_component_relationships"]
                ),
                "key_component_attributes": _text_or_json(
                    conflict["key_component_attributes"]
                ),
                "conflict_is_real": "",
                "conflict_answer_correct": "",
                "clean_answer_correct": "",
                "image_quality_ok": "",
                "template_leakage": "",
                "issue_type": "",
                "notes": "",
                "annotator": "",
                "audit_date": "",
            }
        )

    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"audit_rows={len(rows)}")
    print(f"csv={args.output_csv}")
    print(f"image_dir={args.image_dir}")


if __name__ == "__main__":
    main()
