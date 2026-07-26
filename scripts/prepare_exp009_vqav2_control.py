#!/usr/bin/env python3
"""Prepare a deterministic, balanced VQAv2 answerable control subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vqav2_control import (  # noqa: E402
    SAMPLE_PROTOCOL,
    select_balanced_vqav2_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/exp009_vqav2_control.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Source hash changed for {path}: {actual}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def image_url(image_id: int) -> str:
    filename = f"COCO_val2014_{image_id:012d}.jpg"
    return f"http://images.cocodataset.org/val2014/{filename}"


def download_image(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    request = urllib.request.Request(
        url, headers={"User-Agent": "graduation-paper-exp009/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def prepare_manifest_record(
    record: dict[str, Any],
    *,
    image_directory: Path,
) -> dict[str, Any]:
    filename = f"COCO_val2014_{int(record['image_id']):012d}.jpg"
    destination = image_directory / filename
    url = image_url(int(record["image_id"]))
    download_error = ""
    try:
        download_image(url, destination)
        with Image.open(destination) as image:
            image.verify()
        with Image.open(destination) as image:
            width, height = image.size
        status = "available"
        image_sha256 = sha256_file(destination)
    except Exception as error:  # noqa: BLE001
        status = "failed"
        image_sha256 = ""
        width = None
        height = None
        download_error = f"{type(error).__name__}: {error}"
    return {
        **record,
        "sample_protocol": SAMPLE_PROTOCOL,
        "source_url": url,
        "local_image_path": str(destination.relative_to(PROJECT_ROOT)),
        "download_status": status,
        "download_error": download_error,
        "image_sha256": image_sha256,
        "width": width,
        "height": height,
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    annotation_path = PROJECT_ROOT / str(config["paths"]["annotations"])
    question_path = PROJECT_ROOT / str(config["paths"]["questions"])
    validate_source(
        annotation_path,
        str(config["dataset"]["annotations_json_sha256"]),
    )
    validate_source(
        question_path,
        str(config["dataset"]["questions_json_sha256"]),
    )
    annotations_document = load_json(annotation_path)
    questions_document = load_json(question_path)
    annotations = annotations_document["annotations"]
    questions = questions_document["questions"]
    expected = int(config["dataset"]["expected_official_question_count"])
    if len(annotations) != expected or len(questions) != expected:
        raise ValueError(
            f"Unexpected official counts: {len(annotations)}, {len(questions)}"
        )
    quotas = {
        str(answer_type): int(count)
        for answer_type, count in config["sample"]["quotas"].items()
    }
    selected = select_balanced_vqav2_sample(
        questions,
        annotations,
        quotas=quotas,
        minimum_consensus=int(
            config["sample"]["minimum_exact_consensus_count"]
        ),
        seed=int(config["experiment"]["seed"]),
    )
    print(
        json.dumps(
            {
                "protocol": SAMPLE_PROTOCOL,
                "official_question_count": len(questions),
                "selected_count": len(selected),
                "answer_type_counts": dict(
                    sorted(
                        Counter(
                            record["answer_type"] for record in selected
                        ).items()
                    )
                ),
                "unique_image_count": len(
                    {record["image_id"] for record in selected}
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    image_directory = PROJECT_ROOT / str(
        config["paths"]["image_directory"]
    )
    manifest = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(
                prepare_manifest_record,
                record,
                image_directory=image_directory,
            ): record
            for record in selected
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            manifest.append(result)
            print(
                f"completed={completed}/{len(selected)} "
                f"question_id={result['question_id']} "
                f"status={result['download_status']}",
                flush=True,
            )
    manifest.sort(key=lambda record: int(record["question_id"]))
    manifest_path = PROJECT_ROOT / str(config["paths"]["manifest"])
    summary_path = PROJECT_ROOT / str(
        config["paths"]["manifest_summary"]
    )
    write_jsonl(manifest_path, manifest)
    summary = {
        "protocol": SAMPLE_PROTOCOL,
        "source": config["dataset"],
        "selection": config["sample"],
        "selected_count": len(manifest),
        "available_count": sum(
            record["download_status"] == "available" for record in manifest
        ),
        "failed_count": sum(
            record["download_status"] == "failed" for record in manifest
        ),
        "answer_type_counts": dict(
            sorted(Counter(record["answer_type"] for record in manifest).items())
        ),
        "unique_image_count": len(
            {record["image_id"] for record in manifest}
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "summary": str(summary_path),
                "available_count": summary["available_count"],
                "failed_count": summary["failed_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
