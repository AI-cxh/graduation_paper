#!/usr/bin/env python3
"""Prepare HaloQuest visual-challenge and insufficient-context controls."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from haloquest_data import (  # noqa: E402
    select_control_eval,
    summarize_haloquest_manifest,
    validate_available_image_paths,
)
from prepare_haloquest_eval import (  # noqa: E402
    METADATA_URL,
    OFFICIAL_COMMIT,
    download_image,
    download_metadata,
    write_jsonl,
)


PROTOCOL = "haloquest-official-non-false-premise-control-eval-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/haloquest",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/haloquest_control_eval.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/haloquest_control_eval_summary.json",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.raw_dir / "haloquest-eval.csv"
    metadata_sha256 = download_metadata(metadata_path, args.timeout)
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        records = select_control_eval(csv.DictReader(handle))
    if len(records) != 304:
        raise ValueError(f"Expected 304 control rows, found {len(records)}")

    image_dir = args.raw_dir / "images"
    unique_images = {}
    for record in records:
        image_name = str(record["image_name"])
        existing = unique_images.get(image_name)
        if existing and existing["source_url"] != record["source_url"]:
            raise ValueError(
                f"Image {image_name} has inconsistent source URLs"
            )
        unique_images[image_name] = record

    downloaded = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_image,
                record,
                image_dir,
                retries=args.retries,
                timeout=args.timeout,
            )
            for record in unique_images.values()
        ]
        for future in as_completed(futures):
            downloaded.append(future.result())
    download_lookup = {
        str(record["image_name"]): record for record in downloaded
    }
    download_fields = (
        "local_image_path",
        "download_status",
        "image_sha256",
        "width",
        "height",
        "download_error",
    )
    completed = []
    for record in records:
        merged = dict(record)
        result = download_lookup[str(record["image_name"])]
        merged.update({field: result[field] for field in download_fields})
        completed.append(merged)
    completed.sort(key=lambda record: int(record["haloquest_id"]))
    validate_available_image_paths(completed, PROJECT_ROOT)
    write_jsonl(args.manifest, completed)
    summary = summarize_haloquest_manifest(
        completed,
        protocol=PROTOCOL,
    )
    summary.update(
        {
            "official_repository_commit": OFFICIAL_COMMIT,
            "metadata_url": METADATA_URL,
            "metadata_sha256": metadata_sha256,
            "unique_image_count": len(unique_images),
            "manifest_path": str(args.manifest.relative_to(PROJECT_ROOT)),
        }
    )
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

