#!/usr/bin/env python3
"""Download and validate the official HaloQuest false-premise eval subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from haloquest_data import (  # noqa: E402
    select_false_premise_eval,
    summarize_haloquest_manifest,
    validate_available_image_paths,
)


OFFICIAL_COMMIT = "2f643ebb77aeae40f4c5323db38f99f0dc257cf5"
METADATA_URL = (
    "https://raw.githubusercontent.com/google/haloquest/"
    f"{OFFICIAL_COMMIT}/haloquest-eval.csv"
)


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
        / "data/manifests/haloquest_false_premise_eval.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/haloquest_false_premise_eval_summary.json",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def download_metadata(path: Path, timeout: float) -> str:
    response = requests.get(METADATA_URL, timeout=timeout)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return hashlib.sha256(response.content).hexdigest()


def download_image(
    record: dict[str, Any],
    image_dir: Path,
    *,
    retries: int,
    timeout: float,
) -> dict[str, Any]:
    target = image_dir / str(record["image_name"])
    result = dict(record)
    result["local_image_path"] = str(target.relative_to(PROJECT_ROOT))
    if target.is_file():
        try:
            width, height = validate_image(target)
            result.update(
                {
                    "download_status": "available",
                    "image_sha256": sha256_file(target),
                    "width": width,
                    "height": height,
                    "download_error": "",
                }
            )
            return result
        except Exception:
            target.unlink()

    error = "not_attempted"
    status_code: int | None = None
    for attempt in range(retries):
        partial = target.with_suffix(target.suffix + ".part")
        try:
            response = requests.get(
                str(record["source_url"]),
                timeout=timeout,
            )
            status_code = response.status_code
            response.raise_for_status()
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(response.content)
            width, height = validate_image(partial)
            partial.replace(target)
            result.update(
                {
                    "download_status": "available",
                    "image_sha256": sha256_file(target),
                    "width": width,
                    "height": height,
                    "download_error": "",
                }
            )
            return result
        except Exception as exception:
            partial.unlink(missing_ok=True)
            error = f"{type(exception).__name__}: {exception}"
            if status_code in {404, 410}:
                break
            if attempt + 1 < retries:
                time.sleep(min(2 ** (attempt + 1), 16))
    result.update(
        {
            "download_status": "unavailable",
            "image_sha256": "",
            "width": None,
            "height": None,
            "download_error": error,
        }
    )
    return result


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    metadata_path = args.raw_dir / "haloquest-eval.csv"
    metadata_sha256 = download_metadata(metadata_path, args.timeout)
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        records = select_false_premise_eval(csv.DictReader(handle))
    if len(records) != 304:
        raise ValueError(f"Expected 304 false-premise rows, found {len(records)}")

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
    summary = summarize_haloquest_manifest(completed)
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
