"""Parsing and validation helpers for official HaloQuest evaluation metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


OFFICIAL_FIELDS = {
    "",
    "image_name",
    "url",
    "image type",
    "hallucination type",
    "question",
    "groundtruth responses",
    "split",
}
FALSE_PREMISE_LABEL = "false premises"
CONTROL_LABELS = {"visual challenge", "insufficient context"}


def select_haloquest_eval(
    rows: Iterable[Mapping[str, Any]],
    hallucination_types: set[str],
) -> list[dict[str, Any]]:
    """Select and normalize official eval rows for fixed category labels."""

    if not hallucination_types:
        raise ValueError("At least one HaloQuest category is required")

    normalized = []
    seen_ids: set[int] = set()
    for source in rows:
        missing = OFFICIAL_FIELDS - set(source)
        if missing:
            raise ValueError(f"Missing HaloQuest fields: {sorted(missing)}")
        if str(source["split"]).strip() != "eval":
            continue
        hallucination_type = str(source["hallucination type"]).strip()
        if hallucination_type not in hallucination_types:
            continue
        haloquest_id = int(str(source[""]).strip())
        image_name = str(source["image_name"]).strip()
        if haloquest_id in seen_ids:
            raise ValueError(f"Duplicate HaloQuest ID: {haloquest_id}")
        seen_ids.add(haloquest_id)
        question = str(source["question"]).strip()
        answer = str(source["groundtruth responses"]).strip()
        url = str(source["url"]).strip()
        if not question or not answer or not url:
            raise ValueError(f"Incomplete HaloQuest row: {haloquest_id}")
        normalized.append(
            {
                "haloquest_id": haloquest_id,
                "image_name": image_name,
                "source_url": url,
                "image_type": str(source["image type"]).strip(),
                "hallucination_type": hallucination_type,
                "question": question,
                "groundtruth_response": answer,
                "split": "eval",
            }
        )
    return sorted(normalized, key=lambda record: record["haloquest_id"])


def select_false_premise_eval(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select the official false-premise evaluation subset."""

    return select_haloquest_eval(rows, {FALSE_PREMISE_LABEL})


def select_control_eval(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select visual-challenge and insufficient-context control rows."""

    return select_haloquest_eval(rows, CONTROL_LABELS)


def summarize_haloquest_manifest(
    records: Iterable[Mapping[str, Any]],
    *,
    protocol: str = "haloquest-official-false-premise-eval-v1",
) -> dict[str, Any]:
    """Summarize local image coverage without treating failures as examples."""

    records = list(records)
    statuses = Counter(str(record["download_status"]) for record in records)
    available = [
        record for record in records if record["download_status"] == "available"
    ]
    image_types = Counter(str(record["image_type"]) for record in records)
    return {
        "protocol": protocol,
        "row_count": len(records),
        "available_count": len(available),
        "coverage": len(available) / len(records) if records else 0.0,
        "download_status_counts": dict(sorted(statuses.items())),
        "image_type_counts": dict(sorted(image_types.items())),
        "available_image_type_counts": dict(
            sorted(Counter(str(record["image_type"]) for record in available).items())
        ),
        "hallucination_type_counts": dict(
            sorted(
                Counter(str(record["hallucination_type"]) for record in records).items()
            )
        ),
        "available_hallucination_type_counts": dict(
            sorted(
                Counter(
                    str(record["hallucination_type"]) for record in available
                ).items()
            )
        ),
        "failed_ids": [
            int(record["haloquest_id"])
            for record in records
            if record["download_status"] != "available"
        ],
    }


def validate_available_image_paths(
    records: Iterable[Mapping[str, Any]],
    project_root: Path,
) -> None:
    """Ensure every record marked available points to a real local file."""

    for record in records:
        if record["download_status"] != "available":
            continue
        path = project_root / str(record["local_image_path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"Available HaloQuest image is missing: {path}"
            )
