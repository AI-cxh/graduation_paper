from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from haloquest_data import (  # noqa: E402
    select_false_premise_eval,
    summarize_haloquest_manifest,
)


def _row(
    row_id: int,
    hallucination_type: str = "false premises",
) -> dict[str, str]:
    return {
        "": str(row_id),
        "image_name": f"{row_id}.png",
        "url": f"https://example.test/{row_id}.png",
        "image type": "generated",
        "hallucination type": hallucination_type,
        "question": "What unsupported object is shown?",
        "groundtruth responses": "The object is not present.",
        "split": "eval",
    }


def test_false_premise_selection_normalizes_official_fields() -> None:
    selected = select_false_premise_eval(
        [_row(2), _row(1), _row(3, "visual challenge")]
    )
    assert [record["haloquest_id"] for record in selected] == [1, 2]
    assert selected[0]["groundtruth_response"] == "The object is not present."


def test_duplicate_image_with_distinct_question_ids_is_allowed() -> None:
    first = _row(1)
    second = _row(2)
    second["image_name"] = first["image_name"]
    selected = select_false_premise_eval([first, second])
    assert len(selected) == 2


def test_duplicate_question_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate HaloQuest ID"):
        select_false_premise_eval([_row(1), _row(1)])


def test_manifest_summary_reports_coverage_and_failures() -> None:
    summary = summarize_haloquest_manifest(
        [
            {
                "haloquest_id": 1,
                "image_type": "generated",
                "download_status": "available",
            },
            {
                "haloquest_id": 2,
                "image_type": "real",
                "download_status": "unavailable",
            },
        ]
    )
    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["failed_ids"] == [2]
