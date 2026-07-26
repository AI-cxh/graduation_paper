from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_exp005_haloquest_baseline import (  # noqa: E402
    expand_actions,
    load_available_records,
)


def test_load_available_records_excludes_unavailable_and_sorts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.jsonl"
    rows = [
        {"haloquest_id": 2, "download_status": "available"},
        {"haloquest_id": 1, "download_status": "available"},
        {"haloquest_id": 3, "download_status": "unavailable"},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    selected = load_available_records(path)
    assert [record["haloquest_id"] for record in selected] == [1, 2]


def test_load_available_records_rejects_nonpositive_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps({"haloquest_id": 1, "download_status": "available"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive"):
        load_available_records(path, max_records=0)


def test_expand_actions_keeps_native_and_adds_fixed_intervention() -> None:
    rows = [{"haloquest_id": 1, "question": "What color is the missing cat?"}]
    expanded = expand_actions(
        rows,
        ["native_prompt", "premise_verification"],
    )
    assert [row["action"] for row in expanded] == [
        "native_prompt",
        "premise_verification",
    ]
    assert expanded[0]["prompted_question"] == rows[0]["question"]
    assert "Check the image before answering" in expanded[1]["prompted_question"]

