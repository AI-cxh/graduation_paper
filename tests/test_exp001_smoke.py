from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/run_exp001_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_exp001_smoke", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_message() -> None:
    with_image = MODULE.build_message("Question?", with_image=True)
    text_only = MODULE.build_message("Question?", with_image=False)
    assert [item["type"] for item in with_image[0]["content"]] == ["image", "text"]
    assert [item["type"] for item in text_only[0]["content"]] == ["text"]


def test_condition_records_preserve_task_semantics() -> None:
    manifest = [
        {
            "image_id": 7,
            "conflict_type": "object",
            "baseline_order": 0,
            "conflict_index": 0,
            "clean_index": 1,
        }
    ]

    class FakeImage:
        def convert(self, mode: str):
            return f"image-{mode}"

    class FakeDataset:
        rows = [
            {
                "image_id": 7,
                "question": "Conflict question?",
                "answer": "Reject.",
                "image": FakeImage(),
            },
            {
                "image_id": 7,
                "question": "Clean question?",
                "answer": "Answer.",
                "image": FakeImage(),
            },
        ]

        def __getitem__(self, index: int):
            return self.rows[index]

    records = MODULE.condition_records(manifest, FakeDataset())
    assert records["multimodal_conflict"][0]["question"] == "Conflict question?"
    assert records["text_only_conflict"][0]["question"] == "Conflict question?"
    assert records["text_only_conflict"][0]["image"] is None
    assert records["multimodal_clean"][0]["question"] == "Clean question?"
