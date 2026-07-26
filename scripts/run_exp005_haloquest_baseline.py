#!/usr/bin/env python3
"""Run two fixed prompt actions on the available HaloQuest false-premise eval set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from haloquest_data import validate_available_image_paths  # noqa: E402
from intervention_prompts import (  # noqa: E402
    INTERVENTION_PROTOCOL_VERSION,
    apply_prompt_intervention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/exp005_haloquest.yaml",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data/manifests/haloquest_false_premise_eval.jsonl",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/predictions/exp005/"
        "qwen2_5_vl_3b_native_vs_verification.jsonl",
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_available_records(
    manifest_path: Path,
    *,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    records = [
        record
        for record in read_jsonl(manifest_path)
        if record["download_status"] == "available"
    ]
    records.sort(key=lambda record: int(record["haloquest_id"]))
    if max_records is not None:
        if max_records <= 0:
            raise ValueError("--max-records must be positive")
        records = records[:max_records]
    return records


def expand_actions(
    records: list[dict[str, Any]],
    actions: list[str],
) -> list[dict[str, Any]]:
    expanded = []
    for record in records:
        for action in actions:
            expanded.append(
                {
                    **record,
                    "action": action,
                    "prompted_question": apply_prompt_intervention(
                        str(record["question"]),
                        action,
                    ),
                }
            )
    return expanded


def existing_keys(path: Path) -> set[tuple[int, str]]:
    if not path.is_file():
        return set()
    return {
        (int(record["haloquest_id"]), str(record["action"]))
        for record in read_jsonl(path)
    }


def build_message(prompted_question: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompted_question},
            ],
        }
    ]


def required_model_files(model_path: Path) -> list[Path]:
    return [
        model_path / "config.json",
        model_path / "model.safetensors.index.json",
        model_path / "model-00001-of-00002.safetensors",
        model_path / "model-00002-of-00002.safetensors",
    ]


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_records = load_available_records(
        args.manifest,
        max_records=args.max_records,
    )
    validate_available_image_paths(base_records, PROJECT_ROOT)
    expected = int(config["dataset"]["expected_available_count"])
    if args.max_records is None and len(base_records) != expected:
        raise ValueError(
            f"Available row count changed: expected {expected}, got {len(base_records)}"
        )
    actions = [str(action) for action in config["inference"]["actions"]]
    records = expand_actions(base_records, actions)
    print(
        json.dumps(
            {
                "protocol": config["dataset"]["protocol"],
                "intervention_protocol": INTERVENTION_PROTOCOL_VERSION,
                "available_questions": len(base_records),
                "actions": actions,
                "generation_count": len(records),
                "image_type_counts": {
                    image_type: sum(
                        record["image_type"] == image_type
                        for record in base_records
                    )
                    for image_type in ("generated", "real")
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    missing = [
        str(path) for path in required_model_files(args.model_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Model download is incomplete: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-005 generation")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_keys(args.output)
    pending = [
        record
        for record in records
        if (int(record["haloquest_id"]), str(record["action"])) not in completed
    ]
    if not pending:
        print("all HaloQuest action records already exist; skipping model load")
        return

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(int(config["experiment"]["seed"]))
    torch.cuda.manual_seed_all(int(config["experiment"]["seed"]))
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=int(config["vision"]["min_pixels"]),
        max_pixels=int(config["vision"]["max_pixels"]),
        use_fast=bool(config["vision"]["processor_use_fast"]),
        local_files_only=True,
    )
    processor.tokenizer.padding_side = config["inference"][
        "tokenizer_padding_side"
    ]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=config["model"]["attention"],
        local_files_only=True,
    ).to(args.device)
    model.generation_config.temperature = None
    model.eval()
    metadata = {
        "experiment_id": config["experiment"]["id"],
        "dataset_repository": config["dataset"]["repository"],
        "dataset_revision": config["dataset"]["revision"],
        "dataset_protocol": config["dataset"]["protocol"],
        "intervention_protocol": INTERVENTION_PROTOCOL_VERSION,
        "model_repository": config["model"]["repository"],
        "model_revision": config["model"]["revision"],
        "run_started_at": datetime.now().astimezone().isoformat(),
        "device": args.device,
        "dtype": config["model"]["dtype"],
        "attention": config["model"]["attention"],
        "batch_size": 1,
        "processor_use_fast": bool(config["vision"]["processor_use_fast"]),
        "vision_min_pixels": int(config["vision"]["min_pixels"]),
        "vision_max_pixels": int(config["vision"]["max_pixels"]),
        "do_sample": bool(config["inference"]["do_sample"]),
        "max_new_tokens": int(config["inference"]["max_new_tokens"]),
    }
    with args.output.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            key = (int(record["haloquest_id"]), str(record["action"]))
            if key in completed:
                continue
            with Image.open(PROJECT_ROOT / record["local_image_path"]) as source:
                image = source.convert("RGB")
            message = build_message(str(record["prompted_question"]))
            prompt = processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[prompt],
                images=[image],
                padding=False,
                return_tensors="pt",
            ).to(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=bool(config["inference"]["do_sample"]),
                    max_new_tokens=int(config["inference"]["max_new_tokens"]),
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter() - started
            input_width = inputs["input_ids"].shape[1]
            generated_ids = generated.sequences[:, input_width:]
            text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            transition_scores = model.compute_transition_scores(
                generated.sequences,
                generated.scores,
                normalize_logits=True,
            )[0].detach().float().cpu()
            valid_scores = transition_scores[transition_scores != 0]
            result = {
                field: value
                for field, value in record.items()
                if field not in {"source_url", "download_error"}
            }
            result.update(
                {
                    "generated_text": text,
                    "generated_token_count": int(generated_ids.numel()),
                    "sum_transition_logprob": (
                        float(valid_scores.sum()) if valid_scores.numel() else None
                    ),
                    "mean_transition_logprob": (
                        float(valid_scores.mean()) if valid_scores.numel() else None
                    ),
                    "seconds_per_item": elapsed,
                    "peak_memory_bytes": int(
                        torch.cuda.max_memory_allocated(args.device)
                    ),
                    **metadata,
                }
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"generated={index}/{len(records)} id={key[0]} "
                f"action={key[1]} text={text[:72]!r}",
                flush=True,
            )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

