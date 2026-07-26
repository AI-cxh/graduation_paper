#!/usr/bin/env python3
"""Run native and premise-verification actions on EXP-009 VQAv2 controls."""

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

from intervention_prompts import (  # noqa: E402
    INTERVENTION_PROTOCOL_VERSION,
    apply_prompt_intervention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/exp009_vqav2_control.yaml",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def existing_keys(path: Path) -> set[tuple[int, str]]:
    if not path.is_file():
        return set()
    return {
        (int(record["question_id"]), str(record["action"]))
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


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest_path = PROJECT_ROOT / str(config["paths"]["manifest"])
    base_records = [
        record
        for record in read_jsonl(manifest_path)
        if record["download_status"] == "available"
    ]
    base_records.sort(key=lambda record: int(record["question_id"]))
    if len(base_records) != sum(
        int(value) for value in config["sample"]["quotas"].values()
    ):
        raise ValueError(f"Unexpected EXP-009 available count: {len(base_records)}")
    if args.max_records is not None:
        if args.max_records <= 0:
            raise ValueError("--max-records must be positive")
        base_records = base_records[: args.max_records]
    actions = [str(action) for action in config["inference"]["actions"]]
    common_output_instruction = str(
        config["inference"]["common_output_instruction"]
    )
    records = [
        {
            **record,
            "action": action,
            "prompted_question": (
                f"{apply_prompt_intervention(str(record['question']), action)}"
                f"\n\n{common_output_instruction}"
            ),
        }
        for record in base_records
        for action in actions
    ]
    print(
        json.dumps(
            {
                "experiment": config["experiment"]["id"],
                "question_count": len(base_records),
                "generation_count": len(records),
                "answer_type_counts": {
                    answer_type: sum(
                        record["answer_type"] == answer_type
                        for record in base_records
                    )
                    for answer_type in ("yes/no", "number", "other")
                },
                "actions": actions,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-009 generation")

    output_path = PROJECT_ROOT / str(config["outputs"]["predictions"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_keys(output_path)
    pending = [
        record
        for record in records
        if (int(record["question_id"]), str(record["action"])) not in completed
    ]
    if not pending:
        print("all EXP-009 generations already exist; skipping model load")
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
    processor.tokenizer.padding_side = str(
        config["inference"]["tokenizer_padding_side"]
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=str(config["model"]["attention"]),
        local_files_only=True,
    ).to(args.device)
    model.generation_config.temperature = None
    model.eval()
    metadata = {
        "experiment_id": config["experiment"]["id"],
        "dataset_name": config["dataset"]["name"],
        "dataset_split": config["dataset"]["split"],
        "sample_protocol": config["sample"]["protocol"],
        "intervention_protocol": INTERVENTION_PROTOCOL_VERSION,
        "model_repository": config["model"]["repository"],
        "model_revision": config["model"]["revision"],
        "run_started_at": datetime.now().astimezone().isoformat(),
        "device": args.device,
        "dtype": config["model"]["dtype"],
        "attention": config["model"]["attention"],
        "batch_size": 1,
        "do_sample": bool(config["inference"]["do_sample"]),
        "max_new_tokens": int(config["inference"]["max_new_tokens"]),
        "vision_min_pixels": int(config["vision"]["min_pixels"]),
        "vision_max_pixels": int(config["vision"]["max_pixels"]),
    }
    with output_path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            key = (int(record["question_id"]), str(record["action"]))
            if key in completed:
                continue
            with Image.open(
                PROJECT_ROOT / str(record["local_image_path"])
            ) as source:
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
            generated_text = processor.batch_decode(
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
                    "generated_text": generated_text,
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
                f"action={key[1]} text={generated_text[:72]!r}",
                flush=True,
            )
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
