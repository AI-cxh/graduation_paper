#!/usr/bin/env python3
"""Generate premise-verification intervention outputs on confirmed-valid pairs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from confirmed_sensitivity import confirmed_dataset_exclusions  # noqa: E402
from intervention_prompts import (  # noqa: E402
    INTERVENTION_PROTOCOL_VERSION,
    apply_prompt_intervention,
)
from mmmc_data import load_arrow_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/exp003_intervention.yaml",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/exp001_sample_ids.jsonl",
    )
    parser.add_argument(
        "--prereview",
        type=Path,
        default=PROJECT_ROOT / "annotations/exp001_smoke_ai_prereview.csv",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/mmmc/test",
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
        / "outputs/predictions/exp003/"
        "qwen2_5_vl_3b_premise_verification.jsonl",
    )
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_records(
    manifest_path: Path,
    prereview_path: Path,
    dataset: Any,
    max_pairs: int | None,
) -> list[dict[str, Any]]:
    manifest = [
        record
        for record in read_jsonl(manifest_path)
        if bool(record["in_smoke"])
    ]
    manifest.sort(key=lambda record: int(record["baseline_order"]))
    with prereview_path.open(encoding="utf-8", newline="") as handle:
        exclusions = confirmed_dataset_exclusions(list(csv.DictReader(handle)))
    manifest = [
        pair for pair in manifest if int(pair["image_id"]) not in exclusions
    ]
    if max_pairs is not None:
        manifest = manifest[:max_pairs]

    records = []
    for pair in manifest:
        for condition, dataset_index in (
            ("multimodal_conflict", int(pair["conflict_index"])),
            ("multimodal_clean", int(pair["clean_index"])),
        ):
            row = dataset[dataset_index]
            records.append(
                {
                    "image_id": int(pair["image_id"]),
                    "conflict_type": str(pair["conflict_type"]),
                    "baseline_order": int(pair["baseline_order"]),
                    "condition": condition,
                    "action": "premise_verification",
                    "dataset_index": dataset_index,
                    "question": str(row["question"]),
                    "prompted_question": apply_prompt_intervention(
                        str(row["question"]),
                        "premise_verification",
                    ),
                    "reference_answer": str(row["answer"]),
                    "image": row["image"].convert("RGB"),
                }
            )
    return records


def existing_keys(path: Path) -> set[tuple[int, str, str]]:
    if not path.is_file():
        return set()
    return {
        (
            int(record["image_id"]),
            str(record["condition"]),
            str(record["action"]),
        )
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
    dataset = load_arrow_split(args.split_dir)
    records = load_records(
        args.manifest,
        args.prereview,
        dataset,
        args.max_pairs,
    )
    print(
        json.dumps(
            {
                "protocol": INTERVENTION_PROTOCOL_VERSION,
                "pairs": len(records) // 2,
                "records": len(records),
                "condition_counts": {
                    condition: sum(
                        record["condition"] == condition for record in records
                    )
                    for condition in ("multimodal_conflict", "multimodal_clean")
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-003 generation")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_keys(args.output)
    pending = [
        record
        for record in records
        if (
            int(record["image_id"]),
            str(record["condition"]),
            str(record["action"]),
        )
        not in completed
    ]
    if not pending:
        print("all intervention records already exist; skipping model load")
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
            key = (
                int(record["image_id"]),
                str(record["condition"]),
                str(record["action"]),
            )
            if key in completed:
                continue
            message = build_message(record["prompted_question"])
            prompt = processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[prompt],
                images=[record["image"]],
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
                key: value for key, value in record.items() if key != "image"
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
                f"generated={index}/{len(records)} image_id={key[0]} "
                f"condition={key[1]} text={text[:60]!r}",
                flush=True,
            )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
