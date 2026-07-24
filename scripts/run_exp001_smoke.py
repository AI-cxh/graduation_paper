#!/usr/bin/env python3
"""Run the fixed EXP-001 smoke subset with Qwen2.5-VL."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mmmc_data import load_arrow_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/exp001_baseline.yaml",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/exp001_sample_ids.jsonl",
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
        / "outputs/predictions/exp001/qwen2_5_vl_3b_smoke.jsonl",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_smoke_manifest(path: Path, max_pairs: int | None = None) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if record["in_smoke"]]
    records.sort(key=lambda record: record["baseline_order"])
    return records if max_pairs is None else records[:max_pairs]


def condition_records(
    manifest: list[dict[str, Any]],
    dataset: Any,
) -> dict[str, list[dict[str, Any]]]:
    result = {
        "multimodal_conflict": [],
        "text_only_conflict": [],
        "multimodal_clean": [],
    }
    for pair in manifest:
        conflict = dataset[int(pair["conflict_index"])]
        clean = dataset[int(pair["clean_index"])]
        common = {
            "image_id": int(pair["image_id"]),
            "conflict_type": pair["conflict_type"],
            "baseline_order": int(pair["baseline_order"]),
        }
        result["multimodal_conflict"].append(
            {
                **common,
                "condition": "multimodal_conflict",
                "dataset_index": int(pair["conflict_index"]),
                "question": conflict["question"],
                "reference_answer": conflict["answer"],
                "image": conflict["image"].convert("RGB"),
            }
        )
        result["text_only_conflict"].append(
            {
                **common,
                "condition": "text_only_conflict",
                "dataset_index": int(pair["conflict_index"]),
                "question": conflict["question"],
                "reference_answer": conflict["answer"],
                "image": None,
            }
        )
        result["multimodal_clean"].append(
            {
                **common,
                "condition": "multimodal_clean",
                "dataset_index": int(pair["clean_index"]),
                "question": clean["question"],
                "reference_answer": clean["answer"],
                "image": clean["image"].convert("RGB"),
            }
        )
    return result


def build_message(question: str, with_image: bool) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = []
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def existing_keys(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            keys.add((int(record["image_id"]), record["condition"]))
    return keys


def run_generation(
    *,
    model: Any,
    processor: Any,
    records: list[dict[str, Any]],
    batch_size: int,
    device: str,
    generation_config: dict[str, Any],
    output_path: Path,
    completed: set[tuple[int, str]],
    model_metadata: dict[str, Any],
) -> None:
    records = [
        record
        for record in records
        if (record["image_id"], record["condition"]) not in completed
    ]
    for batch in batched(records, batch_size):
        has_image = batch[0]["image"] is not None
        if any((record["image"] is not None) != has_image for record in batch):
            raise ValueError("A generation batch cannot mix image and text-only records")

        messages = [
            build_message(record["question"], with_image=has_image) for record in batch
        ]
        prompts = [
            processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        images = [record["image"] for record in batch] if has_image else None
        inputs = processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)

        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=bool(generation_config["do_sample"]),
                max_new_tokens=int(generation_config["max_new_tokens"]),
                return_dict_in_generate=True,
                output_scores=True,
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        input_width = inputs["input_ids"].shape[1]
        generated_ids = generated.sequences[:, input_width:]
        texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        transition_scores = model.compute_transition_scores(
            generated.sequences,
            generated.scores,
            normalize_logits=True,
        )
        peak_memory = torch.cuda.max_memory_allocated(device)

        with output_path.open("a", encoding="utf-8") as handle:
            for row_index, (record, text) in enumerate(zip(batch, texts, strict=True)):
                token_scores = transition_scores[row_index].detach().float().cpu()
                valid_scores = token_scores[token_scores != 0]
                result = {
                    key: value for key, value in record.items() if key != "image"
                }
                result.update(
                    {
                        "generated_text": text.strip(),
                        "generated_token_count": int(generated_ids[row_index].numel()),
                        "sum_transition_logprob": (
                            float(valid_scores.sum()) if valid_scores.numel() else None
                        ),
                        "mean_transition_logprob": (
                            float(valid_scores.mean()) if valid_scores.numel() else None
                        ),
                        "batch_seconds": elapsed,
                        "seconds_per_item": elapsed / len(batch),
                        "batch_peak_memory_bytes": int(peak_memory),
                        "prompt_version": generation_config["prompt_version"],
                        **model_metadata,
                    }
                )
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(config["inference"]["batch_size"])
    )
    manifest = load_smoke_manifest(args.manifest, args.max_pairs)
    dataset = load_arrow_split(args.split_dir)
    by_condition = condition_records(manifest, dataset)

    summary = {
        condition: len(records) for condition, records in by_condition.items()
    }
    print(json.dumps({"pairs": len(manifest), "conditions": summary}, ensure_ascii=False))
    for condition, records in by_condition.items():
        if records:
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "example_image_id": records[0]["image_id"],
                        "example_question": records[0]["question"],
                    },
                    ensure_ascii=False,
                )
            )
    if args.dry_run:
        return

    required_files = [
        args.model_path / "config.json",
        args.model_path / "model.safetensors.index.json",
        args.model_path / "model-00001-of-00002.safetensors",
        args.model_path / "model-00002-of-00002.safetensors",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Model download is incomplete: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-001 smoke inference")

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
    processor.tokenizer.padding_side = config["inference"]["tokenizer_padding_side"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=config["model"]["attention"],
        local_files_only=True,
    ).to(args.device)
    if not bool(config["inference"]["do_sample"]):
        model.generation_config.temperature = None
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_keys(args.output)
    metadata = {
        "experiment_id": config["experiment"]["id"],
        "model_repository": config["model"]["repository"],
        "model_revision": config["model"]["revision"],
        "run_started_at": datetime.now().astimezone().isoformat(),
        "device": args.device,
        "dtype": config["model"]["dtype"],
        "attention": config["model"]["attention"],
        "batch_size": batch_size,
        "processor_use_fast": bool(config["vision"]["processor_use_fast"]),
        "tokenizer_padding_side": config["inference"]["tokenizer_padding_side"],
        "vision_min_pixels": int(config["vision"]["min_pixels"]),
        "vision_max_pixels": int(config["vision"]["max_pixels"]),
    }
    for condition in config["inference"]["conditions"]:
        run_generation(
            model=model,
            processor=processor,
            records=by_condition[condition],
            batch_size=batch_size,
            device=args.device,
            generation_config=config["inference"],
            output_path=args.output,
            completed=completed,
            model_metadata=metadata,
        )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
