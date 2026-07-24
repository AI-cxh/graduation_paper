#!/usr/bin/env python3
"""Score official MMMC reference answers under four task-valid conditions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from candidate_scoring import (  # noqa: E402
    SCORING_PROTOCOL_VERSION,
    align_scores_with_free_generation,
    score_candidate_record,
    summarize_reference_scores,
)
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
        "--scores-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_reference_scores.jsonl",
    )
    parser.add_argument(
        "--pair-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_reference_pair_deltas.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_reference_score_summary.json",
    )
    parser.add_argument(
        "--evaluated-generation",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_smoke_scored.jsonl",
    )
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_smoke_manifest(
    path: Path,
    max_pairs: int | None = None,
) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if record["in_smoke"]]
    records.sort(key=lambda record: record["baseline_order"])
    return records if max_pairs is None else records[:max_pairs]


def build_score_records(
    manifest: list[dict[str, Any]],
    dataset: Any,
) -> list[dict[str, Any]]:
    records = []
    for pair in manifest:
        conflict = dataset[int(pair["conflict_index"])]
        clean = dataset[int(pair["clean_index"])]
        common = {
            "image_id": int(pair["image_id"]),
            "conflict_type": str(pair["conflict_type"]),
            "baseline_order": int(pair["baseline_order"]),
        }
        definitions = (
            (
                "multimodal_conflict",
                conflict,
                conflict["image"].convert("RGB"),
            ),
            ("text_only_conflict", conflict, None),
            ("multimodal_clean", clean, clean["image"].convert("RGB")),
            ("text_only_clean", clean, None),
        )
        for condition, row, image in definitions:
            records.append(
                {
                    **common,
                    "condition": condition,
                    "dataset_index": (
                        int(pair["conflict_index"])
                        if "conflict" in condition
                        else int(pair["clean_index"])
                    ),
                    "question": str(row["question"]),
                    "candidate_answer": str(row["answer"]).splitlines()[0].strip(),
                    "image": image,
                }
            )
    return records


def existing_keys(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    return {
        (int(record["image_id"]), str(record["condition"]))
        for record in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = load_smoke_manifest(args.manifest, args.max_pairs)
    dataset = load_arrow_split(args.split_dir)
    records = build_score_records(manifest, dataset)
    print(
        json.dumps(
            {
                "pairs": len(manifest),
                "score_records": len(records),
                "conditions": {
                    condition: sum(
                        record["condition"] == condition for record in records
                    )
                    for condition in sorted(
                        {str(record["condition"]) for record in records}
                    )
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for reference candidate scoring")

    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_keys(args.scores_output)
    pending = [
        record
        for record in records
        if (int(record["image_id"]), str(record["condition"])) not in completed
    ]
    if pending:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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
        model.eval()

        metadata = {
            "scoring_protocol": SCORING_PROTOCOL_VERSION,
            "experiment_id": config["experiment"]["id"],
            "model_repository": config["model"]["repository"],
            "model_revision": config["model"]["revision"],
            "device": args.device,
            "dtype": config["model"]["dtype"],
            "attention": config["model"]["attention"],
            "processor_use_fast": bool(config["vision"]["processor_use_fast"]),
            "vision_min_pixels": int(config["vision"]["min_pixels"]),
            "vision_max_pixels": int(config["vision"]["max_pixels"]),
            "score_run_started_at": datetime.now().astimezone().isoformat(),
        }
        with args.scores_output.open("a", encoding="utf-8") as handle:
            for index, record in enumerate(records, start=1):
                key = (int(record["image_id"]), str(record["condition"]))
                if key in completed:
                    continue
                result = score_candidate_record(
                    record=record,
                    model=model,
                    processor=processor,
                    device=args.device,
                    metadata=metadata,
                )
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"scored={index}/{len(records)} image_id={key[0]} "
                    f"condition={key[1]} "
                    f"mean_logprob={result['candidate_mean_logprob']:.6f}",
                    flush=True,
                )
    else:
        print("all score records already exist; skipping model load", flush=True)

    score_records = [
        json.loads(line)
        for line in args.scores_output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs, summary = summarize_reference_scores(score_records)
    if args.evaluated_generation.is_file():
        evaluated_generation = [
            json.loads(line)
            for line in args.evaluated_generation.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        summary["free_generation_alignment"] = align_scores_with_free_generation(
            pairs,
            evaluated_generation,
        )
    write_jsonl(args.pair_output, pairs)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "scores_output": str(args.scores_output),
                "pair_output": str(args.pair_output),
                "summary_output": str(args.summary_output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
