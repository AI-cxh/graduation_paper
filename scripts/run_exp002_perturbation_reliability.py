#!/usr/bin/env python3
"""Score mild image perturbations and derive EXP-002 reliability features."""

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

from candidate_scoring import score_candidate_record  # noqa: E402
from mmmc_data import load_arrow_split  # noqa: E402
from reliability_features import (  # noqa: E402
    PERTURBATIONS,
    RELIABILITY_PROTOCOL_VERSION,
    apply_image_perturbation,
    build_reliability_features,
    summarize_reliability_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/exp002_reliability.yaml",
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
        "--base-scores",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_reference_scores.jsonl",
    )
    parser.add_argument(
        "--evaluated-generation",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp001/qwen2_5_vl_3b_smoke_scored.jsonl",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp002/qwen2_5_vl_3b_perturbation_scores.jsonl",
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp002/qwen2_5_vl_3b_reliability_features.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/metrics/exp002/qwen2_5_vl_3b_reliability_summary.json",
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_smoke_manifest(
    path: Path,
    max_pairs: int | None,
) -> list[dict[str, Any]]:
    records = [
        record for record in read_jsonl(path) if bool(record["in_smoke"])
    ]
    records.sort(key=lambda record: int(record["baseline_order"]))
    return records if max_pairs is None else records[:max_pairs]


def build_perturbation_records(
    manifest: list[dict[str, Any]],
    dataset: Any,
) -> list[dict[str, Any]]:
    records = []
    for pair in manifest:
        definitions = (
            (
                "multimodal_conflict",
                dataset[int(pair["conflict_index"])],
                int(pair["conflict_index"]),
            ),
            (
                "multimodal_clean",
                dataset[int(pair["clean_index"])],
                int(pair["clean_index"]),
            ),
        )
        for condition, row, dataset_index in definitions:
            original = row["image"].convert("RGB")
            for perturbation in PERTURBATIONS:
                records.append(
                    {
                        "image_id": int(pair["image_id"]),
                        "conflict_type": str(pair["conflict_type"]),
                        "baseline_order": int(pair["baseline_order"]),
                        "condition": condition,
                        "perturbation": perturbation,
                        "dataset_index": dataset_index,
                        "question": str(row["question"]),
                        "candidate_answer": str(row["answer"]).splitlines()[0].strip(),
                        "image": apply_image_perturbation(
                            original,
                            perturbation,
                        ),
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
            str(record["perturbation"]),
        )
        for record in read_jsonl(path)
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = load_smoke_manifest(args.manifest, args.max_pairs)
    selected_ids = {int(pair["image_id"]) for pair in manifest}
    dataset = load_arrow_split(args.split_dir)
    records = build_perturbation_records(manifest, dataset)
    print(
        json.dumps(
            {
                "protocol": RELIABILITY_PROTOCOL_VERSION,
                "pairs": len(manifest),
                "score_records": len(records),
                "perturbations": list(PERTURBATIONS),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for perturbation candidate scoring")
    if not args.base_scores.is_file():
        raise FileNotFoundError(f"Missing base score file: {args.base_scores}")

    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_keys(args.scores_output)
    pending = [
        record
        for record in records
        if (
            int(record["image_id"]),
            str(record["condition"]),
            str(record["perturbation"]),
        )
        not in completed
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
            "scoring_protocol": RELIABILITY_PROTOCOL_VERSION,
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
                key = (
                    int(record["image_id"]),
                    str(record["condition"]),
                    str(record["perturbation"]),
                )
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
                    f"condition={key[1]} perturbation={key[2]} "
                    f"mean_logprob={result['candidate_mean_logprob']:.6f}",
                    flush=True,
                )
    else:
        print("all perturbation scores already exist; skipping model load", flush=True)

    base_records = [
        record
        for record in read_jsonl(args.base_scores)
        if int(record["image_id"]) in selected_ids
    ]
    perturbation_records = [
        record
        for record in read_jsonl(args.scores_output)
        if int(record["image_id"]) in selected_ids
    ]
    features = build_reliability_features(base_records, perturbation_records)
    evaluated_generation = (
        [
            record
            for record in read_jsonl(args.evaluated_generation)
            if int(record["image_id"]) in selected_ids
        ]
        if args.evaluated_generation.is_file()
        else None
    )
    summary = summarize_reliability_features(features, evaluated_generation)
    summary["runtime"] = {
        "new_score_record_count": len(perturbation_records),
        "total_score_seconds": sum(
            float(record["score_seconds"]) for record in perturbation_records
        ),
        "mean_score_seconds": sum(
            float(record["score_seconds"]) for record in perturbation_records
        )
        / len(perturbation_records),
        "peak_memory_bytes": max(
            int(record["peak_memory_bytes"]) for record in perturbation_records
        ),
    }
    write_jsonl(args.features_output, features)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "features_output": str(args.features_output),
                "summary_output": str(args.summary_output),
                "pair_count": len(features),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
