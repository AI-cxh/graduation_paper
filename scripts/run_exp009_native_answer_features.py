#!/usr/bin/env python3
"""Extract EXP-007-compatible no-reference features on VQAv2 controls."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from candidate_scoring import score_candidate_record  # noqa: E402
from haloquest_native_features import (  # noqa: E402
    PROTOCOL_VERSION,
    SCORE_CONDITIONS,
    build_native_answer_features,
    summarize_native_answer_features,
)
from reliability_features import apply_image_perturbation  # noqa: E402


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
    parser.add_argument("--max-samples", type=int)
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
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def load_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = {
        int(record["question_id"]): record
        for record in read_jsonl(
            PROJECT_ROOT / str(config["paths"]["manifest"])
        )
        if record["download_status"] == "available"
    }
    native = {
        int(record["question_id"]): record
        for record in read_jsonl(
            PROJECT_ROOT / str(config["outputs"]["predictions"])
        )
        if record["action"] == "native_prompt"
    }
    pairs = {
        int(record["question_id"]): record
        for record in read_jsonl(
            PROJECT_ROOT / str(config["outputs"]["paired_results"])
        )
    }
    if not (set(manifest) == set(native) == set(pairs)):
        raise ValueError("EXP-009 manifest, native predictions, and pairs differ")
    samples = []
    for question_id in sorted(manifest):
        source = manifest[question_id]
        prediction = native[question_id]
        pair = pairs[question_id]
        candidate = str(prediction["generated_text"]).strip()
        if not candidate:
            raise ValueError(f"Empty native answer: {question_id}")
        samples.append(
            {
                "haloquest_id": question_id,
                "question_id": question_id,
                "image_id": int(source["image_id"]),
                "dataset_scope": "vqav2_answerable",
                "hallucination_type": str(source["answer_type"]),
                "answer_type": str(source["answer_type"]),
                "question_type": str(source["question_type"]),
                "image_type": "real",
                "local_image_path": str(source["local_image_path"]),
                "image_sha256": str(source["image_sha256"]),
                "question": str(prediction["prompted_question"]),
                "original_question": str(source["question"]),
                "candidate_answer": candidate,
                "native_generated_token_count": int(
                    prediction["generated_token_count"]
                ),
                "native_mean_transition_logprob": float(
                    prediction["mean_transition_logprob"]
                ),
                "action_effect": str(pair["effect"]),
                "native_vqa_soft_accuracy": float(
                    pair["native_vqa_soft_accuracy"]
                ),
                "intervention_vqa_soft_accuracy": float(
                    pair["intervention_vqa_soft_accuracy"]
                ),
                "intervention_minus_native": float(
                    pair["intervention_minus_native"]
                ),
            }
        )
    return samples


def existing_keys(path: Path) -> set[tuple[int, str]]:
    if not path.is_file():
        return set()
    return {
        (int(record["haloquest_id"]), str(record["condition"]))
        for record in read_jsonl(path)
    }


def image_for_condition(
    sample: dict[str, Any], condition: str
) -> Image.Image | None:
    if condition == "text_only":
        return None
    with Image.open(PROJECT_ROOT / str(sample["local_image_path"])) as source:
        image = source.convert("RGB")
    if condition == "multimodal_identity":
        return image
    if condition == "multimodal_jpeg_q75":
        return apply_image_perturbation(image, "jpeg_q75")
    if condition == "multimodal_gaussian_blur_r1":
        return apply_image_perturbation(image, "gaussian_blur_r1")
    raise ValueError(f"Unknown EXP-009 score condition: {condition}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    samples = load_samples(config)
    expected = sum(int(value) for value in config["sample"]["quotas"].values())
    if len(samples) != expected:
        raise ValueError(f"EXP-009 feature sample count changed: {len(samples)}")
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    score_count = len(samples) * len(SCORE_CONDITIONS)
    print(
        json.dumps(
            {
                "protocol": PROTOCOL_VERSION,
                "sample_count": len(samples),
                "score_record_count": score_count,
                "score_conditions": list(SCORE_CONDITIONS),
                "reference_answer_used_as_feature": False,
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-009 feature scoring")

    scores_path = PROJECT_ROOT / str(
        config["outputs"]["native_answer_scores"]
    )
    features_path = PROJECT_ROOT / str(
        config["outputs"]["native_answer_features"]
    )
    summary_path = PROJECT_ROOT / str(
        config["outputs"]["native_answer_feature_summary"]
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_keys(scores_path)
    pending = [
        (sample, condition)
        for sample in samples
        for condition in SCORE_CONDITIONS
        if (int(sample["haloquest_id"]), condition) not in completed
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
        processor.tokenizer.padding_side = str(
            config["inference"]["tokenizer_padding_side"]
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=str(config["model"]["attention"]),
            local_files_only=True,
        ).to(args.device)
        model.eval()
        metadata = {
            "scoring_protocol": PROTOCOL_VERSION,
            "experiment_id": config["experiment"]["id"],
            "model_repository": config["model"]["repository"],
            "model_revision": config["model"]["revision"],
            "device": args.device,
            "dtype": config["model"]["dtype"],
            "attention": config["model"]["attention"],
            "score_run_started_at": datetime.now().astimezone().isoformat(),
            "candidate_source": "native_prompt generated_text",
            "reference_answer_used_as_feature": False,
        }
        with scores_path.open("a", encoding="utf-8") as handle:
            for index, (sample, condition) in enumerate(
                [
                    (sample, condition)
                    for sample in samples
                    for condition in SCORE_CONDITIONS
                ],
                start=1,
            ):
                key = (int(sample["haloquest_id"]), condition)
                if key in completed:
                    continue
                result = score_candidate_record(
                    record={
                        **sample,
                        "condition": condition,
                        "image": image_for_condition(sample, condition),
                    },
                    model=model,
                    processor=processor,
                    device=args.device,
                    metadata=metadata,
                )
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"scored={index}/{score_count} id={key[0]} "
                    f"condition={condition} "
                    f"mean_logprob={result['candidate_mean_logprob']:.6f}",
                    flush=True,
                )
    else:
        print("all EXP-009 scores exist; skipping model load", flush=True)

    selected_ids = {int(sample["haloquest_id"]) for sample in samples}
    scores = [
        record
        for record in read_jsonl(scores_path)
        if int(record["haloquest_id"]) in selected_ids
    ]
    features = build_native_answer_features(samples, scores)
    summary = summarize_native_answer_features(
        features,
        bootstrap_resamples=int(
            config["evaluation"]["bootstrap_resamples"]
        ),
        bootstrap_seed=int(config["experiment"]["seed"]),
    )
    summary["experiment"] = config["experiment"]
    summary["dataset"] = config["dataset"]
    summary["runtime"] = {
        "score_record_count": len(scores),
        "total_score_seconds": sum(
            float(record["score_seconds"]) for record in scores
        ),
        "mean_score_seconds": sum(
            float(record["score_seconds"]) for record in scores
        )
        / len(scores),
        "peak_memory_bytes": max(
            int(record["peak_memory_bytes"]) for record in scores
        ),
    }
    write_jsonl(features_path, features)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "features": str(features_path),
                "summary": str(summary_path),
                "sample_count": len(features),
            }
        )
    )


if __name__ == "__main__":
    main()
