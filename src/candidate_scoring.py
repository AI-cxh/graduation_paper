"""Teacher-forced candidate scoring and paired summaries for EXP-001."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import comb, exp
from statistics import mean, median, pstdev
import time
from typing import Any

import torch


SCORING_PROTOCOL_VERSION = "exp001-reference-logprob-v1"
SCORE_CONDITIONS = (
    "multimodal_conflict",
    "text_only_conflict",
    "multimodal_clean",
    "text_only_clean",
)


def build_chat_message(
    question: str,
    with_image: bool,
) -> list[dict[str, Any]]:
    """Build the task-native user message used by candidate scoring."""

    content: list[dict[str, str]] = []
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def encode_candidate_text(
    processor: Any,
    text: str,
    image: Any | None,
) -> Any:
    """Encode one prompt or prompt-candidate sequence without padding."""

    kwargs: dict[str, Any] = {
        "text": [text],
        "padding": False,
        "return_tensors": "pt",
    }
    if image is not None:
        kwargs["images"] = [image]
    return processor(**kwargs)


def score_candidate_record(
    *,
    record: Mapping[str, Any],
    model: Any,
    processor: Any,
    device: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Teacher-force one candidate and return token-aligned log probabilities."""

    image = record.get("image")
    message = build_chat_message(
        str(record["question"]),
        with_image=image is not None,
    )
    prompt_text = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = f"{prompt_text}{record['candidate_answer']}"
    prompt_inputs = encode_candidate_text(processor, prompt_text, image)
    full_inputs = encode_candidate_text(processor, full_text, image)
    prompt_length = int(prompt_inputs["input_ids"].shape[1])
    full_length = int(full_inputs["input_ids"].shape[1])
    if full_length <= prompt_length:
        raise ValueError(
            f"Candidate produced no tokens for image_id={record['image_id']}, "
            f"condition={record['condition']}"
        )
    if not torch.equal(
        prompt_inputs["input_ids"][0],
        full_inputs["input_ids"][0, :prompt_length],
    ):
        raise ValueError(
            f"Prompt is not a token prefix for image_id={record['image_id']}, "
            f"condition={record['condition']}"
        )

    full_inputs = full_inputs.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**full_inputs, use_cache=False, return_dict=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    stats = candidate_logprob_statistics(
        outputs.logits,
        full_inputs["input_ids"],
        prompt_length,
    )
    result = {key: value for key, value in record.items() if key != "image"}
    result.update(
        {
            **stats,
            **metadata,
            "prompt_token_count": prompt_length,
            "full_token_count": full_length,
            "score_seconds": elapsed,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    )
    return result


def candidate_logprob_statistics(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_length: int,
) -> dict[str, Any]:
    """Score tokens after ``prompt_length`` using causal next-token logits."""

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("Expected logits [batch, seq, vocab] and input_ids [batch, seq]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(
            f"Logit/input shape mismatch: {tuple(logits.shape)} vs "
            f"{tuple(input_ids.shape)}"
        )
    if input_ids.shape[0] != 1:
        raise ValueError("Candidate scoring currently requires batch_size=1")
    sequence_length = int(input_ids.shape[1])
    if prompt_length < 1 or prompt_length >= sequence_length:
        raise ValueError(
            f"prompt_length must be in [1, {sequence_length - 1}], "
            f"got {prompt_length}"
        )

    targets = input_ids[0, prompt_length:]
    predicting_logits = logits[0, prompt_length - 1 : sequence_length - 1].float()
    token_logprobs = torch.log_softmax(predicting_logits, dim=-1).gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    if token_logprobs.numel() != targets.numel():
        raise RuntimeError("Candidate token/log-prob alignment failed")

    token_logprobs_cpu = token_logprobs.detach().cpu()
    token_count = int(token_logprobs_cpu.numel())
    sum_logprob = float(token_logprobs_cpu.sum())
    mean_logprob = float(token_logprobs_cpu.mean())
    return {
        "candidate_token_count": token_count,
        "candidate_sum_logprob": sum_logprob,
        "candidate_mean_logprob": mean_logprob,
        "candidate_min_token_logprob": float(token_logprobs_cpu.min()),
        "candidate_max_token_logprob": float(token_logprobs_cpu.max()),
        "candidate_perplexity": exp(min(80.0, -mean_logprob)),
        "candidate_token_ids": [int(token) for token in targets.detach().cpu()],
        "candidate_token_logprobs": [
            float(value) for value in token_logprobs_cpu
        ],
    }


def _exact_sign_test_p_value(positive: int, negative: int) -> float:
    nonzero = positive + negative
    if nonzero == 0:
        return 1.0
    lower_tail = sum(
        comb(nonzero, index) for index in range(min(positive, negative) + 1)
    ) / (2**nonzero)
    return min(1.0, 2 * lower_tail)


def summarize_delta_values(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty delta sequence")
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    zero = len(values) - positive - negative
    nonzero = positive + negative
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "population_std": pstdev(values),
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": zero,
        "positive_rate": positive / len(values),
        "positive_rate_among_nonzero": positive / nonzero if nonzero else 0.0,
        "exact_sign_test_p_value": _exact_sign_test_p_value(positive, negative),
    }


def build_pair_deltas(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Align four score conditions and compute per-pair visual increments."""

    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        image_id = int(record["image_id"])
        condition = str(record["condition"])
        if condition not in SCORE_CONDITIONS:
            raise ValueError(f"Unexpected score condition: {condition}")
        if condition in grouped.setdefault(image_id, {}):
            raise ValueError(f"Duplicate score key: {(image_id, condition)}")
        grouped[image_id][condition] = record

    results = []
    for image_id in sorted(grouped):
        conditions = grouped[image_id]
        missing = set(SCORE_CONDITIONS).difference(conditions)
        if missing:
            raise ValueError(
                f"image_id={image_id} is missing score conditions: {sorted(missing)}"
            )
        conflict_multimodal = conditions["multimodal_conflict"]
        conflict_text_only = conditions["text_only_conflict"]
        clean_multimodal = conditions["multimodal_clean"]
        clean_text_only = conditions["text_only_clean"]

        if (
            conflict_multimodal["candidate_answer"]
            != conflict_text_only["candidate_answer"]
        ):
            raise ValueError(f"Conflict candidate mismatch for image_id={image_id}")
        if clean_multimodal["candidate_answer"] != clean_text_only["candidate_answer"]:
            raise ValueError(f"Clean candidate mismatch for image_id={image_id}")
        if (
            conflict_multimodal["candidate_token_count"]
            != conflict_text_only["candidate_token_count"]
        ):
            raise ValueError(f"Conflict token-count mismatch for image_id={image_id}")
        if (
            clean_multimodal["candidate_token_count"]
            != clean_text_only["candidate_token_count"]
        ):
            raise ValueError(f"Clean token-count mismatch for image_id={image_id}")

        conflict_delta_mean = float(
            conflict_multimodal["candidate_mean_logprob"]
        ) - float(conflict_text_only["candidate_mean_logprob"])
        clean_delta_mean = float(clean_multimodal["candidate_mean_logprob"]) - float(
            clean_text_only["candidate_mean_logprob"]
        )
        conflict_delta_sum = float(
            conflict_multimodal["candidate_sum_logprob"]
        ) - float(conflict_text_only["candidate_sum_logprob"])
        clean_delta_sum = float(clean_multimodal["candidate_sum_logprob"]) - float(
            clean_text_only["candidate_sum_logprob"]
        )
        results.append(
            {
                "image_id": image_id,
                "conflict_type": str(conflict_multimodal["conflict_type"]),
                "conflict_candidate_answer": conflict_multimodal[
                    "candidate_answer"
                ],
                "clean_candidate_answer": clean_multimodal["candidate_answer"],
                "conflict_candidate_token_count": int(
                    conflict_multimodal["candidate_token_count"]
                ),
                "clean_candidate_token_count": int(
                    clean_multimodal["candidate_token_count"]
                ),
                "conflict_visual_delta_mean_logprob": conflict_delta_mean,
                "clean_visual_delta_mean_logprob": clean_delta_mean,
                "conflict_minus_clean_visual_delta_mean_logprob": (
                    conflict_delta_mean - clean_delta_mean
                ),
                "conflict_visual_delta_sum_logprob": conflict_delta_sum,
                "clean_visual_delta_sum_logprob": clean_delta_sum,
                "conflict_minus_clean_visual_delta_sum_logprob": (
                    conflict_delta_sum - clean_delta_sum
                ),
            }
        )
    return results


def _raw_condition_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[float]] = {}
    for record in records:
        by_condition.setdefault(str(record["condition"]), []).append(
            float(record["candidate_mean_logprob"])
        )
    return {
        condition: {
            "count": len(values),
            "mean_candidate_mean_logprob": mean(values),
            "median_candidate_mean_logprob": median(values),
            "mean_candidate_perplexity": mean(
                float(record["candidate_perplexity"])
                for record in records
                if record["condition"] == condition
            ),
        }
        for condition, values in sorted(by_condition.items())
    }


def _delta_summary_for_pairs(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "conflict_visual_delta_mean_logprob": summarize_delta_values(
            [
                float(pair["conflict_visual_delta_mean_logprob"])
                for pair in pairs
            ]
        ),
        "clean_visual_delta_mean_logprob": summarize_delta_values(
            [float(pair["clean_visual_delta_mean_logprob"]) for pair in pairs]
        ),
        "conflict_minus_clean_visual_delta_mean_logprob": summarize_delta_values(
            [
                float(pair["conflict_minus_clean_visual_delta_mean_logprob"])
                for pair in pairs
            ]
        ),
    }


def summarize_reference_scores(
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return per-pair deltas and aggregate summaries."""

    records = list(records)
    if not records:
        raise ValueError("No candidate score records supplied")
    pairs = build_pair_deltas(records)
    type_counts = Counter(str(pair["conflict_type"]) for pair in pairs)
    summary = {
        "scoring_protocol": SCORING_PROTOCOL_VERSION,
        "score_scope": (
            "Teacher-forced length-normalized log-probability of the official "
            "reference answer. Visual deltas compare identical question-answer "
            "candidates and are not semantic accuracy."
        ),
        "record_count": len(records),
        "pair_count": len(pairs),
        "condition_counts": dict(
            sorted(Counter(str(record["condition"]) for record in records).items())
        ),
        "conflict_type_counts": dict(sorted(type_counts.items())),
        "raw_scores_by_condition": _raw_condition_summary(records),
        "paired_deltas": _delta_summary_for_pairs(pairs),
        "paired_deltas_by_conflict_type": {
            conflict_type: _delta_summary_for_pairs(
                [
                    pair
                    for pair in pairs
                    if str(pair["conflict_type"]) == conflict_type
                ]
            )
            for conflict_type in sorted(type_counts)
        },
    }
    return pairs, summary


def _binary_roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [
        score for score, label in zip(scores, labels, strict=True) if not label
    ]
    if not positives or not negatives:
        raise ValueError("ROC AUC requires both positive and negative labels")
    favorable = 0.0
    for positive_score in positives:
        for negative_score in negatives:
            if positive_score > negative_score:
                favorable += 1.0
            elif positive_score == negative_score:
                favorable += 0.5
    return favorable / (len(positives) * len(negatives))


def _score_group_summary(
    grouped_scores: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    return {
        group: {
            "count": len(values),
            "mean_conflict_visual_delta_mean_logprob": mean(values),
            "median_conflict_visual_delta_mean_logprob": median(values),
        }
        for group, values in sorted(grouped_scores.items())
        if values
    }


def align_scores_with_free_generation(
    pairs: Sequence[Mapping[str, Any]],
    evaluated_generation_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Relate candidate-score deltas to paired free-generation stance proxies."""

    generation = {
        (int(record["image_id"]), str(record["condition"])): record
        for record in evaluated_generation_records
        if record["condition"] in {"multimodal_conflict", "text_only_conflict"}
    }
    grouped_scores: dict[str, list[float]] = {}
    scores = []
    multimodal_reject_labels = []
    multimodal_only_labels = []
    for pair in pairs:
        image_id = int(pair["image_id"])
        multimodal_key = (image_id, "multimodal_conflict")
        text_only_key = (image_id, "text_only_conflict")
        if multimodal_key not in generation or text_only_key not in generation:
            raise ValueError(f"Missing evaluated generation stance for image_id={image_id}")
        multimodal_rejects = (
            generation[multimodal_key]["stance"] == "reject_premise"
        )
        text_only_rejects = generation[text_only_key]["stance"] == "reject_premise"
        if multimodal_rejects and text_only_rejects:
            transition = "both_reject"
        elif multimodal_rejects:
            transition = "multimodal_only_reject"
        elif text_only_rejects:
            transition = "text_only_only_reject"
        else:
            transition = "neither_reject"

        score = float(pair["conflict_visual_delta_mean_logprob"])
        grouped_scores.setdefault(transition, []).append(score)
        scores.append(score)
        multimodal_reject_labels.append(multimodal_rejects)
        multimodal_only_labels.append(
            multimodal_rejects and not text_only_rejects
        )

    return {
        "scope": (
            "Association between the reference-answer visual log-probability "
            "delta and rule-based free-generation stance proxies; not semantic "
            "accuracy prediction."
        ),
        "pair_count": len(pairs),
        "by_free_generation_transition": _score_group_summary(grouped_scores),
        "auroc_for_multimodal_explicit_rejection": _binary_roc_auc(
            scores, multimodal_reject_labels
        ),
        "auroc_for_multimodal_only_rejection_transition": _binary_roc_auc(
            scores, multimodal_only_labels
        ),
    }
