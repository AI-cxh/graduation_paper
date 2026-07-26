"""VQAv2 sampling and official-style answer evaluation for EXP-009."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean
from typing import Any

import numpy as np


SAMPLE_PROTOCOL = "exp009-vqav2-balanced-consensus-control-v1"
EVALUATION_PROTOCOL = "dual-vqav2-soft-accuracy-v1"

# Ported from GT-Vision-Lab/VQA at revision
# a013f0043c1e2cdc995922dfe257f7149aa9af06.
# Copyright (c) 2014, Aishwarya Agrawal. BSD-2-Clause terms are retained in
# external/VQA/license.txt and the pinned external source.
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
_MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCTUATION = (
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
)
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_GENERIC_REFUSAL = re.compile(
    r"\b(?:cannot|can't|unable\s+to|not\s+enough\s+information|"
    r"not\s+visible|not\s+shown|does\s+not\s+(?:show|contain|depict)|"
    r"no\s+[^.!?]{0,60}\s+in\s+(?:the\s+)?(?:image|picture))\b",
    re.IGNORECASE,
)


def answer_consensus_count(annotation: Mapping[str, Any]) -> int:
    """Count exact annotator agreement with the official majority answer."""

    majority = str(annotation["multiple_choice_answer"])
    return sum(
        str(answer["answer"]) == majority for answer in annotation["answers"]
    )


def select_balanced_vqav2_sample(
    questions: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    quotas: Mapping[str, int],
    minimum_consensus: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select unique-image, high-consensus records without model outputs."""

    question_by_id = {
        int(question["question_id"]): question for question in questions
    }
    candidates: dict[str, list[tuple[str, Mapping[str, Any]]]] = {
        answer_type: [] for answer_type in quotas
    }
    for annotation in annotations:
        answer_type = str(annotation["answer_type"])
        if answer_type not in candidates:
            continue
        question_id = int(annotation["question_id"])
        if question_id not in question_by_id:
            raise ValueError(f"Annotation missing question: {question_id}")
        consensus = answer_consensus_count(annotation)
        if consensus < minimum_consensus:
            continue
        rank = hashlib.sha256(
            f"{seed}:{question_id}".encode()
        ).hexdigest()
        candidates[answer_type].append((rank, annotation))
    for values in candidates.values():
        values.sort(key=lambda item: item[0])

    selected = []
    selected_images: set[int] = set()
    positions = {answer_type: 0 for answer_type in quotas}
    while any(
        sum(record["answer_type"] == answer_type for record in selected)
        < quota
        for answer_type, quota in quotas.items()
    ):
        made_progress = False
        for answer_type, quota in quotas.items():
            current_count = sum(
                record["answer_type"] == answer_type for record in selected
            )
            if current_count >= quota:
                continue
            values = candidates[answer_type]
            while positions[answer_type] < len(values):
                _, annotation = values[positions[answer_type]]
                positions[answer_type] += 1
                image_id = int(annotation["image_id"])
                if image_id in selected_images:
                    continue
                question_id = int(annotation["question_id"])
                question = question_by_id[question_id]
                answers = [
                    {
                        "answer_id": int(answer["answer_id"]),
                        "answer": str(answer["answer"]),
                        "answer_confidence": str(
                            answer["answer_confidence"]
                        ),
                    }
                    for answer in annotation["answers"]
                ]
                selected.append(
                    {
                        "question_id": question_id,
                        "image_id": image_id,
                        "question": str(question["question"]),
                        "question_type": str(annotation["question_type"]),
                        "answer_type": answer_type,
                        "multiple_choice_answer": str(
                            annotation["multiple_choice_answer"]
                        ),
                        "answers": answers,
                        "exact_consensus_count": answer_consensus_count(
                            annotation
                        ),
                    }
                )
                selected_images.add(image_id)
                made_progress = True
                break
            else:
                raise ValueError(
                    f"Insufficient unique-image candidates for {answer_type}"
                )
        if not made_progress:
            raise ValueError("Balanced sampling made no progress")
    return sorted(selected, key=lambda record: int(record["question_id"]))


def process_vqa_punctuation(text: str) -> str:
    output = text
    for punctuation in _PUNCTUATION:
        if (
            f"{punctuation} " in text
            or f" {punctuation}" in text
            or _COMMA_STRIP.search(text)
        ):
            output = output.replace(punctuation, "")
        else:
            output = output.replace(punctuation, " ")
    return _PERIOD_STRIP.sub("", output)


def process_vqa_digit_article(text: str) -> str:
    words = []
    for word in text.lower().split():
        word = _MANUAL_MAP.get(word, word)
        if word not in _ARTICLES:
            words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def normalize_vqa_answer(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\t", " ").strip()
    return process_vqa_digit_article(process_vqa_punctuation(text))


def vqa_soft_accuracy(
    prediction: str,
    answers: Sequence[Mapping[str, Any]],
) -> float:
    """Reproduce official per-question VQAv2 accuracy on a [0, 1] scale."""

    ground_truth = [
        str(answer["answer"]).replace("\n", " ").replace("\t", " ").strip()
        for answer in answers
    ]
    normalized_prediction = str(prediction).replace(
        "\n", " "
    ).replace("\t", " ").strip()
    if len(set(ground_truth)) > 1:
        ground_truth = [
            normalize_vqa_answer(answer) for answer in ground_truth
        ]
        normalized_prediction = normalize_vqa_answer(normalized_prediction)
    scores = []
    for answer_index in range(len(ground_truth)):
        matches = sum(
            answer == normalized_prediction
            for index, answer in enumerate(ground_truth)
            if index != answer_index
        )
        scores.append(min(1.0, matches / 3.0))
    return mean(scores)


def normalized_vqa_soft_accuracy(
    prediction: str,
    answers: Sequence[Mapping[str, Any]],
) -> float:
    """Apply VQA normalization unconditionally before soft scoring.

    The pinned official evaluator skips normalization when all ten reference
    strings are identical. That makes an otherwise identical ``Yes``/``yes``
    response score differently. EXP-009 retains the exact official score for
    comparability, but uses this formatting-invariant score to label whether an
    intervention helped or harmed semantic task performance.
    """

    ground_truth = [
        normalize_vqa_answer(str(answer["answer"])) for answer in answers
    ]
    normalized_prediction = normalize_vqa_answer(prediction)
    scores = []
    for answer_index in range(len(ground_truth)):
        matches = sum(
            answer == normalized_prediction
            for index, answer in enumerate(ground_truth)
            if index != answer_index
        )
        scores.append(min(1.0, matches / 3.0))
    return mean(scores)


def is_generic_refusal(text: str) -> bool:
    return bool(_GENERIC_REFUSAL.search(str(text)))


def evaluate_vqav2_predictions(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evaluated = []
    for record in records:
        result = dict(record)
        result.update(
            {
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "vqa_soft_accuracy": vqa_soft_accuracy(
                    str(record["generated_text"]),
                    record["answers"],
                ),
                "normalized_vqa_soft_accuracy": normalized_vqa_soft_accuracy(
                    str(record["generated_text"]),
                    record["answers"],
                ),
                "generic_refusal_proxy": is_generic_refusal(
                    str(record["generated_text"])
                ),
            }
        )
        evaluated.append(result)
    return evaluated


def pair_vqav2_actions(
    records: Iterable[Mapping[str, Any]],
    *,
    effect_epsilon: float,
) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        question_id = int(record["question_id"])
        action = str(record["action"])
        if action in by_id.setdefault(question_id, {}):
            raise ValueError(f"Duplicate VQAv2 action: {(question_id, action)}")
        by_id[question_id][action] = record
    expected = {"native_prompt", "premise_verification"}
    pairs = []
    for question_id, actions in sorted(by_id.items()):
        if set(actions) != expected:
            raise ValueError(
                f"Incomplete VQAv2 actions for {question_id}: {sorted(actions)}"
            )
        native = actions["native_prompt"]
        intervention = actions["premise_verification"]
        native_official_score = float(native["vqa_soft_accuracy"])
        intervention_official_score = float(intervention["vqa_soft_accuracy"])
        native_score = float(native["normalized_vqa_soft_accuracy"])
        intervention_score = float(
            intervention["normalized_vqa_soft_accuracy"]
        )
        delta = intervention_score - native_score
        effect = (
            "helps"
            if delta > effect_epsilon
            else "harms"
            if delta < -effect_epsilon
            else "tie"
        )
        pairs.append(
            {
                "question_id": question_id,
                "image_id": int(native["image_id"]),
                "image_sha256": str(native["image_sha256"]),
                "answer_type": str(native["answer_type"]),
                "question_type": str(native["question_type"]),
                "question": str(native["question"]),
                "multiple_choice_answer": str(
                    native["multiple_choice_answer"]
                ),
                "native_response": str(native["generated_text"]),
                "intervention_response": str(
                    intervention["generated_text"]
                ),
                "native_vqa_soft_accuracy": native_score,
                "intervention_vqa_soft_accuracy": intervention_score,
                "intervention_minus_native": delta,
                "native_official_vqa_soft_accuracy": native_official_score,
                "intervention_official_vqa_soft_accuracy": (
                    intervention_official_score
                ),
                "official_intervention_minus_native": (
                    intervention_official_score - native_official_score
                ),
                "effect": effect,
                "native_generic_refusal_proxy": bool(
                    native["generic_refusal_proxy"]
                ),
                "intervention_generic_refusal_proxy": bool(
                    intervention["generic_refusal_proxy"]
                ),
                "new_generic_refusal_proxy": bool(
                    intervention["generic_refusal_proxy"]
                    and not native["generic_refusal_proxy"]
                ),
            }
        )
    return pairs


def _bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    estimates = np.asarray(
        [
            array[
                generator.integers(0, len(array), size=len(array))
            ].mean()
            for _ in range(resamples)
        ]
    )
    return {
        "point_estimate": float(array.mean()),
        "bootstrap_95_percent_interval": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "resamples": resamples,
    }


def summarize_vqav2_control(
    records: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    def action_summary(
        action: str, answer_type: str | None = None
    ) -> dict[str, Any]:
        selected = [
            record
            for record in records
            if record["action"] == action
            and (
                answer_type is None
                or record["answer_type"] == answer_type
            )
        ]
        return {
            "count": len(selected),
            "mean_official_vqa_soft_accuracy": mean(
                float(record["vqa_soft_accuracy"]) for record in selected
            ),
            "mean_normalized_vqa_soft_accuracy": mean(
                float(record["normalized_vqa_soft_accuracy"])
                for record in selected
            ),
            "generic_refusal_proxy_rate": mean(
                bool(record["generic_refusal_proxy"]) for record in selected
            ),
        }

    def pair_summary(
        selected_pairs: Sequence[Mapping[str, Any]],
        summary_seed: int,
    ) -> dict[str, Any]:
        deltas = [
            float(pair["intervention_minus_native"])
            for pair in selected_pairs
        ]
        return {
            "count": len(selected_pairs),
            "effect_counts": dict(
                sorted(
                    Counter(str(pair["effect"]) for pair in selected_pairs).items()
                )
            ),
            "intervention_minus_native": _bootstrap_mean(
                deltas,
                resamples=bootstrap_resamples,
                seed=summary_seed,
            ),
            "new_generic_refusal_proxy_count": sum(
                bool(pair["new_generic_refusal_proxy"])
                for pair in selected_pairs
            ),
        }

    answer_types = ("yes/no", "number", "other")
    return {
        "protocol": EVALUATION_PROTOCOL,
        "action_effect_metric": "normalized_vqa_soft_accuracy",
        "evidence_boundary": (
            "Exact pinned-official and unconditionally normalized VQAv2 soft "
            "accuracy on a deterministic, high-consensus 300-question "
            "validation subset. Action effects use the normalized score to "
            "remove unanimous-reference capitalization artifacts. This is "
            "not the full VQAv2 validation score."
        ),
        "question_count": len(pairs),
        "generation_count": len(records),
        "actions": {
            action: {
                "overall": action_summary(action),
                "by_answer_type": {
                    answer_type: action_summary(action, answer_type)
                    for answer_type in answer_types
                },
            }
            for action in ("native_prompt", "premise_verification")
        },
        "paired": pair_summary(pairs, seed),
        "by_answer_type": {
            answer_type: pair_summary(
                [
                    pair
                    for pair in pairs
                    if pair["answer_type"] == answer_type
                ],
                seed + index + 1,
            )
            for index, answer_type in enumerate(answer_types)
        },
    }
