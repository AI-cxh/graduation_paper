"""Auditable local metrics for EXP-001 free-generation outputs.

The official MMMC evaluation combines ROUGE with external LLM judges.  This
module intentionally implements only deterministic, locally reproducible
signals.  Rule-based stance labels and lexical scores are diagnostics rather
than substitutes for semantic human or LLM judgement.
"""

from __future__ import annotations

import re
import unicodedata
from math import comb
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


PROTOCOL_VERSION = "exp001-local-eval-v1"

_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
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
_CONTENT_STOPWORDS = {
    "a",
    "about",
    "an",
    "any",
    "are",
    "at",
    "be",
    "been",
    "being",
    "can",
    "cannot",
    "capture",
    "contain",
    "contains",
    "depict",
    "depicts",
    "did",
    "do",
    "does",
    "doesn",
    "don",
    "feature",
    "features",
    "for",
    "found",
    "from",
    "has",
    "have",
    "image",
    "information",
    "how",
    "in",
    "include",
    "includes",
    "indicate",
    "indicates",
    "is",
    "it",
    "many",
    "not",
    "of",
    "on",
    "or",
    "photo",
    "photograph",
    "picture",
    "present",
    "provide",
    "provides",
    "scene",
    "see",
    "seen",
    "show",
    "shown",
    "shows",
    "specific",
    "t",
    "that",
    "this",
    "to",
    "visible",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}

_UNCERTAINTY_PATTERN = re.compile(
    r"\b(?:not\s+sure|uncertain|unclear|perhaps|maybe|possibly|"
    r"likely|could\s+be|might\s+be|appears?\s+to\s+be|"
    r"seems?\s+to\s+be)\b",
    re.IGNORECASE,
)
_LOCAL_NEGATION_EXCLUSIONS = {
    "certain",
    "clear",
    "enough",
    "know",
    "possible",
    "sure",
    "understand",
}

_ABSTAIN_RULES = (
    (
        "no_visual_input",
        re.compile(
            r"\b(?:i|we)\s+(?:do not|don['’]t|cannot|can['’]t)\s+"
            r"(?:have(?:\s+access\s+to)?|see|view)\s+"
            r"(?:the|an|any)?\s*(?:image|picture|photo)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "missing_context",
        re.compile(
            r"\b(?:no|not enough|insufficient)\s+"
            r"(?:visual\s+)?(?:context|information)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "without_visual_input",
        re.compile(
            r"\bwithout\s+(?:seeing|viewing|access\s+to)\s+"
            r"(?:the|an)\s+(?:image|picture|photo)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "request_visual_input",
        re.compile(
            r"\bplease\s+(?:provide|upload|share)\b.{0,80}"
            r"\b(?:image|picture|photo|context)\b",
            re.IGNORECASE,
        ),
    ),
)

_REJECTION_RULES = (
    (
        "visual_does_not_contain",
        re.compile(
            r"\b(?:the|this)\s+(?:image|picture|photo|photograph|scene)\s+"
            r"(?:does\s+not|doesn['’]t|do\s+not|don['’]t)\s+"
            r"(?:contain|include|show|depict|feature|display|have|capture|"
            r"provide|indicate)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "there_is_no",
        re.compile(r"\bthere\s+(?:is|are|was|were)\s+no\b", re.IGNORECASE),
    ),
    (
        "not_visually_present",
        re.compile(
            r"\b(?:is|are|was|were)\s+not\s+"
            r"(?:visible|shown|present|depicted|seen|found)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cannot_visually_find",
        re.compile(
            r"\b(?:cannot|can['’]t|could\s+not|couldn['’]t|"
            r"do\s+not|don['’]t)\s+"
            r"(?:see|find|locate|identify|detect|observe)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "visual_lacks",
        re.compile(
            r"\b(?:image|picture|photo|photograph|scene)\s+"
            r"(?:lacks?|is\s+without)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "no_item_visible",
        re.compile(
            r"\bno\b.{0,100}\b(?:visible|shown|present|depicted|seen|found)\b",
            re.IGNORECASE,
        ),
    ),
)


def normalize_text(text: str) -> str:
    """Normalize text for transparent lexical comparison.

    This is deliberately small and deterministic: Unicode normalization,
    leading-list-marker removal, lowercase, punctuation removal, article
    removal, and number-word mapping.
    """

    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text)
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = text.replace("'", " ")
    tokens = []
    for token in text.split():
        if token in _ARTICLES:
            continue
        tokens.append(_NUMBER_WORDS.get(token, token))
    return " ".join(tokens)


def _tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def lexical_metrics(reference: str, prediction: str) -> dict[str, float | bool]:
    """Return normalized EM, token overlap, containment, and ROUGE-L F1."""

    reference_tokens = _tokens(reference)
    prediction_tokens = _tokens(prediction)
    exact_match = bool(reference_tokens) and reference_tokens == prediction_tokens

    overlap = sum(
        (Counter(reference_tokens) & Counter(prediction_tokens)).values()
    )
    precision = overlap / len(prediction_tokens) if prediction_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    token_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    lcs = _lcs_length(reference_tokens, prediction_tokens)
    rouge_precision = lcs / len(prediction_tokens) if prediction_tokens else 0.0
    rouge_recall = lcs / len(reference_tokens) if reference_tokens else 0.0
    rouge_l_f1 = (
        2 * rouge_precision * rouge_recall / (rouge_precision + rouge_recall)
        if rouge_precision + rouge_recall
        else 0.0
    )

    reference_contained = False
    if reference_tokens and len(reference_tokens) <= len(prediction_tokens):
        width = len(reference_tokens)
        reference_contained = any(
            prediction_tokens[start : start + width] == reference_tokens
            for start in range(len(prediction_tokens) - width + 1)
        )

    return {
        "normalized_exact_match": exact_match,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": token_f1,
        "rouge_l_f1": rouge_l_f1,
        "reference_contained": reference_contained,
    }


def _stem_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def reference_content_tokens(reference: str) -> set[str]:
    """Extract target-bearing reference tokens after removing MMMC boilerplate."""

    return {
        _stem_token(token)
        for token in _tokens(reference)
        if token not in _CONTENT_STOPWORDS and len(token) > 1
    }


def classify_conflict_response(
    reference: str,
    prediction: str,
) -> dict[str, str | list[str]]:
    """Classify whether a response explicitly rejects the false premise.

    A rejection rule is high-confidence only when the generated response also
    overlaps target-bearing content from the reference.  Generic inability to
    view an image is kept separate from premise rejection.
    """

    prediction = str(prediction).strip()
    if not prediction:
        return {
            "stance": "empty",
            "stance_rule": "empty_response",
            "stance_evidence": "",
            "reference_content_overlap": [],
        }

    for rule_name, pattern in _ABSTAIN_RULES:
        match = pattern.search(prediction)
        if match:
            return {
                "stance": "abstain_no_visual_input",
                "stance_rule": rule_name,
                "stance_evidence": match.group(0),
                "reference_content_overlap": [],
            }

    reference_tokens = reference_content_tokens(reference)
    prediction_tokens = {_stem_token(token) for token in _tokens(prediction)}
    content_overlap = sorted(reference_tokens & prediction_tokens)

    for rule_name, pattern in _REJECTION_RULES:
        match = pattern.search(prediction)
        if match:
            stance = (
                "reject_premise" if content_overlap else "ambiguous_rejection"
            )
            return {
                "stance": stance,
                "stance_rule": rule_name,
                "stance_evidence": match.group(0),
                "reference_content_overlap": content_overlap,
            }

    stemmed_prediction = [_stem_token(token) for token in _tokens(prediction)]
    target_positions = {
        index
        for index, token in enumerate(stemmed_prediction)
        if token in reference_tokens
    }
    for index, token in enumerate(stemmed_prediction):
        if token != "not":
            continue
        following = (
            stemmed_prediction[index + 1]
            if index + 1 < len(stemmed_prediction)
            else ""
        )
        if following in _LOCAL_NEGATION_EXCLUSIONS:
            continue
        if any(abs(index - target_index) <= 8 for target_index in target_positions):
            start = max(0, index - 5)
            end = min(len(stemmed_prediction), index + 6)
            return {
                "stance": "reject_premise",
                "stance_rule": "target_local_negation",
                "stance_evidence": " ".join(stemmed_prediction[start:end]),
                "reference_content_overlap": content_overlap,
            }

    return {
        "stance": "no_explicit_rejection",
        "stance_rule": "no_rule_match",
        "stance_evidence": "",
        "reference_content_overlap": content_overlap,
    }


def evaluate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Attach local evaluation fields to one prediction record."""

    required = {
        "image_id",
        "condition",
        "conflict_type",
        "question",
        "reference_answer",
        "generated_text",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Prediction record is missing fields: {sorted(missing)}")

    reference = str(record["reference_answer"]).splitlines()[0].strip()
    prediction = str(record["generated_text"]).strip()
    result = dict(record)
    result.update(
        {
            "evaluation_protocol": PROTOCOL_VERSION,
            "reference_first_line": reference,
            "normalized_reference": normalize_text(reference),
            "normalized_prediction": normalize_text(prediction),
            **lexical_metrics(reference, prediction),
        }
    )

    condition = str(record["condition"])
    if condition in {"multimodal_conflict", "text_only_conflict"}:
        stance = classify_conflict_response(reference, prediction)
        result.update(stance)
        result["manual_review_required"] = stance["stance"] != "reject_premise"
        result["manual_review_reason"] = (
            ""
            if stance["stance"] == "reject_premise"
            else f"conflict_stance:{stance['stance']}"
        )
    else:
        result.update(
            {
                "stance": "not_applicable",
                "stance_rule": "not_applicable",
                "stance_evidence": "",
                "reference_content_overlap": [],
            }
        )
        prediction_uncertain = bool(_UNCERTAINTY_PATTERN.search(prediction))
        result["prediction_uncertainty_flag"] = prediction_uncertain
        lexical_support = bool(
            result["normalized_exact_match"]
            or (result["reference_contained"] and not prediction_uncertain)
        )
        result["manual_review_required"] = not lexical_support
        if lexical_support:
            result["manual_review_reason"] = ""
        elif prediction_uncertain and result["reference_contained"]:
            result["manual_review_reason"] = "clean_containment_is_uncertain"
        else:
            result["manual_review_reason"] = "clean_answer_requires_semantic_review"
    return result


def _mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(float(record[key]) for record in records) / len(records)


def summarize_group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one non-empty group without claiming semantic accuracy."""

    if not records:
        raise ValueError("Cannot summarize an empty group")
    stance_counts = Counter(str(record["stance"]) for record in records)
    count = len(records)
    return {
        "count": count,
        "normalized_exact_match_rate": _mean(
            records, "normalized_exact_match"
        ),
        "reference_contained_rate": _mean(records, "reference_contained"),
        "mean_token_precision": _mean(records, "token_precision"),
        "mean_token_recall": _mean(records, "token_recall"),
        "mean_token_f1": _mean(records, "token_f1"),
        "mean_rouge_l_f1": _mean(records, "rouge_l_f1"),
        "stance_counts": dict(sorted(stance_counts.items())),
        "explicit_target_rejection_rate": stance_counts["reject_premise"] / count,
        "abstain_no_visual_input_rate": (
            stance_counts["abstain_no_visual_input"] / count
        ),
        "manual_review_count": sum(
            bool(record["manual_review_required"]) for record in records
        ),
        "manual_review_rate": _mean(records, "manual_review_required"),
    }


def _exact_mcnemar_p_value(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(
        comb(discordant, index) for index in range(min(first_only, second_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def _paired_transition_summary(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    counts = Counter()
    for multimodal, text_only in pairs:
        multimodal_rejects = multimodal["stance"] == "reject_premise"
        text_only_rejects = text_only["stance"] == "reject_premise"
        if multimodal_rejects and text_only_rejects:
            counts["both_reject"] += 1
        elif multimodal_rejects:
            counts["multimodal_only_reject"] += 1
        elif text_only_rejects:
            counts["text_only_only_reject"] += 1
        else:
            counts["neither_reject"] += 1

    count = len(pairs)
    multimodal_rejections = (
        counts["both_reject"] + counts["multimodal_only_reject"]
    )
    text_only_rejections = (
        counts["both_reject"] + counts["text_only_only_reject"]
    )
    return {
        "pair_count": count,
        "transition_counts": {
            key: counts[key]
            for key in (
                "both_reject",
                "multimodal_only_reject",
                "text_only_only_reject",
                "neither_reject",
            )
        },
        "multimodal_explicit_target_rejection_rate": (
            multimodal_rejections / count
        ),
        "text_only_explicit_target_rejection_rate": (
            text_only_rejections / count
        ),
        "multimodal_minus_text_only_rejection_rate": (
            multimodal_rejections - text_only_rejections
        )
        / count,
        "exact_mcnemar_p_value": _exact_mcnemar_p_value(
            counts["multimodal_only_reject"],
            counts["text_only_only_reject"],
        ),
    }


def summarize_paired_conflict(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize paired stance changes after adding the image."""

    multimodal = {
        int(record["image_id"]): record
        for record in records
        if record["condition"] == "multimodal_conflict"
    }
    text_only = {
        int(record["image_id"]): record
        for record in records
        if record["condition"] == "text_only_conflict"
    }
    if not multimodal or not text_only:
        return {
            "available": False,
            "reason": "Both conflict conditions are required for paired analysis.",
        }
    if multimodal.keys() != text_only.keys():
        missing_multimodal = sorted(text_only.keys() - multimodal.keys())
        missing_text_only = sorted(multimodal.keys() - text_only.keys())
        raise ValueError(
            "Conflict conditions are not pair-aligned: "
            f"missing_multimodal={missing_multimodal[:10]}, "
            f"missing_text_only={missing_text_only[:10]}"
        )

    pairs = [(multimodal[key], text_only[key]) for key in sorted(multimodal)]
    for multimodal_record, text_only_record in pairs:
        if multimodal_record["conflict_type"] != text_only_record["conflict_type"]:
            raise ValueError(
                f"Conflict type mismatch for image_id={multimodal_record['image_id']}"
            )

    conflict_types = sorted(
        {str(multimodal_record["conflict_type"]) for multimodal_record, _ in pairs}
    )
    return {
        "available": True,
        "scope": (
            "Paired changes in the explicit target-linked rejection proxy, "
            "not paired semantic accuracy."
        ),
        **_paired_transition_summary(pairs),
        "by_conflict_type": {
            conflict_type: _paired_transition_summary(
                [
                    pair
                    for pair in pairs
                    if str(pair[0]["conflict_type"]) == conflict_type
                ]
            )
            for conflict_type in conflict_types
        },
    }


def summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Produce overall, condition, and condition-by-conflict-type summaries."""

    records = list(records)
    if not records:
        raise ValueError("No evaluated records supplied")

    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    by_condition_type: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        condition = str(record["condition"])
        conflict_type = str(record["conflict_type"])
        by_condition.setdefault(condition, []).append(record)
        by_condition_type.setdefault((condition, conflict_type), []).append(record)

    return {
        "evaluation_protocol": PROTOCOL_VERSION,
        "metric_scope": (
            "Deterministic lexical diagnostics and explicit target-linked "
            "premise-rejection rules; not semantic accuracy."
        ),
        "overall": summarize_group(records),
        "by_condition": {
            condition: summarize_group(group)
            for condition, group in sorted(by_condition.items())
        },
        "by_condition_and_conflict_type": {
            condition: {
                conflict_type: summarize_group(
                    by_condition_type[(condition, conflict_type)]
                )
                for current_condition, conflict_type in sorted(by_condition_type)
                if current_condition == condition
            }
            for condition in sorted(by_condition)
        },
        "paired_conflict_stance": summarize_paired_conflict(records),
    }
