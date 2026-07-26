"""Conservative local diagnostics for HaloQuest false-premise responses.

These labels implement the central public HaloQuest false-premise requirement:
the response must explicitly acknowledge that the questioned element is absent.
They are deterministic diagnostics, not a replacement for HaloQuest Auto-Eval.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from exp001_evaluation import classify_conflict_response, lexical_metrics


PROTOCOL_VERSION = "exp005-high-confidence-false-premise-eval-v1"
_COUNT_QUESTION = re.compile(
    r"\b(?:how\s+many|number\s+of|count)\b",
    re.IGNORECASE,
)
_ZERO_REFERENCE = re.compile(
    r"\b(?:0|zero|no|none|not\s+any|without|"
    r"does?\s+not\s+have|do\s+not\s+have)\b",
    re.IGNORECASE,
)
_EXTRA_ABSENCE = re.compile(
    r"\b(?:does?\s+not\s+exist|do\s+not\s+exist|"
    r"has\s+no|have\s+no|is\s+absent|are\s+absent)\b",
    re.IGNORECASE,
)
_TRUSTED_REJECTION_RULES = {
    "there_is_no",
    "visual_lacks",
    "no_item_visible",
    "target_local_negation",
}


def evaluate_haloquest_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a conservative false-premise verdict to one generation."""

    required = {
        "haloquest_id",
        "action",
        "question",
        "groundtruth_response",
        "generated_text",
        "image_type",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"HaloQuest prediction is missing fields: {sorted(missing)}")
    reference = str(record["groundtruth_response"]).strip()
    prediction = str(record["generated_text"]).strip()
    lexical = lexical_metrics(reference, prediction)
    stance = classify_conflict_response(reference, prediction)
    normalized_prediction = " ".join(prediction.lower().split()).rstrip(".")
    zero_count = bool(
        normalized_prediction in {"0", "zero"}
        and _COUNT_QUESTION.search(str(record["question"]))
        and _ZERO_REFERENCE.search(reference)
    )
    direct_visual_rejection = bool(
        stance["stance_rule"] == "visual_does_not_contain"
        and re.search(
            r"\b(?:contain|include|show|depict|feature|display|have|capture)\b",
            str(stance["stance_evidence"]),
            re.IGNORECASE,
        )
    )
    extra_target_linked_absence = bool(
        stance["reference_content_overlap"]
        and _EXTRA_ABSENCE.search(prediction)
    )

    if (
        lexical["normalized_exact_match"]
        or lexical["reference_contained"]
        or zero_count
    ):
        verdict = "correct"
        verdict_reason = "reference_lexically_supported"
    elif stance["stance"] == "reject_premise" and (
        stance["stance_rule"] in _TRUSTED_REJECTION_RULES
        or direct_visual_rejection
    ):
        verdict = "correct"
        verdict_reason = f"target_linked_rejection:{stance['stance_rule']}"
    elif extra_target_linked_absence:
        verdict = "correct"
        verdict_reason = "target_linked_explicit_absence"
    elif stance["stance"] == "no_explicit_rejection":
        verdict = "incorrect"
        verdict_reason = "no_explicit_false_premise_rejection"
    else:
        verdict = "uncertain"
        verdict_reason = f"non_decisive_stance:{stance['stance']}"

    result = dict(record)
    result.update(lexical)
    result.update(stance)
    result.update(
        {
            "local_evaluation_protocol": PROTOCOL_VERSION,
            "local_verdict": verdict,
            "local_verdict_reason": verdict_reason,
            "local_utility": (
                1 if verdict == "correct" else 0 if verdict == "incorrect" else None
            ),
        }
    )
    return result


def pair_action_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair native/intervention verdicts and expose action transitions."""

    by_id: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        haloquest_id = int(record["haloquest_id"])
        action = str(record["action"])
        if action in by_id.setdefault(haloquest_id, {}):
            raise ValueError(f"Duplicate action record: {(haloquest_id, action)}")
        by_id[haloquest_id][action] = record

    paired = []
    expected = {"native_prompt", "premise_verification"}
    for haloquest_id, actions in sorted(by_id.items()):
        if set(actions) != expected:
            raise ValueError(
                f"Incomplete actions for HaloQuest ID {haloquest_id}: "
                f"{sorted(actions)}"
            )
        native = actions["native_prompt"]
        intervention = actions["premise_verification"]
        native_utility = native["local_utility"]
        intervention_utility = intervention["local_utility"]
        if native_utility is None or intervention_utility is None:
            effect = "uncertain"
        elif intervention_utility > native_utility:
            effect = "helps"
        elif intervention_utility < native_utility:
            effect = "harms"
        else:
            effect = "tie"
        paired.append(
            {
                "haloquest_id": haloquest_id,
                "image_type": str(native["image_type"]),
                "question": str(native["question"]),
                "groundtruth_response": str(native["groundtruth_response"]),
                "native_response": str(native["generated_text"]),
                "intervention_response": str(intervention["generated_text"]),
                "native_verdict": str(native["local_verdict"]),
                "intervention_verdict": str(intervention["local_verdict"]),
                "native_utility": native_utility,
                "intervention_utility": intervention_utility,
                "effect": effect,
            }
        )
    return paired


def _action_summary(
    records: list[Mapping[str, Any]],
    action: str,
) -> dict[str, Any]:
    selected = [record for record in records if record["action"] == action]
    verdicts = Counter(str(record["local_verdict"]) for record in selected)
    decisive = verdicts["correct"] + verdicts["incorrect"]
    return {
        "count": len(selected),
        "verdict_counts": dict(sorted(verdicts.items())),
        "decisive_coverage": decisive / len(selected) if selected else 0.0,
        "local_proxy_success_rate_all": (
            verdicts["correct"] / len(selected) if selected else 0.0
        ),
        "local_proxy_success_rate_decisive": (
            verdicts["correct"] / decisive if decisive else None
        ),
    }


def summarize_haloquest_evaluation(
    records: Iterable[Mapping[str, Any]],
    pairs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize paired local diagnostics without calling them Auto-Eval."""

    records = list(records)
    pairs = list(pairs)
    effect_counts = Counter(str(pair["effect"]) for pair in pairs)
    decisive_pairs = [pair for pair in pairs if pair["effect"] != "uncertain"]
    native_success = sum(int(pair["native_utility"]) for pair in decisive_pairs)
    intervention_success = sum(
        int(pair["intervention_utility"]) for pair in decisive_pairs
    )
    helps = effect_counts["helps"]
    harms = effect_counts["harms"]
    return {
        "protocol": PROTOCOL_VERSION,
        "official_haloquest_autoeval_run": False,
        "evidence_boundary": (
            "Deterministic target-linked rejection proxy. It is not HaloQuest "
            "Gemini Auto-Eval and is not reported as semantic accuracy."
        ),
        "question_count": len(pairs),
        "generation_count": len(records),
        "actions": {
            action: _action_summary(records, action)
            for action in ("native_prompt", "premise_verification")
        },
        "paired": {
            "effect_counts": dict(sorted(effect_counts.items())),
            "decisive_pair_count": len(decisive_pairs),
            "decisive_pair_coverage": (
                len(decisive_pairs) / len(pairs) if pairs else 0.0
            ),
            "native_proxy_utility": (
                native_success / len(decisive_pairs) if decisive_pairs else None
            ),
            "intervention_proxy_utility": (
                intervention_success / len(decisive_pairs)
                if decisive_pairs
                else None
            ),
            "intervention_minus_native": (
                (intervention_success - native_success) / len(decisive_pairs)
                if decisive_pairs
                else None
            ),
            "discordant_pair_count": helps + harms,
            "helps_to_harms_ratio": helps / harms if harms else None,
        },
        "by_image_type": {
            image_type: {
                "count": sum(pair["image_type"] == image_type for pair in pairs),
                "effect_counts": dict(
                    sorted(
                        Counter(
                            str(pair["effect"])
                            for pair in pairs
                            if pair["image_type"] == image_type
                        ).items()
                    )
                ),
            }
            for image_type in ("generated", "real")
        },
    }
