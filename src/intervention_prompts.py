"""Prompt interventions used by EXP-003."""

from __future__ import annotations


INTERVENTION_PROTOCOL_VERSION = "exp003-premise-verification-v1"
PREMISE_VERIFICATION_INSTRUCTION = (
    "Check the image before answering. If the question assumes an object, "
    "attribute, or relation that the image does not support, explicitly say "
    "so and do not guess. Otherwise answer directly."
)


def apply_prompt_intervention(question: str, action: str) -> str:
    """Return the model-facing question for one intervention action."""

    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")
    if action == "native_prompt":
        return question
    if action == "premise_verification":
        return f"{PREMISE_VERIFICATION_INSTRUCTION}\n\nQuestion: {question}"
    raise ValueError(f"Unknown intervention action: {action}")
