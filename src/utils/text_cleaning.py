"""Text normalization helpers used before factual verification.

These helpers keep the displayed answer unchanged while removing citation markers
from the internal detection path. This prevents citation IDs such as
[Evidence-1] from being interpreted as factual numbers.
"""

from __future__ import annotations

import re

_EVIDENCE_CITATION_PATTERN = re.compile(
    r"\[\s*Evidence\s*[-:]?\s*\d+\s*\]|\bEvidence\s*[-:]?\s*\d+\s*\]"
    r"|\[\s*\d+\s*\]",
    re.IGNORECASE,
)

_NON_FACTUAL_PREFIXES = (
    "i couldn't find",
    "i could not find",
    "could you please",
    "please provide",
    "i'll do my best",
    "i will do my best",
    "i cannot verify",
    "i can't verify",
    "i don't have enough information",
    "i do not have enough information",
    "i am unable to",
    "i'm unable to",
    "i cannot determine",
    "i can't determine",
    "based on the available evidence, i cannot",
    "the knowledge base does not provide",
    "the retrieved evidence does not provide",
    "the retrieved evidence is insufficient",
    "insufficient evidence",
)

_NON_FACTUAL_CONTAINS = (
    "please provide more context",
    "clarify what",
    "do my best to assist",
    "does not provide evidence that",
    "does not provide evidence for",
    "not provide evidence that",
    "not provide evidence for",
)


def strip_evidence_citations(text: str) -> str:
    """Remove evidence citation markers from text used for detection.

    Examples removed:
    - [Evidence-1]
    - [Evidence: 2]
    - Evidence-3]
    - [1]
    """
    if not text:
        return ""
    cleaned = _EVIDENCE_CITATION_PATTERN.sub("", text)
    # Remove occasional broken citation fragments left by sentence splitting.
    cleaned = re.sub(r"\bEvidence\s*[-:]?\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("[]", "")
    return cleaned


def normalize_for_detection(text: str) -> str:
    """Normalize text before claim extraction/scoring/verifying."""
    cleaned = strip_evidence_citations(text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    return cleaned.strip()


def is_non_factual_assistant_phrase(sentence: str) -> bool:
    """Return True for meta/refusal/helpfulness phrases, not factual claims."""
    if not sentence:
        return True
    lower = normalize_for_detection(sentence).lower().strip()
    if not lower:
        return True
    if any(lower.startswith(prefix) for prefix in _NON_FACTUAL_PREFIXES):
        return True
    if any(fragment in lower for fragment in _NON_FACTUAL_CONTAINS):
        return True
    return False
