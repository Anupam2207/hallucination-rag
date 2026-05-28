"""Text normalization helpers for detection and display.

The UI may display citations such as [Evidence-1], but the detection path must
not treat citation IDs as factual numbers. These helpers normalize citations for
answers shown to the user and strip them before claim extraction, support
scoring, factual consistency checks, NLI, and metrics.
"""

from __future__ import annotations

import re

# Matches [Evidence-1], [Evidence-1, Evidence-2], [Evidence-1,Evidence-2],
# [1], [1, 2], and common malformed fragments such as Evidence-1].
_EVIDENCE_CITATION_PATTERN = re.compile(
    r"\[\s*(?:Evidence\s*[-:]?\s*\d+\s*(?:,\s*Evidence\s*[-:]?\s*\d+\s*)*)\]"
    r"|\[\s*\d+\s*(?:,\s*\d+\s*)*\]"
    r"|\bEvidence\s*[-:]?\s*\d+\s*\]",
    re.IGNORECASE,
)

_INNER_EVIDENCE_ID_PATTERN = re.compile(r"Evidence\s*[-:]?\s*\d+", re.IGNORECASE)

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


def normalize_evidence_citations(text: str) -> str:
    """Normalize citation formatting for display.

    The preferred displayed citation style is [Evidence-N]. Detection code should
    call strip_evidence_citations instead because citations are display-only.
    """
    if not text:
        return ""

    def _normalize_bracket(match: re.Match[str]) -> str:
        content = match.group(1)
        if not re.search(r"Evidence\s*[-:]?\s*\d+|\b\d+\b", content, flags=re.IGNORECASE):
            return match.group(0)
        values: list[str] = []
        for number in re.findall(r"(?:Evidence\s*[-:]?\s*)?(\d+)", content, flags=re.IGNORECASE):
            if number not in values:
                values.append(number)
        if not values:
            return match.group(0)
        if len(values) == 1:
            return f"[Evidence-{values[0]}]"
        return "[" + ", ".join(f"Evidence-{value}" for value in values) + "]"

    normalized = re.sub(r"\[([^\]]+)\]", _normalize_bracket, text)
    normalized = re.sub(r"\s+([.,;:!?])", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def strip_evidence_citations(text: str) -> str:
    """Remove evidence citation markers from text used for detection.

    Examples removed:
    - [Evidence-1]
    - [Evidence-1, Evidence-2]
    - [Evidence-1,Evidence-2]
    - [1, 2]
    - Evidence-3]
    """
    if not text:
        return ""
    cleaned = _EVIDENCE_CITATION_PATTERN.sub("", text)
    cleaned = _INNER_EVIDENCE_ID_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_for_detection(text: str) -> str:
    """Normalize text before claim extraction/scoring/verifying."""
    cleaned = strip_evidence_citations(text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    # Avoid malformed numbered-list fragments becoming claims.
    if "\n" in cleaned or re.search(r"\s\d+[\).]\s", cleaned):
        cleaned = clean_malformed_lists(cleaned)
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


def clean_malformed_lists(text: str) -> str:
    """Normalize malformed numbered/bulleted lists produced by small LLMs.

    The goal is not to preserve perfect list formatting; it is to prevent
    broken fragments such as ``1. ... 2. 3. ...`` from polluting claim
    extraction and the final user-facing answer.
    """
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\*\*", "", cleaned)
    # Put numbered items on separate lines when they appear inline.
    cleaned = re.sub(r"\s+(\d+)[\).]\s+(?=[A-Z])", r"\n\1. ", cleaned)
    # Remove empty/dangling numeric markers such as `2. 3.` or final `1.`.
    cleaned = re.sub(r"(?:^|\n)\s*\d+[\).]\s*(?=\n|$)", "\n", cleaned)
    cleaned = re.sub(r"\s+\d{1,2}[\).]\s*(?=\d{1,2}[\).]|$)", " ", cleaned)
    # Remove remaining inline list markers, e.g. "item one 3. item two".
    cleaned = re.sub(r"\s+\d+[\).]\s+(?=[A-Z])", " ", cleaned)
    cleaned = re.sub(r"(can|include|including|as follows):\s*\d+[\).]?\s*$", r"\1:", cleaned, flags=re.IGNORECASE)
    # Collapse repeated whitespace but preserve paragraph/list line breaks.
    lines = [re.sub(r"\s+", " ", line).strip() for line in cleaned.split("\n")]
    lines = [line for line in lines if line]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned.strip()


def remove_display_citations(text: str) -> str:
    """Return a clean user-facing answer with evidence citations removed."""
    cleaned = strip_evidence_citations(text or "")
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return clean_malformed_lists(cleaned)
