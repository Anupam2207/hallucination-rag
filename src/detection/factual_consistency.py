"""Lightweight factual consistency checks for claim-evidence verification.

These checks complement embedding similarity. They catch cases where two
sentences are semantically close but disagree on factual values, for example:
claim:    "RAG was introduced in 2021."
evidence: "RAG was introduced in 2020."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
    re.IGNORECASE,
)
_ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[A-Z]{2,})\b")

_STOP_ENTITIES = {
    "The", "A", "An", "This", "That", "It", "These", "Those", "In", "On",
    "For", "And", "Or", "But", "Evidence", "Source", "Claim", "RAG", "LLM",
    "NLP", "BM25", "RRF", "PDF", "JSON", "ChromaDB", "Formula", "One",
}

_RELATION_PATTERNS = [
    "introduced", "invented", "founded", "created", "released", "began",
    "started", "won", "built", "developed", "launched", "established",
]


@dataclass
class FactualCheckResult:
    flags: List[str]
    details: Dict[str, List[str]]


def extract_years(text: str) -> List[str]:
    return sorted(set(_YEAR_PATTERN.findall(text or "")))


def extract_numbers(text: str) -> List[str]:
    numbers = set(_NUMBER_PATTERN.findall(text or ""))
    # Years are handled separately and should not trigger duplicate number flags.
    return sorted(num for num in numbers if not _YEAR_PATTERN.fullmatch(num))


def extract_dates(text: str) -> List[str]:
    return sorted(set(match.group(0) for match in _DATE_PATTERN.finditer(text or "")))


def extract_capitalized_entities(text: str) -> List[str]:
    entities: List[str] = []
    for match in _ENTITY_PATTERN.finditer(text or ""):
        entity = match.group(0).strip()
        if entity in _STOP_ENTITIES:
            continue
        if len(entity) <= 1:
            continue
        entities.append(entity)
    # Preserve order while deduplicating.
    return list(dict.fromkeys(entities))


def _shared_context(claim: str, evidence: str) -> bool:
    claim_l = (claim or "").lower()
    evidence_l = (evidence or "").lower()
    if any(pattern in claim_l and pattern in evidence_l for pattern in _RELATION_PATTERNS):
        return True
    claim_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z]{2,}", claim_l))
    evidence_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z]{2,}", evidence_l))
    stop = {"the", "and", "that", "this", "with", "from", "using", "used", "was", "were", "are", "for"}
    claim_tokens -= stop
    evidence_tokens -= stop
    return len(claim_tokens & evidence_tokens) >= 3


def detect_numeric_mismatch(claim: str, evidence: str) -> List[str]:
    """Return numeric mismatch flags conservatively.

    If the claim has years/numbers and evidence has conflicting years/numbers in
    a related context, flag it. If evidence has no comparable value, do not flag.
    """
    flags: List[str] = []
    if not _shared_context(claim, evidence):
        return flags

    claim_years = set(extract_years(claim))
    evidence_years = set(extract_years(evidence))
    if claim_years and evidence_years and not claim_years.issubset(evidence_years):
        flags.append("numeric_mismatch_with_evidence")

    claim_numbers = set(extract_numbers(claim))
    evidence_numbers = set(extract_numbers(evidence))
    if claim_numbers and evidence_numbers and not claim_numbers.issubset(evidence_numbers):
        if "numeric_mismatch_with_evidence" not in flags:
            flags.append("numeric_mismatch_with_evidence")

    return flags


def detect_entity_mismatch(claim: str, evidence: str) -> List[str]:
    """Return entity mismatch flags conservatively.

    This intentionally catches only simple relation/entity conflicts. NLI remains
    the stronger layer for complex entity contradictions.
    """
    flags: List[str] = []
    if not _shared_context(claim, evidence):
        return flags

    claim_entities = set(extract_capitalized_entities(claim))
    evidence_entities = set(extract_capitalized_entities(evidence))
    if not claim_entities or not evidence_entities:
        return flags

    # Do not count if all claim entities are present in the evidence.
    if claim_entities.issubset(evidence_entities):
        return flags

    relation_overlap = any(
        relation in claim.lower() and relation in evidence.lower()
        for relation in _RELATION_PATTERNS
    )
    if relation_overlap and (claim_entities - evidence_entities) and (evidence_entities - claim_entities):
        flags.append("entity_mismatch_with_evidence")

    return flags


def run_factual_consistency_checks(claim: str, evidence: str) -> Dict[str, object]:
    flags: List[str] = []
    flags.extend(detect_numeric_mismatch(claim, evidence))
    flags.extend(detect_entity_mismatch(claim, evidence))
    flags = list(dict.fromkeys(flags))
    return {
        "flags": flags,
        "details": {
            "claim_years": extract_years(claim),
            "evidence_years": extract_years(evidence),
            "claim_numbers": extract_numbers(claim),
            "evidence_numbers": extract_numbers(evidence),
            "claim_entities": extract_capitalized_entities(claim),
            "evidence_entities": extract_capitalized_entities(evidence),
        },
    }
