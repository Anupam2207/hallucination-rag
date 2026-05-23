"""Lightweight factual consistency checks for claim-evidence verification.

These checks complement embedding similarity and optional NLI. They catch cases
where two sentences are semantically close but disagree on factual values or
where a central numeric/year claim is not supported by the retrieved evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from src.utils.text_cleaning import normalize_for_detection

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
    "Retrieval", "Generation", "Augmented", "Could",
}

_RELATION_PATTERNS = [
    "introduced", "invented", "founded", "created", "released", "began",
    "started", "won", "built", "developed", "launched", "established",
    "proposed", "published", "authored", "written",
]

_CENTRAL_NUMERIC_PATTERNS = [
    "introduced", "invented", "created", "proposed", "released", "published",
    "launched", "developed", "won", "began", "started", "founded", "built",
    "established", "which year", "when", "year", "date",
]

_GENERIC_COUNT_CONTEXTS = (
    "two main stages", "two stages", "two main components", "two components",
    "three stages", "several benefits", "top 5", "top five",
)


@dataclass
class FactualCheckResult:
    flags: List[str]
    details: Dict[str, List[str]]


def _clean(text: str) -> str:
    return normalize_for_detection(text or "")


def extract_years(text: str) -> List[str]:
    return sorted(set(_YEAR_PATTERN.findall(_clean(text))))


def extract_numbers(text: str) -> List[str]:
    cleaned = _clean(text)
    numbers = set(_NUMBER_PATTERN.findall(cleaned))
    # Years are handled separately and should not trigger duplicate number flags.
    return sorted(num for num in numbers if not _YEAR_PATTERN.fullmatch(num))


def extract_dates(text: str) -> List[str]:
    return sorted(set(match.group(0) for match in _DATE_PATTERN.finditer(_clean(text))))


def extract_capitalized_entities(text: str) -> List[str]:
    cleaned = _clean(text)
    entities: List[str] = []
    for match in _ENTITY_PATTERN.finditer(cleaned):
        entity = match.group(0).strip()
        if entity in _STOP_ENTITIES:
            continue
        if len(entity) <= 1:
            continue
        entities.append(entity)
    return list(dict.fromkeys(entities))


def _shared_context(claim: str, evidence: str) -> bool:
    claim_l = _clean(claim).lower()
    evidence_l = _clean(evidence).lower()
    if any(pattern in claim_l and pattern in evidence_l for pattern in _RELATION_PATTERNS):
        return True
    claim_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z]{2,}", claim_l))
    evidence_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z]{2,}", evidence_l))
    stop = {
        "the", "and", "that", "this", "with", "from", "using", "used", "was",
        "were", "are", "for", "into", "about", "does", "not", "provide", "evidence",
    }
    claim_tokens -= stop
    evidence_tokens -= stop
    return len(claim_tokens & evidence_tokens) >= 3


def _numeric_claim_is_central(claim: str) -> bool:
    lower = _clean(claim).lower()
    if any(fragment in lower for fragment in _GENERIC_COUNT_CONTEXTS):
        return False
    return any(pattern in lower for pattern in _CENTRAL_NUMERIC_PATTERNS)


def detect_numeric_mismatch(claim: str, evidence: str) -> List[str]:
    """Return numeric/year/date factual support flags conservatively."""
    flags: List[str] = []
    cleaned_claim = _clean(claim)
    cleaned_evidence = _clean(evidence)
    if not cleaned_claim:
        return flags

    claim_years = set(extract_years(cleaned_claim))
    evidence_years = set(extract_years(cleaned_evidence))
    claim_dates = set(extract_dates(cleaned_claim))
    evidence_dates = set(extract_dates(cleaned_evidence))
    claim_numbers = set(extract_numbers(cleaned_claim))
    evidence_numbers = set(extract_numbers(cleaned_evidence))

    if not (claim_years or claim_dates or claim_numbers):
        return flags

    # Conflicting values in a shared context.
    if _shared_context(cleaned_claim, cleaned_evidence):
        if claim_years and evidence_years and not claim_years.issubset(evidence_years):
            flags.append("numeric_mismatch_with_evidence")
        if claim_dates and evidence_dates and not claim_dates.issubset(evidence_dates):
            flags.append("numeric_mismatch_with_evidence")
        if claim_numbers and evidence_numbers and not claim_numbers.issubset(evidence_numbers):
            flags.append("numeric_mismatch_with_evidence")

    # Missing explicit support for central numeric/year/date claims.
    if _numeric_claim_is_central(cleaned_claim):
        if claim_years and not claim_years.issubset(evidence_years):
            if not evidence_years:
                flags.append("claim_year_not_supported_by_evidence")
        if claim_dates and not claim_dates.issubset(evidence_dates):
            if not evidence_dates:
                flags.append("claim_date_not_supported_by_evidence")
        if claim_numbers and not claim_numbers.issubset(evidence_numbers):
            if not evidence_numbers:
                flags.append("claim_numeric_not_supported_by_evidence")

    return list(dict.fromkeys(flags))


def detect_entity_mismatch(claim: str, evidence: str) -> List[str]:
    flags: List[str] = []
    cleaned_claim = _clean(claim)
    cleaned_evidence = _clean(evidence)
    if not _shared_context(cleaned_claim, cleaned_evidence):
        return flags

    claim_entities = set(extract_capitalized_entities(cleaned_claim))
    evidence_entities = set(extract_capitalized_entities(cleaned_evidence))
    if not claim_entities or not evidence_entities:
        return flags
    if claim_entities.issubset(evidence_entities):
        return flags

    relation_overlap = any(
        relation in cleaned_claim.lower() and relation in cleaned_evidence.lower()
        for relation in _RELATION_PATTERNS
    )
    if relation_overlap and (claim_entities - evidence_entities) and (evidence_entities - claim_entities):
        flags.append("entity_mismatch_with_evidence")

    return flags


def run_factual_consistency_checks(claim: str, evidence: str) -> Dict[str, object]:
    cleaned_claim = _clean(claim)
    cleaned_evidence = _clean(evidence)
    flags: List[str] = []
    flags.extend(detect_numeric_mismatch(cleaned_claim, cleaned_evidence))
    flags.extend(detect_entity_mismatch(cleaned_claim, cleaned_evidence))
    flags = list(dict.fromkeys(flags))
    return {
        "flags": flags,
        "details": {
            "claim_years": extract_years(cleaned_claim),
            "evidence_years": extract_years(cleaned_evidence),
            "claim_dates": extract_dates(cleaned_claim),
            "evidence_dates": extract_dates(cleaned_evidence),
            "claim_numbers": extract_numbers(cleaned_claim),
            "evidence_numbers": extract_numbers(cleaned_evidence),
            "claim_entities": extract_capitalized_entities(cleaned_claim),
            "evidence_entities": extract_capitalized_entities(cleaned_evidence),
        },
    }
