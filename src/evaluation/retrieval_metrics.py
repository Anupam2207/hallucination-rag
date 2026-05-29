from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from src.utils.text_cleaning import normalize_for_detection


def _clean(text: str) -> str:
    return normalize_for_detection(text or "").lower()


def keyword_hit_rate(evidence: Iterable[Dict[str, Any]], expected_keywords: Iterable[str]) -> float:
    keywords = [_clean(str(keyword)) for keyword in expected_keywords or [] if str(keyword).strip()]
    if not keywords:
        return 0.0
    text = _clean(" ".join(str(item.get("text", "")) for item in evidence or []))
    hits = sum(1 for keyword in keywords if keyword and keyword in text)
    return round(hits / len(keywords), 4)


def evidence_contains_keyword(item: Dict[str, Any], expected_keywords: Iterable[str]) -> bool:
    text = _clean(str(item.get("text", "")))
    metadata = item.get("metadata", {}) or {}
    haystack = " ".join([text, _clean(" ".join(str(v) for v in metadata.values()))])
    return any(_clean(str(keyword)) in haystack for keyword in expected_keywords or [] if str(keyword).strip())


def first_relevant_rank(evidence: List[Dict[str, Any]], expected_keywords: Iterable[str]) -> int | None:
    for rank, item in enumerate(evidence or [], start=1):
        if evidence_contains_keyword(item, expected_keywords):
            return rank
    return None


def recall_at_k(records: Iterable[Dict[str, Any]], k: int) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    successes = sum(1 for row in rows if row.get("first_relevant_rank") and int(row["first_relevant_rank"]) <= k)
    return round(successes / len(rows), 4)


def mean_reciprocal_rank(records: Iterable[Dict[str, Any]]) -> float:
    values: list[float] = []
    for row in records:
        rank = row.get("first_relevant_rank")
        values.append(0.0 if not rank else 1.0 / int(rank))
    return round(sum(values) / len(values), 4) if values else 0.0


def answerability_retrieval_success(records: Iterable[Dict[str, Any]]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    successes = sum(1 for row in rows if row.get("answerability_retrieval_success"))
    return round(successes / len(rows), 4)


def average(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0
