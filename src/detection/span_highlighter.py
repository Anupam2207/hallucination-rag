"""REFIND-inspired lightweight span highlighting.

This is not the exact REFIND Context Sensitivity Ratio because Ollama does not
expose reliable token log-probabilities. Instead, this module highlights the
suspicious factual spans that triggered rule/NLI checks.
"""

from __future__ import annotations

import re
from typing import Dict, List

from src.detection.factual_consistency import extract_numbers, extract_years
from src.utils.text_cleaning import normalize_for_detection

_REASON_PATTERNS = {
    "fine_tuning": [r"fine[- ]?tun\w*", r"finetun\w*"],
    "training_data": [r"training data", r"labeled data", r"less data", r"smaller amounts of labeled data"],
    "performance_comparison": [r"better than", r"outperform\w*", r"improved performance", r"higher accuracy than", r"more efficient than"],
    "unsupported_task": [
        r"machine translation", r"language translation", r"sentiment analysis",
        r"text classification", r"document classification", r"summarization",
        r"content generation", r"dialogue systems?",
    ],
    "interpretability": [r"interpretability", r"interpretable", r"explainability", r"explainable", r"transparent", r"clear understanding", r"traceability"],
    "unsupported_domain": [
        r"collaborative BERT", r"collaborative filtering", r"recommendation systems?",
        r"recommendation accuracy", r"e-commerce", r"ecommerce", r"product descriptions",
        r"product reviews", r"reviews", r"open[- ]source library",
    ],
}

_FLAG_REASON_MAP = {
    "numeric_mismatch_with_evidence": "numeric_or_date_mismatch",
    "claim_year_not_supported_by_evidence": "claim_year_not_supported_by_evidence",
    "claim_date_not_supported_by_evidence": "claim_date_not_supported_by_evidence",
    "claim_numeric_not_supported_by_evidence": "claim_numeric_not_supported_by_evidence",
    "entity_mismatch_with_evidence": "entity_mismatch",
    "fine_tuning_not_in_evidence": "unsupported_fine_tuning",
    "training_data_requirement_not_in_evidence": "unsupported_training_data_claim",
    "performance_comparison_not_in_evidence": "unsupported_performance_comparison",
    "interpretability_not_in_evidence": "unsupported_interpretability_claim",
    "unsupported_task_example_not_in_evidence": "unsupported_task_example",
    "machine_translation_not_in_evidence": "unsupported_task_example",
    "text_classification_not_in_evidence": "unsupported_task_example",
    "document_classification_not_in_evidence": "unsupported_task_example",
    "summarization_not_in_evidence": "unsupported_task_example",
    "content_generation_not_in_evidence": "unsupported_task_example",
    "sentiment_analysis_not_in_evidence": "unsupported_task_example",
    "dialogue_systems_not_in_evidence": "unsupported_task_example",
    "nli_contradiction": "nli_contradiction",
    "nli_neutral": "nli_neutral",
    "collaborative_filtering_not_in_evidence": "unsupported_domain_term",
    "recommendation_system_not_in_evidence": "unsupported_domain_term",
    "collaborative_bert_not_in_evidence": "unsupported_domain_term",
    "open_source_library_not_in_evidence": "unsupported_domain_term",
    "ecommerce_not_in_evidence": "unsupported_domain_term",
    "product_review_not_in_evidence": "unsupported_domain_term",
}


def _add_span(spans: List[Dict[str, object]], claim: str, text: str, reason: str) -> None:
    if not text:
        return
    start = claim.find(text)
    if start < 0:
        return
    end = start + len(text)
    if any(span["start"] == start and span["end"] == end for span in spans):
        return
    spans.append({"text": text, "start": start, "end": end, "reason": reason})


def _regex_spans(claim: str, patterns: List[str], reason: str) -> List[Dict[str, object]]:
    spans: List[Dict[str, object]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, claim, flags=re.IGNORECASE):
            spans.append({
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "reason": reason,
            })
    return spans


def detect_hallucinated_spans(claim: str, evidence: str, rule_flags: List[str] | None = None) -> List[Dict[str, object]]:
    rule_flags = rule_flags or []
    spans: List[Dict[str, object]] = []

    clean_claim = normalize_for_detection(claim)
    clean_evidence = normalize_for_detection(evidence)

    numeric_flags = {"numeric_mismatch_with_evidence", "claim_year_not_supported_by_evidence", "claim_date_not_supported_by_evidence", "claim_numeric_not_supported_by_evidence"}
    if any(flag in rule_flags for flag in numeric_flags):
        evidence_values = set(extract_years(clean_evidence)) | set(extract_numbers(clean_evidence))
        for value in extract_years(clean_claim) + extract_numbers(clean_claim):
            if value not in evidence_values:
                reason = "numeric_or_date_mismatch" if "numeric_mismatch_with_evidence" in rule_flags else "claim_year_or_number_not_supported"
                _add_span(spans, claim, value, reason)

    for flag in rule_flags:
        reason = _FLAG_REASON_MAP.get(flag)
        if not reason:
            continue
        if "fine_tuning" in flag:
            spans.extend(_regex_spans(claim, _REASON_PATTERNS["fine_tuning"], reason))
        elif "training_data" in flag:
            spans.extend(_regex_spans(claim, _REASON_PATTERNS["training_data"], reason))
        elif "performance" in flag:
            spans.extend(_regex_spans(claim, _REASON_PATTERNS["performance_comparison"], reason))
        elif "interpretability" in flag:
            spans.extend(_regex_spans(claim, _REASON_PATTERNS["interpretability"], reason))
        elif flag in {"collaborative_filtering_not_in_evidence", "recommendation_system_not_in_evidence", "collaborative_bert_not_in_evidence", "open_source_library_not_in_evidence", "ecommerce_not_in_evidence", "product_review_not_in_evidence"}:
            spans.extend(_regex_spans(claim, _REASON_PATTERNS["unsupported_domain"], reason))
        elif "task" in flag or flag.endswith("_not_in_evidence"):
            spans.extend(_regex_spans(claim, _REASON_PATTERNS["unsupported_task"], reason))

    # Stable order and deduplication.
    unique: Dict[tuple[int, int, str], Dict[str, object]] = {}
    for span in spans:
        unique[(int(span["start"]), int(span["end"]), str(span["reason"]))] = span
    return sorted(unique.values(), key=lambda item: (int(item["start"]), int(item["end"])))


def highlight_claim(claim: str, spans: List[Dict[str, object]]) -> str:
    if not spans:
        return claim
    pieces: List[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: int(item["start"])):
        start = int(span["start"])
        end = int(span["end"])
        if start < cursor:
            continue
        pieces.append(claim[cursor:start])
        pieces.append("[" + claim[start:end] + "]")
        cursor = end
    pieces.append(claim[cursor:])
    return "".join(pieces)


def highlight_hallucinated_spans(claim: str, evidence: str, rule_flags: List[str] | None = None) -> Dict[str, object]:
    spans = detect_hallucinated_spans(claim, evidence, rule_flags)
    return {
        "highlighted_claim": highlight_claim(claim, spans),
        "hallucinated_spans": spans,
    }
