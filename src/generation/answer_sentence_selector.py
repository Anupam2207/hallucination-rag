from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from src.retrieval.query_focus import (
    CONCEPT_MAP,
    concept_terms_for_focus,
    definition_score,
    extract_query_focus,
    find_competing_topics,
    has_entity_match,
    is_definition_query,
    normalize_text,
    sentence_has_topic_match,
    tokenize,
)


META_INSTRUCTION_PATTERNS = (
    "when answering questions about",
    "it is important not to confuse",
    "this example shows",
    "in this tutorial",
    "a robust verification system must",
    "the claim is",
    "the evidence is",
    "the premise",
    "the hypothesis",
    "support score",
    "nli label",
)

DEFINITION_PATTERNS = (
    " is a ",
    " is an ",
    " refers to ",
    " means ",
    " is used to ",
    " requires ",
    " combines ",
    " connects ",
    " stands for ",
)

DETAIL_PATTERNS = (
    r"\bin\s+(?:19|20)\d{2}\b",
    r"\bfirst\s+(?:image|observed|released)\b",
    r"\bcollaboration\s+released\b",
    r"\bimage\b",
    r"\bshadow\b",
    r"\bm87\b",
)

BAD_EVIDENCE_TYPES = {"EXAMPLE", "REFERENCE", "NOISE"}

STOPWORDS = {
    "what", "who", "when", "where", "why", "how", "the", "a", "an", "is", "are", "was", "were", "of", "to",
    "in", "on", "for", "with", "by", "and", "or", "explain", "define", "describe", "about", "does", "do", "did",
}


@dataclass(frozen=True)
class CandidateSentence:
    sentence: str
    score: float
    evidence_index: int
    sentence_index: int
    source: str
    reason: Dict[str, Any]


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pieces: List[str] = []
    for part in re.split(r"\n+", text):
        part = part.strip()
        if not part:
            continue
        pieces.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", part))
    cleaned = [re.sub(r"\s+", " ", piece).strip() for piece in pieces]
    return [piece for piece in cleaned if piece]


def _query_terms(query: str) -> set[str]:
    return {token for token in tokenize(query) if token not in STOPWORDS and len(token) > 2}


def _contains_meta_instruction(sentence: str) -> bool:
    lower = normalize_text(sentence)
    return any(pattern in lower for pattern in META_INSTRUCTION_PATTERNS)


def _is_definition_sentence(sentence: str) -> bool:
    lower = f" {normalize_text(sentence)} "
    return any(pattern in lower for pattern in DEFINITION_PATTERNS)


def _is_overly_specific_detail(sentence: str) -> bool:
    lower = normalize_text(sentence)
    return any(re.search(pattern, lower) for pattern in DETAIL_PATTERNS)


def _concept_hits(sentence: str, focus: Dict[str, Any]) -> int:
    lower = normalize_text(sentence)
    concepts = concept_terms_for_focus(focus)
    return sum(1 for concept in concepts if normalize_text(concept) in lower)


def _source_text(metadata: Dict[str, Any] | None) -> str:
    metadata = metadata or {}
    return " ".join(
        str(metadata.get(key, "") or "")
        for key in ("source_rel", "file_name", "document_title", "section_name")
    )


def _source_priority(metadata: Dict[str, Any] | None) -> float:
    section = normalize_text(str((metadata or {}).get("section_name", "")))
    if section in {"abstract", "introduction", "overview", "background", "body"}:
        return 0.35
    if any(section.startswith(prefix) for prefix in ("abstract", "introduction", "overview", "background")):
        return 0.25
    if section in {"references", "appendix", "results", "experiments"}:
        return -0.20
    return 0.0


def sentence_score(query: str, sentence: str, evidence_item: Dict[str, Any], evidence_index: int, sentence_index: int) -> CandidateSentence | None:
    metadata = evidence_item.get("metadata", {}) or {}
    evidence_type = str(evidence_item.get("evidence_type", "FACTUAL") or "FACTUAL").upper()
    if evidence_type in BAD_EVIDENCE_TYPES:
        return None

    stripped = sentence.strip()
    if len(stripped) < 18 or len(stripped.split()) < 4:
        return None
    if _contains_meta_instruction(stripped):
        return None

    focus = extract_query_focus(query)
    if find_competing_topics(focus, stripped, None):
        return None
    if not sentence_has_topic_match(focus, stripped, metadata):
        return None

    lower = normalize_text(stripped)
    q_terms = _query_terms(query)
    sentence_terms = set(tokenize(lower))
    overlap = len(q_terms & sentence_terms) / max(1, len(q_terms))

    score = 0.0
    reason: Dict[str, Any] = {}

    if has_entity_match(focus, stripped, metadata):
        score += 2.0
        reason["entity_match"] = 2.0

    term_score = min(1.0, overlap)
    score += term_score
    reason["query_overlap"] = round(term_score, 3)

    if is_definition_query(query):
        d_score = definition_score(query, stripped, metadata)
        if _is_definition_sentence(stripped):
            d_score = max(d_score, 0.65)
        score += 1.6 * d_score
        reason["definition_score"] = round(d_score, 3)
        concept_bonus = min(0.9, 0.25 * _concept_hits(stripped, focus))
        score += concept_bonus
        reason["concept_bonus"] = round(concept_bonus, 3)
        if _is_overly_specific_detail(stripped) and not _is_definition_sentence(stripped):
            score -= 0.9
            reason["specific_detail_penalty"] = -0.9
    else:
        if _is_definition_sentence(stripped):
            score += 0.5
            reason["definition_sentence_bonus"] = 0.5

    source_text = _source_text(metadata)
    if has_entity_match(focus, source_text, metadata):
        score += 0.45
        reason["same_source_topic"] = 0.45

    section_bonus = _source_priority(metadata)
    score += section_bonus
    if section_bonus:
        reason["section_priority"] = section_bonus

    early_bonus = max(0.0, 0.25 - 0.04 * sentence_index)
    score += early_bonus
    reason["early_sentence_bonus"] = round(early_bonus, 3)

    if evidence_type == "EXPLANATORY":
        score -= 0.10
        reason["explanatory_penalty"] = -0.10

    source = str(metadata.get("source_rel") or metadata.get("file_name") or evidence_item.get("chunk_id") or "")
    return CandidateSentence(stripped, round(score, 4), evidence_index, sentence_index, source, reason)


def rank_answer_sentences(query: str, evidence: Iterable[Dict[str, Any]]) -> List[CandidateSentence]:
    candidates: List[CandidateSentence] = []
    for evidence_index, item in enumerate(evidence):
        for sentence_index, sentence in enumerate(split_sentences(str(item.get("text", "")))):
            candidate = sentence_score(query, sentence, item, evidence_index, sentence_index)
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(key=lambda c: (-c.score, c.evidence_index, c.sentence_index))
    return candidates


def _dedupe_preserve_order(candidates: List[CandidateSentence]) -> List[CandidateSentence]:
    seen: set[str] = set()
    output: List[CandidateSentence] = []
    for candidate in candidates:
        key = normalize_text(candidate.sentence)
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def select_answer_sentences(query: str, evidence: Iterable[Dict[str, Any]], max_sentences: int = 3, min_score: float = 2.0) -> List[str]:
    ranked = _dedupe_preserve_order(rank_answer_sentences(query, evidence))
    if not ranked:
        return []
    ranked_for_selection = ranked
    if is_definition_query(query):
        non_detail = [candidate for candidate in ranked if "specific_detail_penalty" not in candidate.reason]
        if non_detail:
            ranked_for_selection = non_detail
    selected = [candidate for candidate in ranked_for_selection if candidate.score >= min_score]
    if not selected and ranked_for_selection and ranked_for_selection[0].score >= 1.4:
        selected = [ranked_for_selection[0]]
    selected = selected[:max_sentences]
    # Keep the strongest sentence first. For additional sentences from the same
    # source, preserve local source order to avoid jumbled answers.
    if len(selected) > 1:
        first = selected[0]
        rest = sorted(selected[1:], key=lambda c: (c.source != first.source, c.evidence_index, c.sentence_index))
        selected = [first, *rest]
    return [candidate.sentence for candidate in selected]
