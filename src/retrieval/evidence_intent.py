"""Evidence intent classification and credibility scoring utilities.

The retrieval stack uses these helpers as a light-weight factuality layer on top
of the existing dense/BM25/RRF ranking.  The functions are intentionally
heuristic and dependency-free so they can run during retrieval, answerability
checking, correction, and tests without loading LLMs or embedding models.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple


EvidenceType = str

FACTUAL: EvidenceType = "FACTUAL"
EXPLANATORY: EvidenceType = "EXPLANATORY"
EXAMPLE: EvidenceType = "EXAMPLE"
REFERENCE: EvidenceType = "REFERENCE"
DISCUSSION: EvidenceType = "DISCUSSION"
NOISE: EvidenceType = "NOISE"

EXAMPLE_PATTERNS: Tuple[str, ...] = (
    r"\bfor example\b",
    r"\bfor instance\b",
    r"\bexample\s*:",
    r"\ba claim such as\b",
    r"\bif a claim says\b",
    r"\bsuppose\b",
    r"\bimagine\b",
    r"\bhypothetical\b",
    r"\bconsider the claim\b",
    r"\btutorial\b",
    r"\bdemonstration\b",
    # Educational fact-verification snippets often contain factual-looking
    # claims but are not source evidence for the claim itself.  They explain how
    # verification works, so strict factual mode must not treat them as support.
    r"\bfactually contradictory\b",
    r"\bsemantic closeness\b",
    r"\bnli label\b",
    r"\bevidence is the premise\b",
    r"\bclaim is the hypothesis\b",
    r"\bentailment means\b",
    r"\bcontradiction means\b",
)

TUTORIAL_PATTERNS: Tuple[str, ...] = (
    r"\btutorial\b",
    r"\bdemonstration\b",
    r"\bdemo\b",
    r"\bwalkthrough\b",
    r"\bstep[- ]by[- ]step\b",
    r"\bhow to\b",
)

HYPOTHETICAL_PATTERNS: Tuple[str, ...] = (
    r"\bsuppose\b",
    r"\bimagine\b",
    r"\bhypothetical\b",
    r"\bif a claim says\b",
    r"\bconsider the claim\b",
    r"\ba claim such as\b",
)

STRICT_FACTUAL_QUERY_PATTERNS: Tuple[str, ...] = (
    r"\bintroduced\b",
    r"\bpublished\b",
    r"\breleased\b",
    r"\binvented\b",
    r"\bcreated\b",
    r"\bfounded\b",
    r"\bestablished\b",
    r"\blaunched\b",
    r"\bbuilt\b",
    r"\byear\b",
    r"\bdate\b",
    r"\bwhen\b",
    r"\bwhere\b",
    r"\bwho\s+(?:introduced|invented|founded|created|developed|proposed|published|released|launched|built)\b",
    r"\bwhen\s+(?:was|were|did)\b",
    r"\bwhere\s+(?:was|were|is|are)\b",
)

FACTUAL_ASSERTION_PATTERNS: Tuple[str, ...] = (
    r"\bintroduced by\b",
    r"\bproposed by\b",
    r"\bdeveloped by\b",
    r"\bpublished in\b",
    r"\breleased in\b",
    r"\binvented by\b",
    r"\bcreated by\b",
    r"\bfounded by\b",
    r"\bestablished by\b",
    r"\bbuilt by\b",
    r"\blaunched by\b",
    r"\bfounded in\b",
    r"\bestablished in\b",
    r"\blaunched in\b",
    r"\bwas introduced\b",
    r"\bwere introduced\b",
    r"\bwas proposed\b",
    r"\bwere proposed\b",
    r"\bwas founded\b",
    r"\bwere founded\b",
    r"\bwas invented\b",
    r"\bwas created\b",
    r"\bwas launched\b",
    r"\bwe introduce\b",
    r"\bwe introduced\b",
    r"\bwe propose\b",
    r"\bwe proposed\b",
    r"\bthis paper introduces\b",
    r"\bthis paper proposes\b",
)

REFERENCE_SECTIONS = {
    "reference",
    "references",
    "bibliography",
    "works cited",
    "citation",
    "citations",
}

HIGH_PRIORITY_SECTIONS = {
    "abstract",
    "introduction",
    "methodology",
    "methods",
    "method",
    "approach",
}

MEDIUM_PRIORITY_SECTIONS = {
    "background",
    "overview",
    "related work",
    "discussion",
    "analysis",
    "body",
}

LOW_PRIORITY_SECTIONS = {
    "tutorial",
    "example",
    "examples",
    "demonstration",
    "demo",
    "appendix",
    "conclusion",
    "limitations",
    "future work",
}

ORIGINAL_RESEARCH_HINTS = {
    "lewis_rag",
    "rag-sequence",
    "rag-token",
    "colbert",
    "dpr",
    "realm",
    "truthfulqa",
    "factverification",
    "fact_verification",
}

SURVEY_HINTS = {
    "survey",
    "review",
    "hallucinationsurvey",
    "ragtruth",
    "hallulens",
    "refind",
    "proofver",
}

TRAILING_NOISE_RE = re.compile(r"^[\W_]+$|(?:\b\w\b\s*){8,}", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-]{1,}")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _metadata_text(metadata: Dict[str, Any] | None) -> str:
    metadata = metadata or {}
    return " ".join(
        str(metadata.get(key, ""))
        for key in ("file_name", "source_rel", "document_title", "section_name", "file_type")
    ).lower()


def _matches_any(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _matched_patterns(patterns: Iterable[str], text: str) -> List[str]:
    hits: List[str] = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def _section(metadata: Dict[str, Any] | None) -> str:
    return _normalize(str((metadata or {}).get("section_name", "")))


def is_strict_factual_query(query: str) -> bool:
    q = _normalize(query)
    return _matches_any(STRICT_FACTUAL_QUERY_PATTERNS, q)


def has_example_or_tutorial_signal(text: str, metadata: Dict[str, Any] | None = None) -> bool:
    haystack = f"{_normalize(text)} {_metadata_text(metadata)}"
    return _matches_any(EXAMPLE_PATTERNS, haystack) or _matches_any(TUTORIAL_PATTERNS, haystack)


def has_hypothetical_signal(text: str, metadata: Dict[str, Any] | None = None) -> bool:
    haystack = f"{_normalize(text)} {_metadata_text(metadata)}"
    return _matches_any(HYPOTHETICAL_PATTERNS, haystack)


def has_factual_assertion(text: str) -> bool:
    return _matches_any(FACTUAL_ASSERTION_PATTERNS, _normalize(text))


def factual_assertion_score(text: str) -> float:
    """Return a 0..1 score for explicit factual relation statements."""
    clean = _normalize(text)
    if not clean:
        return 0.0
    hits = _matched_patterns(FACTUAL_ASSERTION_PATTERNS, clean)
    if not hits:
        # A weaker but still useful signal for dated factual statements.
        if re.search(r"\b(?:in|by|during)\s+(?:19|20)\d{2}\b", clean):
            return 0.35
        return 0.0
    strong_hits = sum(1 for pattern in hits if " by" in pattern or " in" in pattern)
    return min(1.0, 0.65 + 0.15 * len(hits) + 0.10 * strong_hits)


def classify_evidence_intent(text: str, metadata: Dict[str, Any] | None = None) -> EvidenceType:
    """Classify a retrieved chunk by its intent.

    EXAMPLE patterns intentionally take precedence over factual-looking text so
    tutorial/demo snippets cannot become support evidence by lexical overlap.
    """
    raw = text or ""
    clean = _normalize(raw)
    metadata_text = _metadata_text(metadata)
    section = _section(metadata)
    haystack = f"{clean} {metadata_text}"

    if not clean or len(TOKEN_RE.findall(clean)) < 2:
        return NOISE
    if TRAILING_NOISE_RE.search(clean) and len(TOKEN_RE.findall(clean)) < 20:
        return NOISE
    alnum = sum(ch.isalnum() for ch in clean)
    if clean and alnum / max(1, len(clean)) < 0.45:
        return NOISE

    if _matches_any(EXAMPLE_PATTERNS, haystack):
        return EXAMPLE

    if section in REFERENCE_SECTIONS or any(section.startswith(s) for s in REFERENCE_SECTIONS):
        return REFERENCE
    if re.match(r"^\s*(references|bibliography)\b", raw, flags=re.IGNORECASE):
        return REFERENCE
    if len(re.findall(r"\[[0-9]{1,3}\]|doi:|arxiv:", clean, flags=re.IGNORECASE)) >= 3:
        return REFERENCE

    if section in LOW_PRIORITY_SECTIONS or any(section.startswith(s) for s in LOW_PRIORITY_SECTIONS):
        if _matches_any(TUTORIAL_PATTERNS, haystack):
            return EXAMPLE
        return DISCUSSION

    if has_factual_assertion(clean):
        return FACTUAL

    if section in HIGH_PRIORITY_SECTIONS or any(section.startswith(s) for s in HIGH_PRIORITY_SECTIONS):
        return FACTUAL

    if section in {"discussion", "conclusion", "limitations", "future work"} or any(
        marker in clean for marker in ("we discuss", "the results suggest", "may indicate", "could be", "future work")
    ):
        return DISCUSSION

    if any(marker in clean for marker in (" is a ", " is an ", " refers to ", " means ", " is defined as ")):
        return EXPLANATORY

    # Default to factual for declarative source text with named entities, years,
    # or explicit paper-language; otherwise explanatory is safer.
    if re.search(r"\b(?:19|20)\d{2}\b", clean) or re.search(r"\b[A-Z][A-Za-z]{2,}\b", raw):
        return FACTUAL
    return EXPLANATORY


def metadata_quality_score(metadata: Dict[str, Any] | None) -> float:
    metadata = metadata or {}
    useful = ["source_rel", "file_name", "file_type", "document_title", "section_name"]
    present = sum(1 for key in useful if metadata.get(key))
    score = present / len(useful)
    try:
        importance = float(metadata.get("importance_score", 0) or 0)
        score += max(0.0, min(0.2, importance / 25.0))
    except (TypeError, ValueError):
        pass
    return max(0.0, min(1.0, score))


def section_priority_score(metadata: Dict[str, Any] | None) -> float:
    section = _section(metadata)
    if not section:
        return 0.45
    if section in REFERENCE_SECTIONS or any(section.startswith(s) for s in REFERENCE_SECTIONS):
        return 0.0
    if section in HIGH_PRIORITY_SECTIONS or any(section.startswith(s) for s in HIGH_PRIORITY_SECTIONS):
        return 1.0
    if section in MEDIUM_PRIORITY_SECTIONS or any(section.startswith(s) for s in MEDIUM_PRIORITY_SECTIONS):
        return 0.65
    if section in LOW_PRIORITY_SECTIONS or any(section.startswith(s) for s in LOW_PRIORITY_SECTIONS):
        return 0.25
    return 0.5


def source_authority_score(metadata: Dict[str, Any] | None, evidence_type: EvidenceType) -> float:
    metadata = metadata or {}
    haystack = _metadata_text(metadata)
    file_type = str(metadata.get("file_type", "")).lower()

    if evidence_type == NOISE:
        return 0.0
    if evidence_type == REFERENCE:
        return 0.15
    if evidence_type == EXAMPLE:
        return 0.10
    if _matches_any(TUTORIAL_PATTERNS, haystack):
        return 0.15

    if any(hint in haystack.replace("-", "_") for hint in ORIGINAL_RESEARCH_HINTS):
        return 1.0
    if any(hint in haystack.replace("-", "_") for hint in SURVEY_HINTS):
        return 0.55
    if file_type == "pdf":
        return 0.75
    if file_type in {"txt", "md", "json"}:
        # Curated KB files are reliable, but rank below original papers for
        # authorship/date questions.
        return 0.70
    return 0.50


def _semantic_similarity_component(item: Dict[str, Any]) -> float:
    for key in ("dense_similarity", "similarity", "semantic_similarity", "base_final_score", "weighted_score", "final_score"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    rrf = item.get("rrf_score")
    try:
        return max(0.0, min(1.0, float(rrf) * 30.0)) if rrf is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def evidence_penalty_components(text: str, metadata: Dict[str, Any] | None, evidence_type: EvidenceType) -> Dict[str, float]:
    haystack = f"{_normalize(text)} {_metadata_text(metadata)}"
    example = 1.0 if evidence_type == EXAMPLE or _matches_any(EXAMPLE_PATTERNS, haystack) else 0.0
    tutorial = 0.70 if _matches_any(TUTORIAL_PATTERNS, haystack) else 0.0
    hypothetical = 0.80 if _matches_any(HYPOTHETICAL_PATTERNS, haystack) else 0.0
    if evidence_type == NOISE:
        example = max(example, 0.50)
    return {
        "example_penalty": example,
        "tutorial_penalty": tutorial,
        "hypothetical_penalty": hypothetical,
    }


def compute_evidence_credibility(item: Dict[str, Any], query: str | None = None) -> Dict[str, Any]:
    """Return credibility components and normalized evidence confidence.

    The raw score follows the requested formula.  The public
    evidence_credibility_score is normalized to 0..1 for answerability and
    downstream calibration, while evidence_credibility_raw preserves the direct
    additive value.
    """
    text = str(item.get("text", ""))
    metadata = item.get("metadata", {}) or {}
    evidence_type = str(item.get("evidence_type") or classify_evidence_intent(text, metadata))

    semantic_similarity = _semantic_similarity_component(item)
    metadata_quality = metadata_quality_score(metadata)
    section_priority = section_priority_score(metadata)
    source_authority = source_authority_score(metadata, evidence_type)
    factual_score = factual_assertion_score(text)
    penalties = evidence_penalty_components(text, metadata, evidence_type)

    raw = (
        semantic_similarity
        + metadata_quality
        + section_priority
        + source_authority
        + factual_score
        - penalties["example_penalty"]
        - penalties["tutorial_penalty"]
        - penalties["hypothetical_penalty"]
    )
    normalized = max(0.0, min(1.0, raw / 5.0))

    strict_mode = is_strict_factual_query(query or "")
    rejected = evidence_type == NOISE or (
        strict_mode
        and (
            evidence_type == EXAMPLE
            or penalties["tutorial_penalty"] > 0.0
            or penalties["hypothetical_penalty"] > 0.0
        )
    )

    components: Dict[str, Any] = {
        "evidence_type": evidence_type,
        "semantic_similarity_component": round(semantic_similarity, 6),
        "metadata_quality": round(metadata_quality, 6),
        "section_priority": round(section_priority, 6),
        "source_authority": round(source_authority, 6),
        "factual_assertion_score": round(factual_score, 6),
        "example_penalty": round(penalties["example_penalty"], 6),
        "tutorial_penalty": round(penalties["tutorial_penalty"], 6),
        "hypothetical_penalty": round(penalties["hypothetical_penalty"], 6),
        "evidence_credibility_raw": round(raw, 6),
        "evidence_credibility_score": round(normalized, 6),
        "strict_factual_mode": strict_mode,
        "rejected_by_strict_factual_mode": bool(rejected),
    }
    return components


def annotate_evidence(item: Dict[str, Any], query: str | None = None) -> Dict[str, Any]:
    copied = dict(item)
    metadata = copied.get("metadata", {}) or {}
    copied["metadata"] = metadata
    copied["evidence_type"] = classify_evidence_intent(str(copied.get("text", "")), metadata)
    copied.update(compute_evidence_credibility(copied, query=query))
    return copied


def annotate_evidence_list(evidence: Iterable[Dict[str, Any]], query: str | None = None) -> List[Dict[str, Any]]:
    return [annotate_evidence(item, query=query) for item in evidence]


def is_rejected_support_evidence(item: Dict[str, Any], strict_factual_mode: bool | None = None) -> bool:
    evidence_type = str(item.get("evidence_type") or classify_evidence_intent(str(item.get("text", "")), item.get("metadata", {})))
    if evidence_type in {NOISE, REFERENCE, EXAMPLE}:
        return True
    if strict_factual_mode is None:
        strict_factual_mode = bool(item.get("strict_factual_mode"))
    if strict_factual_mode and (
        item.get("tutorial_penalty", 0.0) or item.get("hypothetical_penalty", 0.0)
    ):
        return True
    return False


def is_credible_support_evidence(
    item: Dict[str, Any],
    min_credibility: float = 0.25,
    strict_factual_mode: bool | None = None,
) -> bool:
    if is_rejected_support_evidence(item, strict_factual_mode=strict_factual_mode):
        return False
    try:
        score = float(item.get("evidence_credibility_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return score >= min_credibility


def rank_key(item: Dict[str, Any], strict_factual_mode: bool = False) -> Tuple[float, float, float]:
    try:
        final_score = float(item.get("final_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        final_score = 0.0
    try:
        credibility = float(item.get("evidence_credibility_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        credibility = 0.0
    try:
        factual = float(item.get("factual_assertion_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        factual = 0.0
    if strict_factual_mode:
        return (credibility + 0.15 * factual, final_score, factual)
    return (final_score, credibility, factual)
