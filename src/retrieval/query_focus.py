import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from src.config import get_config_value

TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-]{1,}")

STOPWORDS = {
    "what", "who", "when", "where", "why", "how", "is", "are", "was", "were", "the", "a", "an",
    "of", "for", "to", "in", "on", "with", "and", "or", "by", "about", "explain", "define", "meaning",
    "overview", "introduction", "tell", "me", "please", "does", "do", "did", "be", "been", "being", "it",
    "this", "that", "these", "those", "into", "from", "using", "use", "used", "uses"
}

RELATION_TERMS = {
    "introduced", "invented", "created", "proposed", "released", "published", "developed",
    "author", "authors", "paper", "who", "when", "what", "explain", "define", "built", "won", "started",
    "began", "launched", "founded", "designed", "created", "made", "wrote", "written"
}

ALIAS_MAP: Dict[str, List[str]] = {
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "colbert": ["colbert", "contextualized late interaction", "late interaction retrieval"],
    "truthfulqa": ["truthfulqa", "truthful qa"],
    "dpr": ["dpr", "dense passage retrieval"],
    "realm": ["realm", "retrieval augmented language model pretraining"],
    "bm25": ["bm25", "sparse retrieval", "lexical retrieval"],
    "chromadb": ["chromadb", "chroma database", "vector database"],
    "llm": ["llm", "large language model", "large language models"],
    "mfa": ["mfa", "multi-factor authentication", "multi factor authentication"],
    "domain name system": ["domain name system", "dns"],
    "vehicle safety": ["vehicle safety", "automotive safety", "car safety"],
    "smartphone": ["smartphone", "mobile phone", "mobile device", "smartphones"],
    "ai newsroom": ["ai newsroom", "artificial intelligence in newsrooms", "ai in journalism", "modern newsrooms"],
    "hallucination": ["hallucination", "hallucinations", "llm hallucination", "large language model hallucination"],
    "fact verification": ["fact verification", "factual verification", "claim verification"],
}

DEFINITION_PATTERNS = (
    "what is", "define", "explain", "overview of", "meaning of", "introduction to", "what are", "describe"
)

DEFINITION_TEXT_PATTERNS = (
    " is a ", " is an ", " refers to ", " is defined as ", " means ", " benchmark", " model", " method", " system", " framework"
)

GOOD_DEFINITION_SECTIONS = {"introduction", "background", "core concepts", "overview", "body", "abstract"}
BAD_DEFINITION_SECTIONS = {
    "results", "experiments", "experiment", "evaluation", "rq", "datasets", "datasets & metrics",
    "references", "copyright", "appendix", "ablation", "discussion"
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


def _phrase_tokens(phrase: str) -> Set[str]:
    return {t for t in tokenize(phrase) if t not in STOPWORDS and t not in RELATION_TERMS}


def extract_query_focus(query: str) -> Dict[str, Any]:
    q_norm = normalize_text(query)
    tokens = tokenize(q_norm)
    relation_terms = [t for t in tokens if t in RELATION_TERMS]
    alias_groups: Dict[str, List[str]] = {}
    core_terms: Set[str] = set()

    for canonical, aliases in ALIAS_MAP.items():
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if alias_norm and (alias_norm in q_norm or all(t in tokens for t in tokenize(alias_norm))):
                alias_groups[canonical] = aliases
                core_terms.update(_phrase_tokens(canonical))
                for item in aliases:
                    core_terms.update(_phrase_tokens(item))
                break

    if not core_terms:
        for token in tokens:
            if token not in STOPWORDS and token not in RELATION_TERMS and len(token) > 2:
                core_terms.add(token)

    return {
        "query": query,
        "normalized_query": q_norm,
        "tokens": tokens,
        "core_entity_terms": sorted(core_terms),
        "relation_terms": sorted(set(relation_terms)),
        "alias_groups": alias_groups,
        "is_definition_query": is_definition_query(query),
    }


def is_definition_query(query: str) -> bool:
    q_norm = normalize_text(query)
    return any(pattern in q_norm for pattern in DEFINITION_PATTERNS)


def _contains_alias(text_norm: str, aliases: Iterable[str]) -> bool:
    text_tokens = set(tokenize(text_norm))
    for alias in aliases:
        alias_norm = normalize_text(alias)
        if alias_norm in text_norm:
            return True
        alias_tokens = _phrase_tokens(alias_norm)
        if alias_tokens and alias_tokens.issubset(text_tokens):
            return True
    return False


def topical_score(text: str, metadata: Dict[str, Any] | None, focus: Dict[str, Any]) -> float:
    text_norm = normalize_text(" ".join([
        text or "",
        str((metadata or {}).get("file_name", "")),
        str((metadata or {}).get("document_title", "")),
        str((metadata or {}).get("section_name", "")),
        str((metadata or {}).get("source_rel", "")),
    ]))
    alias_groups = focus.get("alias_groups") or {}
    core_terms = set(focus.get("core_entity_terms") or [])
    if not alias_groups and not core_terms:
        return 0.5

    # For known entities/acronyms, require the canonical alias or one of its
    # explicit aliases to appear.  Do not let a generic token inside an alias
    # satisfy the entity match.  Example: a RAG query should not match a
    # ColBERT chunk only because that chunk contains the word "retrieval".
    if alias_groups:
        return 1.0 if any(_contains_alias(text_norm, aliases) for aliases in alias_groups.values()) else 0.0

    text_tokens = set(tokenize(text_norm))
    if core_terms:
        overlap = len(core_terms.intersection(text_tokens)) / max(1, len(core_terms))
        return max(0.0, min(0.8, overlap))
    return 0.0


def has_topical_match(text: str, metadata: Dict[str, Any] | None, focus: Dict[str, Any]) -> bool:
    core_terms = set(focus.get("core_entity_terms") or [])
    alias_groups = focus.get("alias_groups") or {}
    if not core_terms and not alias_groups:
        return True
    return topical_score(text, metadata, focus) > 0.0


def definition_score(query: str, text: str, metadata: Dict[str, Any] | None = None) -> float:
    if not is_definition_query(query):
        return 0.5
    text_norm = " " + normalize_text(text) + " "
    section = normalize_text(str((metadata or {}).get("section_name", "")))
    score = 0.0
    if any(pattern in text_norm for pattern in DEFINITION_TEXT_PATTERNS):
        score += 0.55
    if section in GOOD_DEFINITION_SECTIONS or any(section.startswith(s) for s in GOOD_DEFINITION_SECTIONS):
        score += 0.25
    if section in BAD_DEFINITION_SECTIONS or any(section.startswith(s) for s in BAD_DEFINITION_SECTIONS):
        score -= 0.45
    if re.search(r"\bRQ\d\b|datasets?\s*&\s*metrics|ablation|experiment", text or "", re.IGNORECASE):
        score -= 0.35
    return max(0.0, min(1.0, score))


def _keyword_list(config_key: str, default: list[str]) -> list[str]:
    value = get_config_value("settings", "knowledge_base", config_key, default=default)
    return [str(item).lower() for item in (value or [])]


def source_quality(metadata: Dict[str, Any] | None) -> float:
    metadata = metadata or {}
    file_name = str(metadata.get("file_name", "")).lower()
    file_type = str(metadata.get("file_type", "")).lower()
    clean_keywords = _keyword_list("clean_pdf_keywords", [])
    research_keywords = _keyword_list("exclude_pdf_keywords", [])
    if file_type in {"txt", "md", "json"}:
        return 0.9
    if file_type == "pdf":
        if any(key.lower() in file_name for key in clean_keywords):
            return 1.0
        if any(key.lower() in file_name for key in research_keywords):
            return 0.5
        return 0.6
    return 0.6


def sparse_query_tokens(query: str) -> List[str]:
    focus = extract_query_focus(query)
    tokens = []
    for token in focus.get("tokens", []):
        if token in STOPWORDS:
            continue
        # Keep relation terms only when an entity exists; never let them dominate alone.
        if token in RELATION_TERMS and not focus.get("core_entity_terms"):
            continue
        tokens.append(token)
    return tokens or [t for t in tokenize(query) if t not in STOPWORDS]
