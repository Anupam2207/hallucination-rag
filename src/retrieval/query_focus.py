import re
from typing import Any, Dict, Iterable, List, Set

from src.config import get_config_value

TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-]{1,}")

STOPWORDS = {
    "what", "who", "when", "where", "why", "how", "is", "are", "was", "were", "the", "a", "an",
    "of", "for", "to", "in", "on", "with", "and", "or", "by", "about", "explain", "define", "meaning",
    "overview", "introduction", "tell", "me", "please", "does", "do", "did", "be", "been", "being", "it",
    "this", "that", "these", "those", "into", "from", "using", "use", "used", "uses", "give", "short",
    "answer", "details", "brief", "briefly", "describe", "write", "note", "notes",
}

RELATION_TERMS = {
    "introduced", "invented", "created", "proposed", "released", "published", "developed",
    "author", "authors", "paper", "who", "when", "what", "explain", "define", "built", "won", "started",
    "began", "launched", "founded", "designed", "made", "wrote", "written", "year", "date",
}

# Canonical topic aliases. Keep these lightweight; they are used only for
# retrieval grounding and support-calibration, not for generation.
ALIAS_MAP: Dict[str, List[str]] = {
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "colbert": ["colbert", "contextualized late interaction", "contextualized late interaction over bert", "late interaction retrieval"],
    "truthfulqa": ["truthfulqa", "truthful qa"],
    "dpr": ["dpr", "dense passage retrieval", "dense passage retriever"],
    "realm": ["realm", "retrieval augmented language model pretraining"],
    "bm25": ["bm25", "sparse retrieval", "lexical retrieval"],
    "rrf": ["rrf", "reciprocal rank fusion"],
    "chromadb": ["chromadb", "chroma database", "vector database"],
    "llm": ["llm", "large language model", "large language models"],
    "hallucination": ["hallucination", "hallucinations", "llm hallucination", "large language model hallucination"],
    "fact verification": ["fact verification", "factual verification", "claim verification", "support scoring"],
    "formula 1": ["formula 1", "formula1", "f1", "formula one"],
    "vehicle safety": ["vehicle safety", "automotive safety", "car safety", "airbag", "seat belt"],
    "smartphone": ["smartphone", "mobile phone", "mobile device", "smartphones"],
    "ai newsroom": ["ai newsroom", "artificial intelligence in newsrooms", "ai in journalism", "modern newsrooms"],
    "ancient indian architecture": ["ancient indian architecture", "indian architecture", "temple architecture", "stupa", "rock cut"],
    "mfa": ["mfa", "multi factor authentication", "multi-factor authentication", "multifactor authentication"],
    "phishing": ["phishing", "spear phishing", "suspicious links", "verification codes"],
    "eiffel tower": ["eiffel tower", "gustave eiffel", "champ de mars", "1889 exposition"],
    "kubernetes": ["kubernetes", "container orchestration", "orchestrate containers"],
    "crispr": ["crispr", "crispr-cas9", "cas9", "gene editing"],
    "black hole": ["black hole", "black holes", "event horizon"],
    "crop rotation": ["crop rotation", "rotating crops", "planned crop sequence"],
    "drip irrigation": ["drip irrigation", "emitters", "filters", "clog"],
}

COMPETING_TOPICS: Dict[str, Set[str]] = {
    "rag": {"colbert", "dpr", "realm", "truthfulqa", "bm25", "formula 1"},
    "colbert": {"rag", "dpr", "realm", "truthfulqa", "bm25", "formula 1"},
    "formula 1": {"rag", "colbert", "vehicle safety", "smartphone"},
    "crop rotation": {"drip irrigation"},
    "drip irrigation": {"crop rotation"},
}

DEFINITION_PATTERNS = (
    "what is", "define", "explain", "overview of", "meaning of", "introduction to", "what are", "describe"
)

DEFINITION_TEXT_PATTERNS = (
    " is a ", " is an ", " refers to ", " is defined as ", " means ", " requires ",
    " combines ", " connects ", " model", " method", " system", " framework", " technique",
)

GOOD_DEFINITION_SECTIONS = {"introduction", "background", "core concepts", "overview", "body", "abstract"}
BAD_DEFINITION_SECTIONS = {
    "results", "experiments", "experiment", "evaluation", "rq", "datasets", "datasets & metrics",
    "references", "copyright", "appendix", "ablation", "discussion"
}

GENERIC_ENTITY_TERMS = {
    "retrieval", "generation", "model", "models", "method", "system", "paper", "year", "date",
    "introduced", "proposed", "created", "developed", "published", "released", "author", "authors",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("-", " ").lower()).strip()


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


def _phrase_tokens(phrase: str) -> Set[str]:
    return {t for t in tokenize(phrase) if t not in STOPWORDS and t not in RELATION_TERMS and t not in GENERIC_ENTITY_TERMS}


def is_definition_query(query: str) -> bool:
    q_norm = normalize_text(query)
    return any(pattern in q_norm for pattern in DEFINITION_PATTERNS)


def _contains_alias(text_norm: str, aliases: Iterable[str]) -> bool:
    text_tokens = set(tokenize(text_norm))
    for alias in aliases:
        alias_norm = normalize_text(alias)
        if not alias_norm:
            continue
        if f" {alias_norm} " in f" {text_norm} ":
            return True
        alias_tokens = _phrase_tokens(alias_norm)
        if alias_tokens and alias_tokens.issubset(text_tokens):
            return True
    return False


def _metadata_text(metadata: Dict[str, Any] | None) -> str:
    metadata = metadata or {}
    return " ".join(
        str(metadata.get(key, "") or "")
        for key in ("file_name", "document_title", "section_name", "source_rel")
    )


def extract_query_focus(query: str) -> Dict[str, Any]:
    q_norm = normalize_text(query)
    tokens = tokenize(q_norm)
    relation_terms = [t for t in tokens if t in RELATION_TERMS]
    alias_groups: Dict[str, List[str]] = {}
    core_terms: Set[str] = set()

    for canonical, aliases in ALIAS_MAP.items():
        if _contains_alias(q_norm, aliases):
            alias_groups[canonical] = aliases
            core_terms.update(_phrase_tokens(canonical))
            for item in aliases:
                core_terms.update(_phrase_tokens(item))

    if not core_terms:
        for token in tokens:
            if token in STOPWORDS or token in RELATION_TERMS or token in GENERIC_ENTITY_TERMS:
                continue
            if len(token) > 2:
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


def text_contains_known_topic(text: str, metadata: Dict[str, Any] | None = None) -> Set[str]:
    haystack = normalize_text(" ".join([text or "", _metadata_text(metadata)]))
    found: Set[str] = set()
    for canonical, aliases in ALIAS_MAP.items():
        if _contains_alias(haystack, aliases):
            found.add(canonical)
    return found


def has_competing_topic(text: str, metadata: Dict[str, Any] | None, focus: Dict[str, Any]) -> bool:
    query_topics = set((focus.get("alias_groups") or {}).keys())
    if not query_topics:
        return False
    present = text_contains_known_topic(text, metadata)
    for query_topic in query_topics:
        if present.intersection(COMPETING_TOPICS.get(query_topic, set())):
            return True
    return False


def topical_score(text: str, metadata: Dict[str, Any] | None, focus: Dict[str, Any]) -> float:
    text_norm = normalize_text(" ".join([text or "", _metadata_text(metadata)]))
    alias_groups = focus.get("alias_groups") or {}
    core_terms = set(focus.get("core_entity_terms") or [])
    if not alias_groups and not core_terms:
        return 0.5

    if alias_groups:
        # Any explicit alias/entity match is strong evidence of topicality.
        for aliases in alias_groups.values():
            if _contains_alias(text_norm, aliases):
                return 1.0
        return 0.0

    text_tokens = set(tokenize(text_norm))
    if core_terms:
        overlap = len(core_terms.intersection(text_tokens)) / max(1, len(core_terms))
        return max(0.0, min(0.85, overlap))
    return 0.0


def has_entity_match(query: str, chunk_text: str, metadata: Dict[str, Any] | None = None) -> bool:
    focus = extract_query_focus(query)
    return topical_score(chunk_text, metadata, focus) > 0.0 and not has_competing_topic(chunk_text, metadata, focus)


def has_topical_match(text: str, metadata: Dict[str, Any] | None, focus: Dict[str, Any]) -> bool:
    core_terms = set(focus.get("core_entity_terms") or [])
    alias_groups = focus.get("alias_groups") or {}
    if not core_terms and not alias_groups:
        return True
    return topical_score(text, metadata, focus) > 0.0 and not has_competing_topic(text, metadata, focus)


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
        score -= 0.40
    if re.search(r"\bRQ\d\b|datasets?\s*&\s*metrics|ablation|experiment", text or "", re.IGNORECASE):
        score -= 0.30
    return max(0.0, min(1.0, score))


def _keyword_list(config_key: str, default: list[str]) -> list[str]:
    value = get_config_value("settings", "knowledge_base", config_key, default=default)
    return [str(item).lower() for item in (value or [])]


def source_quality(metadata: Dict[str, Any] | None) -> float:
    metadata = metadata or {}
    file_name = str(metadata.get("file_name", "")).lower()
    file_type = str(metadata.get("file_type", "")).lower()
    source_rel = str(metadata.get("source_rel", "")).lower()
    clean_keywords = _keyword_list("clean_pdf_keywords", [])
    research_keywords = _keyword_list("exclude_pdf_keywords", [])
    if file_type == "pdf" or "/pdf/" in source_rel or source_rel.startswith("raw/pdf"):
        if any(key.lower() in file_name for key in clean_keywords):
            return 1.0
        if any(key.lower() in file_name for key in research_keywords):
            return 0.65
        return 0.75
    if file_type in {"txt", "md", "json"}:
        # Kept for backward-compatible tests; runtime retrieval can still use
        # knowledge_base.pdf_only to filter non-PDF sources.
        return 0.9
    return 0.35


def is_pdf_source(metadata: Dict[str, Any] | None) -> bool:
    metadata = metadata or {}
    file_type = str(metadata.get("file_type", "")).lower()
    source_rel = str(metadata.get("source_rel", "")).replace("\\", "/").lower()
    return file_type == "pdf" or source_rel.startswith("raw/pdf/") or "/pdf/" in source_rel


def sparse_query_tokens(query: str) -> List[str]:
    focus = extract_query_focus(query)
    tokens = []
    for token in focus.get("tokens", []):
        if token in STOPWORDS:
            continue
        if token in RELATION_TERMS and not focus.get("core_entity_terms"):
            continue
        tokens.append(token)
    return tokens or [t for t in tokenize(query) if t not in STOPWORDS]
