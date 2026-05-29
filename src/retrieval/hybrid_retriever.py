import re
from typing import Any, Dict, List

from src.config import get_config_value
from src.retrieval.evidence_intent import annotate_evidence, is_strict_factual_query, rank_key
from src.retrieval.query_focus import extract_query_focus, topical_score
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.sparse_retriever import BM25SparseRetriever


_STOPWORDS = {
    "what", "who", "when", "where", "why", "how", "is", "are", "was", "were", "be",
    "the", "a", "an", "of", "to", "in", "on", "for", "with", "by", "and", "or", "from",
    "about", "explain", "define", "describe", "tell", "me", "please", "introduced", "invented",
    "created", "proposed", "released", "published", "launched", "developed", "began", "started",
}

_ALIAS_MAP = {
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "llm": ["llm", "large language model", "large language models"],
    "llms": ["llm", "large language model", "large language models"],
    "bm25": ["bm25"],
    "rrf": ["rrf", "reciprocal rank fusion"],
    "mfa": ["mfa", "multi-factor authentication", "multi factor authentication"],
    "dns": ["dns", "domain name system"],
    "domain name system": ["domain name system", "dns"],
    "colbert": ["colbert"],
    "truthfulqa": ["truthfulqa", "truthful qa"],
    "dpr": ["dpr", "dense passage retrieval", "dense passage retriever"],
    "realm": ["realm"],
    "fever": ["fever"],
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-]{1,}", (text or "").lower())


def extract_focus_terms(query: str) -> list[str]:
    terms: list[str] = []
    for token in _tokens(query):
        if token in _STOPWORDS:
            continue
        if len(token) < 3 and token not in {"ai", "qa"}:
            continue
        terms.append(token)
    # Keep insertion order and remove duplicates.
    return list(dict.fromkeys(terms))


def expand_query(query: str) -> str:
    """Conservative expansion only for relation words, never generic sections.

    Earlier versions added terms such as "abstract" and "introduction", which
    caused unrelated abstracts to outrank relevant definition chunks. This version
    keeps the user's key entity terms dominant.
    """
    q = (query or "").lower()
    additions: list[str] = []
    if "who introduced" in q or "who proposed" in q or "who invented" in q:
        additions.extend(["author", "authors", "paper", "proposed", "introduced"])
    if "which year" in q or "when" in q:
        additions.extend(["year", "published", "introduced"])
    return " ".join([query, *additions]).strip()


def importance_norm(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    return max(0.0, min(1.0, (score + 5.0) / 10.0))


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """Compatibility helper used by tests and simple RRF demonstrations."""
    fused: Dict[str, Dict[str, Any]] = {}
    for rank, item in enumerate(dense_results, start=1):
        cid = item.get("chunk_id") or f"dense_{rank}"
        fused[cid] = {**item, "dense_rank": rank, "sparse_rank": None}
    for rank, item in enumerate(sparse_results, start=1):
        cid = item.get("chunk_id") or f"sparse_{rank}"
        if cid not in fused:
            fused[cid] = {**item, "dense_rank": None}
        fused[cid]["sparse_rank"] = rank
        fused[cid]["sparse_score"] = item.get("sparse_score", 0)
    for item in fused.values():
        score = 0.0
        if item.get("dense_rank"):
            score += 1.0 / (rrf_k + item["dense_rank"])
        if item.get("sparse_rank"):
            score += 1.0 / (rrf_k + item["sparse_rank"])
        item["rrf_score"] = round(score, 6)
    results = list(fused.values())
    results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
    return results[:top_k]


class HybridRetriever:
    """Hybrid dense + sparse retriever with topical gating and RRF fusion."""

    def __init__(self, dense_retriever=None, sparse_retriever=None, vector_store=None, embedder=None):
        self.top_k = int(get_config_value("settings", "retrieval", "top_k", default=4))
        self.dense_top_k = int(get_config_value("settings", "retrieval", "dense_top_k", default=8))
        self.sparse_top_k = int(get_config_value("settings", "retrieval", "sparse_top_k", default=8))
        self.rrf_k = int(get_config_value("settings", "retrieval", "rrf_k", default=60))
        self.dense_weight = float(get_config_value("settings", "retrieval", "dense_weight", default=0.50))
        self.sparse_weight = float(get_config_value("settings", "retrieval", "sparse_weight", default=0.25))
        self.importance_weight = float(get_config_value("settings", "retrieval", "importance_weight", default=0.10))
        self.topical_weight = float(get_config_value("settings", "retrieval", "topical_weight", default=0.15))
        self.content_weight = float(get_config_value("settings", "retrieval", "content_weight", default=0.70))
        self.rrf_weight = float(get_config_value("settings", "retrieval", "rrf_weight", default=0.30))
        self.min_topical_score = float(get_config_value("settings", "retrieval", "min_topical_score", default=0.15))

        self.dense_retriever = dense_retriever or SemanticRetriever(vector_store=vector_store, embedder=embedder)
        self.sparse_retriever = sparse_retriever or BM25SparseRetriever()

    def _rrf(self, dense_rank: int | None, sparse_rank: int | None) -> float:
        score = 0.0
        if dense_rank:
            score += 1.0 / (self.rrf_k + dense_rank)
        if sparse_rank:
            score += 1.0 / (self.rrf_k + sparse_rank)
        return score

    @staticmethod
    def _haystack(item: Dict[str, Any]) -> str:
        metadata = item.get("metadata", {}) or {}
        return " ".join(
            str(part or "")
            for part in (
                item.get("text", ""),
                metadata.get("file_name"),
                metadata.get("document_title"),
                metadata.get("section_name"),
                metadata.get("source_rel"),
            )
        ).lower().replace("-", " ")

    @staticmethod
    def _aliases(term: str) -> list[str]:
        return _ALIAS_MAP.get(term.lower(), [term.lower().replace("-", " ")])

    def _topical_score(self, item: Dict[str, Any], focus_terms: list[str]) -> float:
        if not focus_terms:
            return 1.0
        haystack = self._haystack(item)
        matches = 0
        for term in focus_terms:
            if any(alias in haystack for alias in self._aliases(term)):
                matches += 1
        return matches / max(1, len(focus_terms))

    @staticmethod
    def _title_match_bonus(item: Dict[str, Any], focus_terms: list[str]) -> float:
        if not focus_terms:
            return 0.0
        metadata = item.get("metadata", {}) or {}
        title_haystack = " ".join(
            str(part or "") for part in (metadata.get("file_name"), metadata.get("document_title"), metadata.get("source_rel"))
        ).lower().replace("-", " ")
        hits = sum(1 for term in focus_terms if any(alias.replace("-", " ") in title_haystack for alias in _ALIAS_MAP.get(term, [term])))
        exact_phrase_bonus = 0.0
        for term in focus_terms:
            aliases = _ALIAS_MAP.get(term, [term])
            if any(alias.replace("-", " ") in title_haystack for alias in aliases):
                exact_phrase_bonus = max(exact_phrase_bonus, 0.20)
        return min(0.45, exact_phrase_bonus + 0.05 * hits)

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        top_k = int(top_k or self.top_k)
        expanded = expand_query(query)
        focus = extract_query_focus(query)
        focus_terms = list(focus.get("core_entity_terms") or extract_focus_terms(query))
        strict_factual_mode = is_strict_factual_query(query)
        must_match_topic = bool(focus.get("alias_groups")) or strict_factual_mode

        dense = self.dense_retriever.retrieve(expanded, top_k=max(self.dense_top_k, top_k))
        sparse = self.sparse_retriever.retrieve(expanded, top_k=max(self.sparse_top_k, top_k))

        fused: Dict[str, Dict[str, Any]] = {}
        for rank, item in enumerate(dense, start=1):
            cid = item.get("chunk_id") or f"dense_{rank}"
            fused[cid] = {**item, "dense_rank": item.get("dense_rank") or rank, "sparse_rank": None}

        for rank, item in enumerate(sparse, start=1):
            cid = item.get("chunk_id") or f"sparse_{rank}"
            if cid not in fused:
                fused[cid] = {**item, "dense_rank": None}
            fused[cid]["sparse_rank"] = item.get("sparse_rank") or rank
            fused[cid]["sparse_score"] = item.get("sparse_score", 0.0)
            # Preserve sparse text/metadata if the dense side did not have them.
            fused[cid].setdefault("text", item.get("text", ""))
            fused[cid].setdefault("metadata", item.get("metadata", {}))

        max_sparse = max((float(x.get("sparse_score") or 0.0) for x in fused.values()), default=1.0) or 1.0
        results: List[Dict[str, Any]] = []
        for item in fused.values():
            topical = topical_score(item.get("text", ""), item.get("metadata", {}) or {}, focus)
            # Strictly demote sparse-only/dense-only noise that does not mention the
            # query entity.  Entity aliases must match explicitly; a generic token
            # inside an alias (for example "retrieval" inside RAG) is not enough.
            if must_match_topic and focus_terms and topical <= 0.0:
                continue

            dense_score = float(item.get("dense_similarity") or item.get("similarity") or 0.0)
            sparse_score_raw = float(item.get("sparse_score") or 0.0)
            sparse_score = sparse_score_raw / max_sparse
            metadata = item.get("metadata", {}) or {}
            importance = importance_norm(metadata.get("importance_score", 0))
            rrf = self._rrf(item.get("dense_rank"), item.get("sparse_rank"))
            rrf_norm = min(1.0, rrf * 30.0)
            title_bonus = self._title_match_bonus(item, focus_terms)

            weighted = (
                self.dense_weight * dense_score
                + self.sparse_weight * sparse_score
                + self.importance_weight * importance
                + self.topical_weight * topical
                + title_bonus
            )
            final = self.content_weight * weighted + self.rrf_weight * rrf_norm

            copied = dict(item)
            copied["dense_similarity"] = dense_score if dense_score else item.get("dense_similarity")
            copied["sparse_score"] = round(sparse_score_raw, 6) if sparse_score_raw else item.get("sparse_score")
            copied["normalized_sparse_score"] = round(sparse_score, 6)
            copied["importance_score"] = metadata.get("importance_score", 0)
            copied["importance_norm"] = round(importance, 6)
            copied["topical_score"] = round(topical, 6)
            copied["weighted_score"] = round(weighted, 6)
            copied["rrf_score"] = round(rrf, 6)
            copied["base_final_score"] = round(final, 6)
            copied["final_score"] = round(final, 6)
            copied["retrieval_method"] = "hybrid"
            copied = annotate_evidence(copied, query=query)
            # Keep the existing relevance score dominant, but allow credibility,
            # source authority, and factual assertions to break semantic ties.
            credibility = float(copied.get("evidence_credibility_score") or 0.0)
            factual_boost = float(copied.get("factual_assertion_score") or 0.0)
            section_boost = float(copied.get("section_priority") or 0.0)
            adjusted_final = 0.65 * final + 0.35 * credibility
            adjusted_final += 0.30 * title_bonus
            if strict_factual_mode:
                adjusted_final += 0.10 * factual_boost + 0.05 * section_boost
            copied["final_score"] = round(max(0.0, min(1.0, adjusted_final)), 6)
            if copied.get("rejected_by_strict_factual_mode"):
                continue
            results.append(copied)

        # Fallback: if topical filtering was too strict and the query did not
        # contain a clear entity, return the best fused list.  Do not fall back to
        # off-entity evidence for strict factual questions such as "Who introduced
        # RAG?" or "RAG was introduced in 2020".
        if not results and not must_match_topic and not focus_terms:
            for item in fused.values():
                copied = dict(item)
                rrf = self._rrf(copied.get("dense_rank"), copied.get("sparse_rank"))
                copied["rrf_score"] = round(rrf, 6)
                copied["base_final_score"] = round(rrf, 6)
                copied["final_score"] = round(rrf, 6)
                copied["retrieval_method"] = "hybrid"
                copied = annotate_evidence(copied, query=query)
                if copied.get("rejected_by_strict_factual_mode"):
                    continue
                results.append(copied)

        results.sort(key=lambda x: rank_key(x, strict_factual_mode=strict_factual_mode), reverse=True)
        return results[:top_k]
