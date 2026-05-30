import re
from typing import Any, Dict, List

from src.config import get_config_value
from src.retrieval.query_focus import (
    definition_score,
    extract_query_focus,
    has_competing_topic,
    has_topical_match,
    is_pdf_source,
    source_quality,
    topical_score,
)
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.sparse_retriever import BM25SparseRetriever


_STOPWORDS = {
    "what", "who", "when", "where", "why", "how", "is", "are", "was", "were", "be",
    "the", "a", "an", "of", "to", "in", "on", "for", "with", "by", "and", "or", "from",
    "about", "explain", "define", "describe", "tell", "me", "please", "introduced", "invented",
    "created", "proposed", "released", "published", "launched", "developed", "began", "started",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-]{1,}", (text or "").lower())


def extract_focus_terms(query: str) -> list[str]:
    focus = extract_query_focus(query)
    terms = list(focus.get("core_entity_terms") or [])
    if terms:
        return terms
    output: list[str] = []
    for token in _tokens(query):
        if token in _STOPWORDS:
            continue
        if len(token) < 3 and token not in {"ai", "qa"}:
            continue
        output.append(token)
    return list(dict.fromkeys(output))


def expand_query(query: str) -> str:
    """Small query expansion that keeps the target entity dominant."""
    q = (query or "").lower()
    additions: list[str] = []
    if any(p in q for p in ("who introduced", "who proposed", "who invented", "who developed", "who created")):
        additions.extend(["author", "authors", "paper", "proposed", "introduced", "developed"])
    if "which year" in q or "when" in q:
        additions.extend(["year", "published", "introduced"])
    if "what is" in q or "explain" in q or "define" in q:
        additions.extend(["definition", "overview", "is a", "is an"])
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
    """Hybrid dense + sparse retriever with PDF-only, entity-aware reranking.

    Dense retrieval finds paraphrases, BM25 catches exact terms, and the final
    ranking explicitly rewards PDF source quality and query-topic coverage. This
    prevents generic chunks from being accepted merely because they are broadly
    similar to the query.
    """

    def __init__(self, dense_retriever=None, sparse_retriever=None, vector_store=None, embedder=None):
        self.top_k = int(get_config_value("settings", "retrieval", "top_k", default=4))
        self.dense_top_k = int(get_config_value("settings", "retrieval", "dense_top_k", default=12))
        self.sparse_top_k = int(get_config_value("settings", "retrieval", "sparse_top_k", default=12))
        self.rrf_k = int(get_config_value("settings", "retrieval", "rrf_k", default=60))
        self.dense_weight = float(get_config_value("settings", "retrieval", "dense_weight", default=0.40))
        self.sparse_weight = float(get_config_value("settings", "retrieval", "sparse_weight", default=0.30))
        self.importance_weight = float(get_config_value("settings", "retrieval", "importance_weight", default=0.08))
        self.topical_weight = float(get_config_value("settings", "retrieval", "topical_weight", default=0.17))
        self.source_weight = float(get_config_value("settings", "retrieval", "source_weight", default=0.05))
        self.content_weight = float(get_config_value("settings", "retrieval", "content_weight", default=0.78))
        self.rrf_weight = float(get_config_value("settings", "retrieval", "rrf_weight", default=0.22))
        self.min_topical_score = float(get_config_value("settings", "retrieval", "min_topical_score", default=0.10))
        self.pdf_only = bool(get_config_value("settings", "knowledge_base", "pdf_only", default=True))

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
    def _source_title_bonus(item: Dict[str, Any], focus: Dict[str, Any]) -> float:
        metadata = item.get("metadata", {}) or {}
        title_text = " ".join(str(metadata.get(k, "") or "") for k in ("file_name", "document_title", "source_rel"))
        if has_topical_match(title_text, metadata, focus):
            return 0.08
        return 0.0

    def _should_keep(self, item: Dict[str, Any], focus: Dict[str, Any], topical: float) -> tuple[bool, str]:
        metadata = item.get("metadata", {}) or {}
        text = str(item.get("text", ""))
        if self.pdf_only and not is_pdf_source(metadata):
            return False, "non_pdf_source_filtered"
        if has_competing_topic(text, metadata, focus):
            return False, "competing_topic_filtered"
        if not has_topical_match(text, metadata, focus):
            return False, "topic_mismatch"
        if topical < self.min_topical_score:
            return False, "low_topical_score"
        return True, ""

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        top_k = int(top_k or self.top_k)
        expanded = expand_query(query)
        focus = extract_query_focus(query)

        # Pull a little deeper than final top_k so filters do not accidentally
        # remove the only good PDF chunk.
        dense = self.dense_retriever.retrieve(expanded, top_k=max(self.dense_top_k, top_k * 3))
        sparse = self.sparse_retriever.retrieve(expanded, top_k=max(self.sparse_top_k, top_k * 3))

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
            fused[cid].setdefault("text", item.get("text", ""))
            fused[cid].setdefault("metadata", item.get("metadata", {}))

        max_sparse = max((float(x.get("sparse_score") or 0.0) for x in fused.values()), default=1.0) or 1.0
        results: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for item in fused.values():
            metadata = item.get("metadata", {}) or {}
            text = str(item.get("text", ""))
            topical = topical_score(text, metadata, focus)
            keep, reason = self._should_keep(item, focus, topical)
            if not keep:
                copied_reject = dict(item)
                copied_reject["topical_score"] = round(topical, 6)
                copied_reject["rejected_by_retrieval_filter"] = True
                copied_reject["rejection_reason"] = reason
                rejected.append(copied_reject)
                continue

            dense_score = float(item.get("dense_similarity") or item.get("similarity") or 0.0)
            sparse_score_raw = float(item.get("sparse_score") or 0.0)
            sparse_score = sparse_score_raw / max_sparse
            importance = importance_norm(metadata.get("importance_score", 0))
            source = source_quality(metadata)
            definition = definition_score(query, text, metadata)
            rrf = self._rrf(item.get("dense_rank"), item.get("sparse_rank"))
            rrf_norm = min(1.0, rrf * 30.0)
            title_bonus = self._source_title_bonus(item, focus)

            weighted = (
                self.dense_weight * dense_score
                + self.sparse_weight * sparse_score
                + self.importance_weight * importance
                + self.topical_weight * topical
                + self.source_weight * source
                + 0.05 * definition
                + title_bonus
            )
            final = self.content_weight * weighted + self.rrf_weight * rrf_norm

            copied = dict(item)
            copied["dense_similarity"] = round(dense_score, 6) if dense_score else item.get("dense_similarity")
            copied["sparse_score"] = round(sparse_score_raw, 6) if sparse_score_raw else item.get("sparse_score")
            copied["normalized_sparse_score"] = round(sparse_score, 6)
            copied["importance_score"] = metadata.get("importance_score", 0)
            copied["importance_norm"] = round(importance, 6)
            copied["topical_score"] = round(topical, 6)
            copied["source_quality"] = round(source, 6)
            copied["definition_score"] = round(definition, 6)
            copied["weighted_score"] = round(weighted, 6)
            copied["rrf_score"] = round(rrf, 6)
            copied["final_score"] = round(max(0.0, min(1.0, final)), 6)
            copied["retrieval_method"] = "hybrid"
            copied["rejected_by_retrieval_filter"] = False
            copied["rejection_reason"] = ""
            results.append(copied)

        # Controlled fallback: if every result was filtered, return the best PDF
        # candidates with a visible warning. This prevents the UI from saying
        # "no evidence" when the index does contain a low-scoring but usable PDF.
        if not results:
            pdf_candidates = [item for item in rejected if not self.pdf_only or is_pdf_source((item.get("metadata") or {}))]
            for item in pdf_candidates[: max(top_k, 1)]:
                copied = dict(item)
                rrf = self._rrf(copied.get("dense_rank"), copied.get("sparse_rank"))
                copied["rrf_score"] = round(rrf, 6)
                copied["final_score"] = round(max(float(copied.get("topical_score") or 0.0), min(1.0, rrf * 20.0)), 6)
                copied["retrieval_method"] = "hybrid"
                copied["retrieval_fallback"] = True
                results.append(copied)

        results.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
        return results[:top_k]
