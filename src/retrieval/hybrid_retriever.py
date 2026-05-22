from typing import Any, Dict, List

from src.config import get_config_value
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.sparse_retriever import BM25SparseRetriever
from src.retrieval.vector_store import ChromaVectorStore


class HybridRetriever:
    """Combine dense Chroma results and sparse BM25 results with RRF."""

    def __init__(
        self,
        dense_retriever: SemanticRetriever | None = None,
        sparse_retriever: BM25SparseRetriever | None = None,
        vector_store: ChromaVectorStore | None = None,
        embedder: EmbeddingModel | None = None,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self.top_k = int(top_k if top_k is not None else get_config_value('settings', 'retrieval', 'top_k', default=4))
        self.dense_top_k = int(dense_top_k if dense_top_k is not None else get_config_value('settings', 'retrieval', 'dense_top_k', default=max(8, self.top_k)))
        self.sparse_top_k = int(sparse_top_k if sparse_top_k is not None else get_config_value('settings', 'retrieval', 'sparse_top_k', default=max(8, self.top_k)))
        self.rrf_k = int(rrf_k if rrf_k is not None else get_config_value('settings', 'retrieval', 'rrf_k', default=60))
        self.dense_retriever = dense_retriever or SemanticRetriever(vector_store=vector_store, embedder=embedder, top_k=self.dense_top_k)
        self.sparse_retriever = sparse_retriever or BM25SparseRetriever()

    def is_ready(self) -> bool:
        return self.dense_retriever.is_ready() or self.sparse_retriever.is_ready()

    @staticmethod
    def _stable_chunk_id(item: Dict[str, Any], fallback_prefix: str, fallback_index: int) -> str:
        chunk_id = item.get('chunk_id')
        if chunk_id:
            return str(chunk_id)
        metadata = item.get('metadata') or {}
        if metadata.get('doc_id') is not None and metadata.get('chunk_index') is not None:
            return f"{metadata['doc_id']}_chunk_{metadata['chunk_index']}"
        return f'{fallback_prefix}_{fallback_index}'

    def _rrf_score(self, dense_rank: int | None, sparse_rank: int | None) -> float:
        score = 0.0
        if dense_rank is not None:
            score += 1.0 / (self.rrf_k + dense_rank)
        if sparse_rank is not None:
            score += 1.0 / (self.rrf_k + sparse_rank)
        return score

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        requested_top_k = int(top_k or self.top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=max(self.dense_top_k, requested_top_k))
        try:
            sparse_results = self.sparse_retriever.retrieve(query, top_k=max(self.sparse_top_k, requested_top_k))
        except Exception:
            sparse_results = []

        if not sparse_results:
            # Safe fallback: preserve dense retrieval if BM25 artifacts are missing.
            fallback = []
            for rank, item in enumerate(dense_results[:requested_top_k], start=1):
                copied = dict(item)
                copied.setdefault('dense_rank', rank)
                copied.setdefault('dense_similarity', copied.get('similarity'))
                copied.setdefault('sparse_rank', None)
                copied.setdefault('sparse_score', None)
                copied.setdefault('rrf_score', round(self._rrf_score(rank, None), 6))
                copied['retrieval_method'] = 'dense_fallback'
                fallback.append(copied)
            return fallback

        fused: dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, start=1):
            chunk_id = self._stable_chunk_id(item, 'dense', rank)
            entry = fused.setdefault(
                chunk_id,
                {
                    'chunk_id': chunk_id,
                    'text': item.get('text', ''),
                    'metadata': item.get('metadata') or {},
                    'dense_rank': None,
                    'sparse_rank': None,
                    'dense_similarity': None,
                    'sparse_score': None,
                },
            )
            entry['text'] = entry.get('text') or item.get('text', '')
            entry['metadata'] = entry.get('metadata') or item.get('metadata') or {}
            entry['dense_rank'] = rank
            entry['dense_similarity'] = item.get('dense_similarity', item.get('similarity'))
            entry['distance'] = item.get('distance')
            entry['similarity'] = item.get('similarity')

        for rank, item in enumerate(sparse_results, start=1):
            chunk_id = self._stable_chunk_id(item, 'sparse', rank)
            entry = fused.setdefault(
                chunk_id,
                {
                    'chunk_id': chunk_id,
                    'text': item.get('text', ''),
                    'metadata': item.get('metadata') or {},
                    'dense_rank': None,
                    'sparse_rank': None,
                    'dense_similarity': None,
                    'sparse_score': None,
                },
            )
            entry['text'] = entry.get('text') or item.get('text', '')
            entry['metadata'] = entry.get('metadata') or item.get('metadata') or {}
            entry['sparse_rank'] = rank
            entry['sparse_score'] = item.get('sparse_score')

        ranked: List[Dict[str, Any]] = []
        for entry in fused.values():
            dense_rank = entry.get('dense_rank')
            sparse_rank = entry.get('sparse_rank')
            entry['rrf_score'] = round(self._rrf_score(dense_rank, sparse_rank), 6)
            entry['retrieval_method'] = 'hybrid'
            if entry.get('similarity') is None and entry.get('dense_similarity') is not None:
                entry['similarity'] = entry['dense_similarity']
            ranked.append(entry)

        ranked.sort(key=lambda item: item.get('rrf_score', 0.0), reverse=True)
        return ranked[:requested_top_k]


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """Pure helper used by unit tests and documentation examples."""
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.rrf_k = rrf_k
    retriever._rrf_score = HybridRetriever._rrf_score.__get__(retriever, HybridRetriever)
    retriever._stable_chunk_id = HybridRetriever._stable_chunk_id
    fused: dict[str, Dict[str, Any]] = {}
    for rank, item in enumerate(dense_results, start=1):
        cid = item.get('chunk_id') or f'dense_{rank}'
        fused.setdefault(cid, {'chunk_id': cid, 'dense_rank': None, 'sparse_rank': None})
        fused[cid].update(item)
        fused[cid]['dense_rank'] = rank
    for rank, item in enumerate(sparse_results, start=1):
        cid = item.get('chunk_id') or f'sparse_{rank}'
        fused.setdefault(cid, {'chunk_id': cid, 'dense_rank': None, 'sparse_rank': None})
        # Do not overwrite dense text/metadata if present; just add sparse info.
        for key, value in item.items():
            fused[cid].setdefault(key, value)
        fused[cid]['sparse_rank'] = rank
        fused[cid]['sparse_score'] = item.get('sparse_score')
    rows = []
    for item in fused.values():
        item['rrf_score'] = round(retriever._rrf_score(item.get('dense_rank'), item.get('sparse_rank')), 6)
        item['retrieval_method'] = 'hybrid'
        rows.append(item)
    rows.sort(key=lambda x: x['rrf_score'], reverse=True)
    return rows[:top_k]
