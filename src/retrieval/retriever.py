from typing import Any, Dict, List

from src.config import load_all_configs
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore


class SemanticRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        embedder: EmbeddingModel | None = None,
        top_k: int | None = None,
    ) -> None:
        configs = load_all_configs()
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or EmbeddingModel()
        self.top_k = top_k or configs["settings"]["retrieval"]["top_k"]

    def is_ready(self) -> bool:
        return self.vector_store.count() > 0

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if not query.strip() or not self.is_ready():
            return []

        query_embedding = self.embedder.encode([query])[0]
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()
        raw_results = self.vector_store.query(query_embedding, top_k=top_k or self.top_k)

        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        ids = raw_results.get("ids", [[]])[0] if raw_results.get("ids") else [None] * len(documents)

        normalized_results: List[Dict[str, Any]] = []
        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            distance_value = float(distance) if distance is not None else None
            similarity = None if distance_value is None else max(0.0, 1.0 - distance_value)
            normalized_results.append(
                {
                    "chunk_id": doc_id,
                    "text": document,
                    "metadata": metadata or {},
                    "distance": distance_value,
                    "similarity": similarity,
                }
            )
        return normalized_results
