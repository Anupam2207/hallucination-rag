from typing import Any, Dict, List

from src.config import load_all_configs
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore


class SemanticRetriever:
    """Dense semantic retriever backed by ChromaDB.

    This class remains available for backward compatibility and for cases where
    configs/settings.yaml sets retrieval.mode = "dense".
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        embedder: EmbeddingModel | None = None,
        top_k: int | None = None,
    ) -> None:
        configs = load_all_configs()
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or EmbeddingModel()
        self.top_k = int(top_k or configs["settings"]["retrieval"]["top_k"])

    def is_ready(self) -> bool:
        return self.vector_store.count() > 0

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if not query.strip() or not self.is_ready():
            return []

        # Convert the query into a dense embedding for semantic search.
        query_embedding = self.embedder.encode([query])[0]
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        raw_results = self.vector_store.query(query_embedding, top_k=top_k or self.top_k)

        # Normalize Chroma query output into a consistent list of evidence items.
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        ids = raw_results.get("ids", [[]])[0] if raw_results.get("ids") else [None] * len(documents)

        normalized_results: List[Dict[str, Any]] = []
        for rank, (doc_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            distance_value = float(distance) if distance is not None else None
            similarity = None if distance_value is None else max(0.0, 1.0 - distance_value)
            normalized_results.append(
                {
                    "chunk_id": doc_id,
                    "text": document,
                    "metadata": metadata or {},
                    "distance": distance_value,
                    "similarity": similarity,
                    "dense_rank": rank,
                    "dense_similarity": similarity,
                    "retrieval_method": "dense",
                }
            )
        return normalized_results
