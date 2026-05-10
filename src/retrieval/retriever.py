from src.retrieval.embedder import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore


class SemanticRetriever:

    def __init__(self):

        self.embedder = EmbeddingModel()

        self.vector_store = ChromaVectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ):

        query_embedding = self.embedder.encode(
            query
        )[0]

        results = self.vector_store.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k
        )

        return results