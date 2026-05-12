from src.retrieval.retriever import SemanticRetriever


class FakeEmbedder:
    def encode(self, texts, normalize=True):
        return [[0.1, 0.2, 0.3]]


class FakeStore:
    def count(self):
        return 1

    def query(self, query_embedding, top_k=4):
        return {
            "documents": [["retrieved evidence"]],
            "metadatas": [[{"source_rel": "demo.txt"}]],
            "distances": [[0.2]],
            "ids": [["chunk_1"]],
        }


def test_retriever_returns_normalized_results() -> None:
    retriever = SemanticRetriever(vector_store=FakeStore(), embedder=FakeEmbedder(), top_k=1)
    results = retriever.retrieve("test query")
    assert len(results) == 1
    assert results[0]["text"] == "retrieved evidence"
    assert results[0]["similarity"] == 0.8
