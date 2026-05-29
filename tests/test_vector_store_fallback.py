from src.retrieval.vector_store import ChromaVectorStore


def test_jsonl_vector_store_fallback_round_trip(tmp_path, monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "chromadb" or name.startswith("chromadb."):
            raise ModuleNotFoundError("chromadb hidden for fallback test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    store = ChromaVectorStore(persist_dir=tmp_path, collection_name="test_collection")
    assert store.backend == "jsonl"
    store.upsert_documents(
        ids=["a", "b"],
        documents=["RAG combines retrieval and generation.", "ColBERT ranks passages."],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"source_rel": "a.txt"}, {"source_rel": "b.txt"}],
    )
    assert store.count() == 2
    results = store.query([1.0, 0.0], top_k=1)
    assert results["ids"] == [["a"]]
    assert results["documents"] == [["RAG combines retrieval and generation."]]
