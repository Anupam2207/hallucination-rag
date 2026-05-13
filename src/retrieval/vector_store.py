import os
from pathlib import Path
from typing import Any, Dict, List

from src.paths import CHROMA_DIR


# Safe telemetry suppression for Chroma.
# Do NOT set CHROMA_PRODUCT_TELEMETRY_IMPL here; some Chroma versions do not have
# chromadb.telemetry.product.null.NullTelemetry and will crash during startup.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = "rag_knowledge",
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        self.persist_dir = str(Path(persist_dir or CHROMA_DIR).resolve())
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self.collection.count())

    def upsert_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError("ids, documents, embeddings, and metadatas must have the same length")

        if not ids:
            return

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: List[float], top_k: int = 4) -> Dict[str, Any]:
        collection_count = self.count()
        if collection_count == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]],
            }

        n_results = min(max(1, int(top_k)), collection_count)

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )