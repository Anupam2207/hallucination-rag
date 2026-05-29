import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.paths import CHROMA_DIR


# Disable Chroma telemetry by default so queries remain privacy-preserving.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")


class ChromaVectorStore:
    """Vector store wrapper with an offline JSONL fallback.

    The public interface mirrors the original Chroma-backed store.  When
    ``chromadb`` is installed, Chroma remains the backend.  On low-resource or
    offline laptops where Chroma is unavailable, the class persists a compact
    JSONL index under ``db/chroma`` and performs cosine search in NumPy.  This
    keeps the demo commands runnable without changing callers.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = "rag_knowledge",
    ) -> None:
        self.persist_dir = str(Path(persist_dir or CHROMA_DIR).resolve())
        self.collection_name = collection_name
        self.backend = "chromadb"
        self.client = None
        self.collection = None
        self.index_path = Path(self.persist_dir) / f"{self.collection_name}.jsonl"
        self._records: list[dict[str, Any]] = []

        try:
            import chromadb
            from chromadb.config import Settings

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
        except Exception:
            self.backend = "jsonl"
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._records = self._load_jsonl_records()

    def _load_jsonl_records(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _write_jsonl_records(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def reset_collection(self) -> None:
        if self.backend == "chromadb" and self.client is not None:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return

        self._records = []
        try:
            self.index_path.unlink()
        except FileNotFoundError:
            pass

    def count(self) -> int:
        if self.backend == "chromadb" and self.collection is not None:
            return int(self.collection.count())
        return len(self._records)

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

        if self.backend == "chromadb" and self.collection is not None:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            return

        by_id = {str(record.get("id")): dict(record) for record in self._records}
        for doc_id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            by_id[str(doc_id)] = {
                "id": str(doc_id),
                "document": document,
                "embedding": [float(value) for value in embedding],
                "metadata": metadata or {},
            }
        self._records = list(by_id.values())
        self._write_jsonl_records()

    @staticmethod
    def _cosine_distance(query: np.ndarray, vector: np.ndarray) -> float:
        if query.size == 0 or vector.size == 0 or query.shape != vector.shape:
            return 1.0
        denom = float(np.linalg.norm(query) * np.linalg.norm(vector))
        if denom == 0.0:
            return 1.0
        similarity = float(np.dot(query, vector) / denom)
        return max(0.0, min(2.0, 1.0 - similarity))

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

        if self.backend == "chromadb" and self.collection is not None:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

        query = np.asarray(query_embedding, dtype=np.float32)
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in self._records:
            vector = np.asarray(record.get("embedding", []), dtype=np.float32)
            distance = self._cosine_distance(query, vector)
            scored.append((distance, record))
        scored.sort(key=lambda item: item[0])
        selected = scored[:n_results]
        return {
            "documents": [[record.get("document", "") for _distance, record in selected]],
            "metadatas": [[record.get("metadata", {}) for _distance, record in selected]],
            "distances": [[round(float(distance), 6) for distance, _record in selected]],
            "ids": [[record.get("id") for _distance, record in selected]],
        }
