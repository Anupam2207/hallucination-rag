from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

from src.config import get_config_value, load_all_configs


class HashEmbeddingBackend:
    """Small deterministic embedding fallback for offline/low-memory demos.

    It is not a replacement for a sentence-transformer in research runs, but it
    keeps ingestion, Chroma indexing, tests, and demo scripts runnable when model
    weights are unavailable on a laptop.
    """

    def __init__(self, n_features: int = 384) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            ngram_range=(1, 2),
        )

    def encode(self, texts: list[str], normalize_embeddings: bool = True, **_: object) -> np.ndarray:
        matrix = self.vectorizer.transform(texts).astype(np.float32).toarray()
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms
        return matrix


@lru_cache(maxsize=3)
def _load_model(model_name: str, allow_hash_fallback: bool = True):
    """Load a sentence-transformers model and cache it for reuse.

    In demo mode the project falls back to a deterministic hashing embedder if
    the real model cannot be imported, downloaded, or loaded from cache.
    """
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - depends on local model cache/network
        if allow_hash_fallback:
            backend = HashEmbeddingBackend()
            backend.load_error = str(exc)  # type: ignore[attr-defined]
            return backend
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}'. Ensure the model can be downloaded or is cached locally. "
            f"Original error: {exc}"
        ) from exc


class EmbeddingModel:
    def __init__(self, model_name: str | None = None, allow_hash_fallback: bool | None = None) -> None:
        configs = load_all_configs()
        resolved_name = model_name or configs["models"]["embedding"]["model_name"]
        if allow_hash_fallback is None:
            allow_hash_fallback = bool(
                get_config_value("settings", "runtime", "allow_embedding_hash_fallback", default=True)
            )
        backend_preference = str(
            get_config_value("settings", "runtime", "embedding_backend", default="auto")
        ).strip().lower()
        self.model_name = resolved_name
        self.allow_hash_fallback = bool(allow_hash_fallback)
        if backend_preference in {"hash", "hashing", "hashingvectorizer"}:
            self.model = HashEmbeddingBackend()
            self.backend = "hashing"
            self.load_error = None
            return
        self.model = _load_model(resolved_name, self.allow_hash_fallback)
        self.backend = "hashing" if isinstance(self.model, HashEmbeddingBackend) else "sentence_transformers"
        self.load_error = getattr(self.model, "load_error", None)

    def encode(self, texts: str | Iterable[str], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
