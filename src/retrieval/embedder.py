from functools import lru_cache
from typing import Iterable

import numpy as np

from src.config import load_all_configs


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load the sentence-transformers model and cache it for reuse."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - depends on local model cache/network
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}'. Ensure the model can be downloaded or is cached locally. "
            f"Original error: {exc}"
        ) from exc


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        configs = load_all_configs()
        resolved_name = model_name or configs['models']['embedding']['model_name']
        self.model_name = resolved_name
        self.model = _load_model(resolved_name)

    def encode(self, texts: str | Iterable[str], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        # Use the sentence transformer to create normalized dense embeddings.
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
