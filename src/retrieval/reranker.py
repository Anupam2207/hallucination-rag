"""Optional cross-encoder reranker for research/evaluation mode."""

from __future__ import annotations

from typing import Any, Dict, List

from src.config import get_config_value


class CrossEncoderReranker:
    def __init__(self, enabled: bool | None = None, model_name: str | None = None, device: str | None = None) -> None:
        self.enabled = bool(
            get_config_value("settings", "retrieval", "enable_reranker", default=False)
            if enabled is None else enabled
        )
        self.model_name = model_name or str(
            get_config_value("settings", "retrieval", "reranker_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        self.device = device or str(get_config_value("settings", "retrieval", "reranker_device", default="cpu"))
        self.model = None
        self.available = False
        self.error: str | None = None
        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
            self.available = True
        except Exception as exc:  # pragma: no cover - model/cache/network dependent
            self.error = str(exc)
            self.available = False
            self.model = None

    def rerank(self, query: str, evidence: List[Dict[str, Any]], top_k: int | None = None) -> List[Dict[str, Any]]:
        if not self.enabled or not self.available or self.model is None or not evidence:
            return evidence[:top_k] if top_k else evidence
        pairs = [(query, item.get("text", "")) for item in evidence]
        try:
            scores = self.model.predict(pairs)
            ranked = []
            for item, score in zip(evidence, scores):
                copied = dict(item)
                copied["rerank_score"] = round(float(score), 4)
                ranked.append(copied)
            ranked.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
            return ranked[:top_k] if top_k else ranked
        except Exception as exc:  # pragma: no cover
            fallback = [dict(item) for item in evidence]
            for item in fallback:
                item["rerank_error"] = str(exc)
            return fallback[:top_k] if top_k else fallback
