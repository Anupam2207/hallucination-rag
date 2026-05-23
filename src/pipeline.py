from typing import Any, Dict, List

from src.config import get_config_value
from src.detection.detector import HallucinationDetector
from src.detection.support_scorer import SupportScorer
from src.evaluation.metrics import HallucinationMetrics
from src.generation.base_answer import BaseAnswerGenerator
from src.generation.correction import AnswerCorrector
from src.generation.ollama_client import OllamaClient, OllamaServiceError
from src.logger import get_logger
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import SemanticRetriever


class HallucinationRAGPipeline:
    """End-to-end pipeline for retrieval-grounded hallucination detection."""

    def __init__(self, retrieval_mode: str | None = None) -> None:
        self.logger = get_logger("pipeline")
        shared_embedder = EmbeddingModel()
        shared_ollama_client = OllamaClient()

        mode = (retrieval_mode or get_config_value("settings", "retrieval", "mode", default="dense")).lower()
        if mode == "hybrid":
            self.retriever = HybridRetriever(embedder=shared_embedder)
        else:
            self.retriever = SemanticRetriever(embedder=shared_embedder)

        self.generator = BaseAnswerGenerator(client=shared_ollama_client)
        self.corrector = AnswerCorrector(client=shared_ollama_client)
        self.detector = HallucinationDetector(support_scorer=SupportScorer(embedder=shared_embedder))
        self.retrieval_mode = mode
        self.answerability_threshold = float(
            get_config_value("settings", "retrieval", "answerability_threshold", default=0.45)
        )

    @staticmethod
    def _evidence_strength(item: Dict[str, Any]) -> float:
        for key in ("dense_similarity", "similarity"):
            value = item.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        # RRF scores are small; only use them as a weak fallback.
        value = item.get("rrf_score")
        try:
            return min(1.0, float(value) * 30.0) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _is_answerable(cls, evidence: List[Dict[str, Any]], threshold: float) -> bool:
        if not evidence:
            return False
        return max(cls._evidence_strength(item) for item in evidence) >= threshold

    @staticmethod
    def _add_evidence_ids(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for index, item in enumerate(evidence, start=1):
            copied = dict(item)
            copied["evidence_id"] = f"Evidence-{index}"
            output.append(copied)
        return output

    @staticmethod
    def _empty_detection() -> Dict[str, Any]:
        return {
            "claims": [],
            "claim_count": 0,
            "supported_count": 0,
            "weak_count": 0,
            "unsupported_count": 0,
            "support_ratio": 0.0,
            "hallucination_rate": 0.0,
            "average_support_score": 0.0,
        }

    def run(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        warnings: List[str] = []
        evidence = self._add_evidence_ids(self.retriever.retrieve(query, top_k=top_k))
        if not evidence:
            warnings.append("No evidence was retrieved. Build the index or expand the knowledge base for better results.")

        if not self._is_answerable(evidence, self.answerability_threshold):
            warnings.append("insufficient_evidence")
            insufficient = "Insufficient evidence available in the knowledge base."
            empty_detection = self._empty_detection()
            metrics = HallucinationMetrics.summarize(empty_detection, empty_detection)
            return {
                "query": query,
                "retrieval_mode": self.retrieval_mode,
                "answerable": False,
                "warnings": warnings,
                "evidence": evidence,
                "raw_answer": insufficient,
                "raw_detection": empty_detection,
                "corrected_answer": insufficient,
                "corrected_detection": empty_detection,
                "metrics": metrics,
            }

        try:
            raw_answer = self.generator.generate_answer(query)
        except OllamaServiceError as exc:
            raise RuntimeError(str(exc)) from exc

        raw_detection = self.detector.detect(raw_answer, evidence)
        corrected_answer = self.corrector.correct(query, raw_answer, evidence)
        corrected_detection = self.detector.detect(corrected_answer, evidence)
        metrics = HallucinationMetrics.summarize(raw_detection, corrected_detection)

        return {
            "query": query,
            "retrieval_mode": self.retrieval_mode,
            "answerable": True,
            "warnings": warnings,
            "evidence": evidence,
            "raw_answer": raw_answer,
            "raw_detection": raw_detection,
            "corrected_answer": corrected_answer,
            "corrected_detection": corrected_detection,
            "metrics": metrics,
        }
