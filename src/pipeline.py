from typing import Any, Dict, List

from src.detection.detector import HallucinationDetector
from src.detection.support_scorer import SupportScorer
from src.evaluation.metrics import HallucinationMetrics
from src.generation.base_answer import BaseAnswerGenerator
from src.generation.correction import AnswerCorrector
from src.generation.ollama_client import OllamaClient, OllamaServiceError
from src.logger import get_logger
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.retriever import SemanticRetriever


class HallucinationRAGPipeline:
    def __init__(self) -> None:
        self.logger = get_logger('pipeline')
        shared_embedder = EmbeddingModel()
        shared_ollama_client = OllamaClient()
        self.retriever = SemanticRetriever(embedder=shared_embedder)
        self.generator = BaseAnswerGenerator(client=shared_ollama_client)
        self.corrector = AnswerCorrector(client=shared_ollama_client)
        self.detector = HallucinationDetector(support_scorer=SupportScorer(embedder=shared_embedder))

    def run(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError('Query must not be empty.')

        warnings: List[str] = []
        evidence = self.retriever.retrieve(query, top_k=top_k)
        if not evidence:
            warnings.append(
                'No evidence was retrieved. Build the index or expand the knowledge base for better results.'
            )

        try:
            raw_answer = self.generator.generate_answer(query)
            corrected_answer = self.corrector.correct(query, raw_answer, evidence)
        except OllamaServiceError as exc:
            raise RuntimeError(str(exc)) from exc

        raw_detection = self.detector.detect(raw_answer, evidence)
        corrected_detection = self.detector.detect(corrected_answer, evidence)
        metrics = HallucinationMetrics.summarize(raw_detection, corrected_detection)

        return {
            'query': query,
            'warnings': warnings,
            'evidence': evidence,
            'raw_answer': raw_answer,
            'raw_detection': raw_detection,
            'corrected_answer': corrected_answer,
            'corrected_detection': corrected_detection,
            'metrics': metrics,
        }
