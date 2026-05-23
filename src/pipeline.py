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
    """End-to-end pipeline for hallucination detection and correction.

    Runtime order intentionally follows the project objective:
    1. Generate an initial ungrounded/raw LLM answer.
    2. Retrieve evidence from the local index.
    3. Detect unsupported claims in the raw answer.
    4. Correct the answer using the original answer + retrieved evidence.
    5. Detect support again and compute before/after metrics.
    """

    def __init__(self, retrieval_mode: str | None = None) -> None:
        self.logger = get_logger('pipeline')

        # Share a single embedder instance across retrieval and detection to avoid
        # re-loading models multiple times and to keep embeddings consistent.
        shared_embedder = EmbeddingModel()
        shared_ollama_client = OllamaClient()

        mode = (retrieval_mode or get_config_value('settings', 'retrieval', 'mode', default='dense')).lower()
        if mode == 'hybrid':
            self.retriever = HybridRetriever(embedder=shared_embedder)
        else:
            self.retriever = SemanticRetriever(embedder=shared_embedder)

        # The generator and corrector both use the Ollama client.
        self.generator = BaseAnswerGenerator(client=shared_ollama_client)
        self.corrector = AnswerCorrector(client=shared_ollama_client)

        # Detector uses similarity scoring plus optional rule-based adjustments.
        self.detector = HallucinationDetector(support_scorer=SupportScorer(embedder=shared_embedder))
        self.retrieval_mode = mode

    def run(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError('Query must not be empty.')

        warnings: List[str] = []

        try:
            raw_answer = self.generator.generate_answer(query)
        except OllamaServiceError as exc:
            raise RuntimeError(str(exc)) from exc

        # Retrieve evidence for the query from Chroma or hybrid retrieval.
        evidence = self.retriever.retrieve(query, top_k=top_k)
        if not evidence:
            warnings.append(
                'No evidence was retrieved. Build the index or expand the knowledge base for better results.'
            )

        # Use the original raw LLM answer along with retrieved evidence to create a
        # grounded corrected answer.
        corrected_answer = self.corrector.correct(query, raw_answer, evidence)

        # Compute claim-level detection results before and after correction.
        raw_detection = self.detector.detect(raw_answer, evidence)
        corrected_detection = self.detector.detect(corrected_answer, evidence)
        metrics = HallucinationMetrics.summarize(raw_detection, corrected_detection)

        return {
            'query': query,
            'retrieval_mode': self.retrieval_mode,
            'warnings': warnings,
            'evidence': evidence,
            'raw_answer': raw_answer,
            'raw_detection': raw_detection,
            'corrected_answer': corrected_answer,
            'corrected_detection': corrected_detection,
            'metrics': metrics,
        }
