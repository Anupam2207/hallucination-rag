from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.embedder import EmbeddingModel


class SupportScorer:
    def __init__(self, embedder: EmbeddingModel | None = None) -> None:
        self.embedder = embedder or EmbeddingModel()

    @staticmethod
    def _extract_texts(evidence_list: Iterable[Any]) -> List[str]:
        texts: List[str] = []
        for item in evidence_list:
            if isinstance(item, str):
                text = item.strip()
            else:
                text = str(item.get('text', '')).strip()
            if text:
                texts.append(text)
        return texts

    def score_claim(self, claim: str, evidence_list: Iterable[Any]) -> Tuple[float, int | None]:
        result = self.score_claim_against_evidence(claim, evidence_list)
        return float(result['score']), result['best_evidence_index']

    def score_claim_against_evidence(self, claim: str, evidence_list: Iterable[Any]) -> Dict[str, Any]:
        evidence_texts = self._extract_texts(evidence_list)
        if not claim.strip() or not evidence_texts:
            return {
                'score': 0.0,
                'best_evidence': None,
                'best_evidence_index': None,
            }

        claim_vector = self.embedder.encode([claim])
        evidence_vectors = self.embedder.encode(evidence_texts)
        similarities = cosine_similarity(claim_vector, evidence_vectors)[0]
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])
        return {
            'score': best_score,
            'best_evidence': evidence_texts[best_index],
            'best_evidence_index': best_index,
        }
