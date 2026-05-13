import re
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.embedder import EmbeddingModel


class SupportScorer:
    def __init__(self, embedder: EmbeddingModel | None = None) -> None:
        self.embedder = embedder or EmbeddingModel()

    @staticmethod
    def _extract_text(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return str(item.get("text", ""))
        return str(item)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "and", "or", "of", "to", "in", "on", "for", "with", "by", "as", "at",
            "from", "that", "this", "it", "its", "into", "than", "then", "while",
            "using", "use", "used", "uses", "both", "these", "those", "can", "could",
            "would", "should", "typically",
        }
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
        return {word for word in words if word not in stopwords}

    def lexical_overlap_score(self, claim: str, evidence: str) -> float:
        claim_tokens = self._tokens(claim)
        evidence_tokens = self._tokens(evidence)
        if not claim_tokens or not evidence_tokens:
            return 0.0

        overlap = claim_tokens.intersection(evidence_tokens)
        if len(overlap) < 3:
            return 0.0

        recall = len(overlap) / len(claim_tokens)
        precision = len(overlap) / len(evidence_tokens)
        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)

    def score_claim(self, claim: str, evidence_list: list[Any]) -> tuple[float, int | None]:
        if not claim.strip() or not evidence_list:
            return 0.0, None

        evidence_texts = [self._extract_text(item) for item in evidence_list]
        evidence_texts = [text for text in evidence_texts if text.strip()]
        if not evidence_texts:
            return 0.0, None

        claim_embedding = self.embedder.encode([claim])
        evidence_embeddings = self.embedder.encode(evidence_texts)
        similarities = cosine_similarity(claim_embedding, evidence_embeddings)[0]

        best_score = 0.0
        best_index: int | None = None
        for index, semantic_score in enumerate(similarities):
            lexical_score = self.lexical_overlap_score(claim, evidence_texts[index])
            if lexical_score >= 0.60:
                hybrid_score = max(float(semantic_score), float(lexical_score))
            else:
                hybrid_score = float(semantic_score)

            if hybrid_score > best_score:
                best_score = hybrid_score
                best_index = index

        return best_score, best_index

    def score_claim_against_evidence(self, claim: str, evidence_list: list[Any]) -> dict[str, Any]:
        """Backward-compatible richer scoring API used by tests and reports."""
        score, best_index = self.score_claim(claim, evidence_list)
        best_evidence = None
        if best_index is not None and 0 <= best_index < len(evidence_list):
            best_evidence = self._extract_text(evidence_list[best_index])
        return {
            "claim": claim,
            "score": round(float(score), 4),
            "best_evidence_index": best_index,
            "best_evidence": best_evidence,
        }
