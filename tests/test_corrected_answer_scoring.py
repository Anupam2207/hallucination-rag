import numpy as np
from src.detection.support_scorer import SupportScorer


class FakeEmbedder:
    def encode(self, texts, normalize=True):
        return np.array([[1.0, 0.0] for _ in texts])


def test_corrected_claim_can_use_combined_evidence_context() -> None:
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        "RAG combines information retrieval with language generation and improves factual accuracy.",
        [
            {"text": "RAG combines information retrieval with language generation."},
            {"text": "RAG improves factual accuracy by retrieving relevant external knowledge."},
        ],
    )
    assert result["score"] >= 0.7
    assert result["combined_context_score"] > 0.0
    assert result["rule_flags"] == []


def test_citation_does_not_reduce_support_score() -> None:
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        "ColBERT stands for Contextualized Late Interaction over BERT [Evidence-1].",
        [{"text": "ColBERT stands for Contextualized Late Interaction over BERT."}],
    )
    assert result["score"] >= 0.7
    assert result["rule_flags"] == []
