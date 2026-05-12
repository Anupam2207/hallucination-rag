import numpy as np

from src.detection.support_scorer import SupportScorer


class FakeEmbedder:
    def encode(self, texts, normalize=True):
        mapping = {
            "claim": np.array([1.0, 0.0]),
            "good evidence": np.array([1.0, 0.0]),
            "bad evidence": np.array([0.0, 1.0]),
        }
        vectors = []
        for text in texts:
            vectors.append(mapping[text])
        return np.array(vectors)


def test_support_scorer_selects_best_evidence() -> None:
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        "claim",
        [{"text": "bad evidence"}, {"text": "good evidence"}],
    )
    assert result["best_evidence"] == "good evidence"
    assert result["score"] > 0.99
