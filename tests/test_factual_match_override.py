import numpy as np

from src.detection.detector import HallucinationDetector
from src.detection.support_scorer import SupportScorer


class FakeEmbedder:
    def encode(self, texts, normalize=True):
        return np.array([[1.0, 0.0] for _ in texts])


class NeutralNLI:
    enabled = True
    def verify(self, claim, evidence):
        return {"label": "neutral", "score": 0.99, "scores": {"neutral": 0.99}, "available": True, "error": None}


def test_factual_exact_match_overrides_nli_neutral():
    detector = HallucinationDetector(
        support_scorer=SupportScorer(embedder=FakeEmbedder()),
        nli_verifier=NeutralNLI(),
    )
    claim = "It was introduced by Omar Khattab and Matei Zaharia in 2020 as a retrieval architecture that combines contextual language representations with scalable search."
    evidence = [{"text": "ColBERT was introduced by Omar Khattab and Matei Zaharia in 2020 as a retrieval architecture that combines contextual language representations with scalable search."}]
    result = detector.detect(claim, evidence)
    assert result["claims"][0]["label"] == "supported"
    assert result["claims"][0]["support_score"] >= 0.72
