from src.detection.detector import HallucinationDetector


class DummyScorer:
    def lexical_overlap_score(self, claim, evidence):
        return 1.0


def test_nli_neutral_does_not_force_high_similarity_claim_to_weak() -> None:
    detector = HallucinationDetector(support_scorer=DummyScorer())
    label, score, flags, adjusted, reason, composite = detector._fuse_decision(
        score=0.88,
        raw_similarity_score=0.88,
        similarity_label="supported",
        rule_flags=[],
        nli_result={"label": "neutral", "score": 0.99, "scores": {"neutral": 0.99}},
        claim="RAG combines retrieval and generation.",
        evidence="RAG combines retrieval and generation.",
    )
    assert label == "supported"
    assert score >= 0.7
    assert "nli_neutral" in flags
