from src.detection.claim_extractor import ClaimExtractor


def test_claim_extractor_splits_sentences() -> None:
    extractor = ClaimExtractor()
    claims = extractor.extract_claims('RAG uses retrieval. It can improve factual grounding.')
    assert len(claims) == 2
