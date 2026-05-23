from src.detection.claim_extractor import ClaimExtractor


def test_non_factual_assistant_phrases_are_skipped() -> None:
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(
        "I couldn't find any information on RAG being introduced in 2021. "
        "Could you please provide more context? "
        "RAG combines retrieval and generation."
    )
    assert claims == ["RAG combines retrieval and generation."]


def test_citation_markers_do_not_split_claims() -> None:
    extractor = ClaimExtractor()
    claims = extractor.extract_claims("RAG combines retrieval and generation [Evidence-1].")
    assert claims == ["RAG combines retrieval and generation."]
