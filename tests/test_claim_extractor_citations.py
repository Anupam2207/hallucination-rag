from src.detection.claim_extractor import ClaimExtractor


def test_claim_extractor_does_not_keep_citation_tokens() -> None:
    claims = ClaimExtractor().extract_claims("ColBERT is a retrieval model [Evidence-1, Evidence-2].")
    assert claims == ["ColBERT is a retrieval model."]
    assert all("Evidence" not in claim for claim in claims)


def test_claim_extractor_cleans_numbered_list_fragments() -> None:
    text = """By integrating BERT with collaborative filtering, ColBERT can:
1. Improve recommendation accuracy
2. Enhance text-based search results
"""
    claims = ClaimExtractor().extract_claims(text)
    assert all("can:" not in claim.lower() for claim in claims)
    assert all(not claim.strip().endswith(("1.", "2.", "3.")) for claim in claims)
