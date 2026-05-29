from src.detection.claim_extractor import ClaimExtractor


def test_multi_claim_sentence_is_split_atomically():
    claims = ClaimExtractor(atomic_splitting=True).extract_claims(
        "RAG retrieves relevant documents and generates grounded answers."
    )
    assert "RAG retrieves relevant documents." in claims
    assert "RAG generates grounded answers." in claims


def test_bullet_answer_extracts_each_factual_item():
    claims = ClaimExtractor(atomic_splitting=True).extract_claims(
        "- RAG retrieves evidence.\n- It uses the evidence as context.\n3. ColBERT ranks passages."
    )
    assert claims == [
        "RAG retrieves evidence.",
        "It uses the evidence as context.",
        "ColBERT ranks passages.",
    ]


def test_answer_with_citations_removes_evidence_markers_not_facts():
    claims = ClaimExtractor(atomic_splitting=True).extract_claims(
        "RAG combines retrieval and generation [Evidence-1]. ColBERT was introduced in 2020 [Evidence-2]."
    )
    assert "RAG combines retrieval and generation." in claims
    assert "ColBERT was introduced in 2020." in claims


def test_pronoun_heavy_answer_preserves_years_and_pronoun_claims():
    claims = ClaimExtractor(atomic_splitting=True).extract_claims(
        "It was introduced in 2020. It uses late interaction to rank passages."
    )
    assert claims == [
        "It was introduced in 2020.",
        "It uses late interaction to rank passages.",
    ]


def test_malformed_answer_drops_dangling_list_fragments():
    claims = ClaimExtractor(atomic_splitting=True).extract_claims(
        "- RAG retrieves evidence\n2. and generation\n[broken citation"
    )
    assert claims == ["RAG retrieves evidence."]
