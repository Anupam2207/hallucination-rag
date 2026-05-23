from src.utils.text_cleaning import normalize_for_detection, strip_evidence_citations


def test_strip_evidence_citations_removes_citation_numbers() -> None:
    text = "RAG combines retrieval [Evidence-1] and generation [2]."
    cleaned = normalize_for_detection(text)
    assert "Evidence" not in cleaned
    assert "[2]" not in cleaned
    assert "1" not in cleaned
    assert "2" not in cleaned


def test_malformed_citation_fragment_removed() -> None:
    assert strip_evidence_citations("Evidence-1] RAG is useful.").strip() == "RAG is useful."
