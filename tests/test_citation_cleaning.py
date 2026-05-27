from src.utils.text_cleaning import normalize_evidence_citations, normalize_for_detection, strip_evidence_citations
from src.detection.factual_consistency import extract_numbers


def test_grouped_evidence_citations_are_stripped() -> None:
    text = "RAG uses retrieval [Evidence-1, Evidence-2]."
    cleaned = normalize_for_detection(text)
    assert "Evidence" not in cleaned
    assert cleaned == "RAG uses retrieval."
    assert extract_numbers(cleaned) == []


def test_compact_grouped_evidence_citations_are_stripped() -> None:
    text = "RAG uses retrieval [Evidence-1,Evidence-2]."
    cleaned = normalize_for_detection(text)
    assert "Evidence" not in cleaned
    assert extract_numbers(cleaned) == []


def test_numeric_citations_are_stripped() -> None:
    text = "RAG uses retrieval [1, 2]."
    cleaned = normalize_for_detection(text)
    assert cleaned == "RAG uses retrieval."
    assert extract_numbers(cleaned) == []


def test_malformed_citation_fragment_is_stripped() -> None:
    assert strip_evidence_citations("Evidence-1] RAG uses retrieval.") == "RAG uses retrieval."


def test_citation_normalizer_standardizes_spacing() -> None:
    text = "RAG uses retrieval [Evidence-1,Evidence-2]."
    assert normalize_evidence_citations(text) == "RAG uses retrieval [Evidence-1, Evidence-2]."
