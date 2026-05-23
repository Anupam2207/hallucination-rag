from src.detection.span_highlighter import highlight_hallucinated_spans


def test_numeric_span_highlight() -> None:
    result = highlight_hallucinated_spans(
        "RAG was introduced in 2021.",
        "RAG was introduced in 2020.",
        ["numeric_mismatch_with_evidence"],
    )
    assert "[2021]" in result["highlighted_claim"]
    assert result["hallucinated_spans"][0]["text"] == "2021"


def test_unsupported_year_span_highlight() -> None:
    result = highlight_hallucinated_spans(
        "RAG was introduced in 2021.",
        "RAG combines retrieval and generation.",
        ["claim_year_not_supported_by_evidence"],
    )
    assert "[2021]" in result["highlighted_claim"]


def test_fine_tuning_span_highlight() -> None:
    result = highlight_hallucinated_spans(
        "RAG fine-tunes a generator model on retrieved passages.",
        "RAG uses retrieved passages as context.",
        ["fine_tuning_not_in_evidence"],
    )
    assert result["hallucinated_spans"]


def test_citation_number_not_highlighted() -> None:
    result = highlight_hallucinated_spans(
        "RAG combines retrieval and generation [Evidence-1].",
        "RAG combines retrieval and generation.",
        [],
    )
    assert result["hallucinated_spans"] == []
    assert "[Evidence-1]" in result["highlighted_claim"]
