from src.detection.factual_consistency import run_factual_consistency_checks


def test_year_mismatch_is_flagged() -> None:
    result = run_factual_consistency_checks(
        "RAG was introduced in 2021.",
        "RAG was introduced in 2020.",
    )
    assert "numeric_mismatch_with_evidence" in result["flags"]


def test_year_not_supported_is_flagged() -> None:
    result = run_factual_consistency_checks(
        "RAG was introduced in 2021.",
        "RAG combines retrieval with language generation.",
    )
    assert "claim_year_not_supported_by_evidence" in result["flags"]


def test_matching_year_is_not_flagged() -> None:
    result = run_factual_consistency_checks(
        "The Formula 1 hybrid era began in 2014.",
        "The hybrid era officially began in Formula 1 in 2014.",
    )
    assert result["flags"] == []


def test_entity_mismatch_is_flagged_for_same_relation() -> None:
    result = run_factual_consistency_checks(
        "Max Verstappen won the 2021 Formula 1 championship.",
        "Lewis Hamilton won the 2021 Formula 1 championship.",
    )
    assert "entity_mismatch_with_evidence" in result["flags"]


def test_citation_number_is_ignored() -> None:
    result = run_factual_consistency_checks(
        "RAG combines retrieval and generation [Evidence-1].",
        "RAG combines retrieval with language generation.",
    )
    assert result["details"]["claim_numbers"] == []
    assert result["flags"] == []
