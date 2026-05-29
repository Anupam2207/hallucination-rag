from src.pipeline import HallucinationRAGPipeline


def _evidence(text, section="Introduction"):
    return [{
        "text": text,
        "dense_similarity": 0.92,
        "metadata": {"section_name": section, "file_type": "pdf", "source_rel": "kb/test.pdf"},
    }]


def test_answerable_definition_query_passes():
    supported, reason = HallucinationRAGPipeline._specific_fact_supported_by_evidence(
        "What is retrieval-augmented generation?",
        _evidence("Retrieval-augmented generation is a method that combines retrieval with language generation."),
    )
    assert supported is True
    assert reason is None


def test_unanswerable_who_query_needs_person_or_organization_fact():
    supported, reason = HallucinationRAGPipeline._specific_fact_supported_by_evidence(
        "Who introduced retrieval-augmented generation?",
        _evidence("Retrieval-augmented generation is a method that combines retrieval with language generation."),
    )
    assert supported is False
    assert reason == "insufficient_evidence_for_specific_fact"


def test_wrong_year_query_does_not_pass_specific_fact_gate():
    supported, reason = HallucinationRAGPipeline._specific_fact_supported_by_evidence(
        "When was RAG introduced in 2021?",
        _evidence("RAG was introduced by Patrick Lewis and colleagues in 2020."),
    )
    assert supported is False
    assert reason == "insufficient_evidence_for_specific_fact"


def test_retrieved_example_text_does_not_count_as_specific_support():
    supported, _ = HallucinationRAGPipeline._specific_fact_supported_by_evidence(
        "Who introduced RAG?",
        _evidence('For example, the claim "RAG was introduced by Patrick Lewis in 2020" can be checked.', "Example"),
    )
    assert supported is False


def test_reference_section_text_does_not_count_as_specific_support():
    supported, _ = HallucinationRAGPipeline._specific_fact_supported_by_evidence(
        "Who introduced RAG?",
        _evidence("References [1] Patrick Lewis et al. 2020. Retrieval-Augmented Generation.", "References"),
    )
    assert supported is False
