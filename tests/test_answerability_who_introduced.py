from src.pipeline import HallucinationRAGPipeline


def test_who_introduced_rag_needs_person_evidence():
    evidence = [{"text": "RAG combines retrieval with generation."}]
    supported, reason = HallucinationRAGPipeline._specific_fact_supported_by_evidence("Who introduced RAG?", evidence)
    assert not supported
    assert reason == "insufficient_evidence_for_specific_fact"
