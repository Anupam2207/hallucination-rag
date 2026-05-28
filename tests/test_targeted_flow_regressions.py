from src.detection.claim_extractor import ClaimExtractor
from src.generation.correction import AnswerCorrector
from src.pipeline import HallucinationRAGPipeline
from src.retrieval.evidence_intent import annotate_evidence, classify_evidence_intent, EXAMPLE
from src.retrieval.query_focus import extract_query_focus, has_topical_match


def test_rag_query_does_not_match_colbert_only_because_of_retrieval_word():
    focus = extract_query_focus("RAG was introduced in 2020")
    colbert_chunk = "ColBERT is a neural retrieval model. It was introduced by Omar Khattab and Matei Zaharia in 2020."
    assert not has_topical_match(colbert_chunk, {"document_title": "ColBERT Retrieval Model"}, focus)


def test_fact_verification_demo_chunk_is_example_not_factual_support():
    text = (
        'and "RAG was introduced in 2020" are highly similar but factually contradictory. '
        'A robust verification system must check factual details. The NLI label can be entailment.'
    )
    assert classify_evidence_intent(text, {"section_name": "Background"}) == EXAMPLE


def test_claim_extractor_keeps_four_digit_year_at_sentence_end():
    claims = ClaimExtractor().extract_claims(
        "ColBERT is a content-based image retrieval system that was introduced by S. Sivakumar, M. Swaminathan, and B. S. Manjunath in 2001."
    )
    assert claims
    assert "2001" in claims[0]


def test_colbert_relation_correction_uses_retrieved_factual_evidence_without_llm():
    evidence = [
        annotate_evidence(
            {
                "text": "ColBERT Retrieval Model ColBERT is a neural information retrieval model. It was introduced by Omar Khattab and Matei Zaharia in 2020 as a retrieval architecture.",
                "metadata": {
                    "file_name": "colbert_retrieval_model.pdf",
                    "file_type": "pdf",
                    "document_title": "ColBERT Retrieval Model",
                    "section_name": "Introduction",
                    "source_rel": "raw/pdf/colbert_retrieval_model.pdf",
                    "importance_score": 4,
                },
                "final_score": 0.95,
            },
            query="Who invented ColBERT and when?",
        )
    ]
    answer = AnswerCorrector()._deterministic_relation_answer("Who invented ColBERT and when?", evidence)
    assert answer == "ColBERT was introduced by Omar Khattab and Matei Zaharia in 2020."


def test_specific_rag_year_is_not_supported_by_off_entity_colbert_evidence():
    evidence = [
        annotate_evidence(
            {
                "text": "ColBERT is a neural retrieval model. It was introduced by Omar Khattab and Matei Zaharia in 2020.",
                "metadata": {"document_title": "ColBERT Retrieval Model", "section_name": "Introduction", "file_name": "colbert_retrieval_model.pdf", "file_type": "pdf"},
            },
            query="RAG was introduced in 2020",
        )
    ]
    supported, reason = HallucinationRAGPipeline._specific_fact_supported_by_evidence("RAG was introduced in 2020", evidence)
    assert not supported
    assert reason == "insufficient_evidence_for_specific_fact"
