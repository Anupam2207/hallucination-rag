import numpy as np

from src.detection.support_scorer import SupportScorer
from src.generation.correction import AnswerCorrector, INSUFFICIENT_EVIDENCE_RESPONSE
from src.pipeline import HallucinationRAGPipeline
from src.retrieval.evidence_intent import (
    EXAMPLE,
    FACTUAL,
    annotate_evidence,
    classify_evidence_intent,
    is_strict_factual_query,
)


class FakeEmbedder:
    def encode(self, texts, normalize=True):
        vectors = []
        for text in texts:
            if "introduced by patrick lewis" in text.lower() or "rag was introduced" in text.lower():
                vectors.append(np.array([1.0, 0.0]))
            elif "for example" in text.lower():
                vectors.append(np.array([0.5, 0.5]))
            else:
                vectors.append(np.array([0.8, 0.2]))
        return np.array(vectors)


def test_example_patterns_force_example_type():
    text = 'For example, "RAG was introduced in 2021" is a hypothetical claim.'
    assert classify_evidence_intent(text, {"section_name": "Introduction"}) == EXAMPLE


def test_strict_mode_rejects_example_chunks_and_keeps_factual_assertions():
    query = "RAG was introduced in 2021."
    example = annotate_evidence(
        {"text": 'For example, "RAG was introduced in 2021" may be unsupported.', "metadata": {"section_name": "Background"}},
        query=query,
    )
    factual = annotate_evidence(
        {"text": "RAG was introduced by Patrick Lewis and colleagues in 2020.", "metadata": {"section_name": "Introduction", "file_type": "pdf", "file_name": "Lewis_RAG.pdf"}},
        query=query,
    )
    assert is_strict_factual_query(query)
    assert example["evidence_type"] == EXAMPLE
    assert example["rejected_by_strict_factual_mode"] is True
    assert factual["evidence_type"] == FACTUAL
    assert factual["evidence_credibility_score"] > example["evidence_credibility_score"]


def test_answerability_false_when_only_examples_are_available():
    evidence = [
        annotate_evidence({"text": 'For example, "RAG was introduced in 2021" is a hypothetical claim.', "metadata": {"section_name": "Background"}}, query="RAG was introduced in 2021."),
    ]
    assert not HallucinationRAGPipeline._is_answerable_by_score(evidence, 0.20, strict_factual_mode=True)


def test_support_scorer_uses_factual_evidence_not_example_leakage():
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        "RAG was introduced in 2020.",
        [
            {"text": 'For example, "RAG was introduced in 2021" is a hypothetical claim.', "metadata": {"section_name": "Background"}},
            {"text": "RAG was introduced by Patrick Lewis and colleagues in 2020.", "metadata": {"section_name": "Introduction", "file_type": "pdf", "file_name": "Lewis_RAG.pdf"}},
        ],
    )
    assert result["best_evidence_index"] == 1
    assert result["combined_evidence_types"] == [FACTUAL]
    assert "numeric_mismatch_with_evidence" not in result["rule_flags"]


def test_malformed_corrected_answer_falls_back():
    assert AnswerCorrector._is_malformed_answer("RAG was introduced by")
    assert not AnswerCorrector._is_malformed_answer(INSUFFICIENT_EVIDENCE_RESPONSE)
