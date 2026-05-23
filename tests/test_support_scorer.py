import numpy as np

from src.detection.support_scorer import SupportScorer


class FakeEmbedder:
    def encode(self, texts, normalize=True):
        vectors = []
        for text in texts:
            if text == 'bad evidence':
                vectors.append(np.array([0.0, 1.0]))
            else:
                vectors.append(np.array([1.0, 0.0]))
        return np.array(vectors)


def test_support_scorer_selects_best_evidence() -> None:
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        'claim',
        [{'text': 'bad evidence'}, {'text': 'good evidence'}],
    )
    assert result['best_evidence'] == 'good evidence'
    assert result['score'] > 0.99


def test_fine_tuning_claim_gets_rule_flag() -> None:
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        'RAG fine-tunes a generator model on retrieved passages for each query.',
        [{'text': 'RAG uses retrieved passages as context while generating an answer.'}],
    )
    assert 'fine_tuning_not_in_evidence' in result['rule_flags']
    assert result['score'] <= 0.35


def test_valid_rag_context_claim_not_flagged() -> None:
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        'RAG uses retrieved passages as context while generating an answer.',
        [{'text': 'RAG uses retrieved passages as context while generating an answer.'}],
    )
    assert result['rule_flags'] == []
    assert result['score'] > 0.5


def test_unsupported_task_examples_get_rule_flag() -> None:
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        'RAG is effective for question answering, text summarization, and document classification.',
        [{'text': 'RAG is commonly used to reduce unsupported claims in question answering systems.'}],
    )
    assert 'unsupported_task_example_not_in_evidence' in result['rule_flags']
    assert result['score'] <= 0.35


def test_numeric_mismatch_caps_score() -> None:
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        'RAG was introduced in 2021.',
        [{'text': 'RAG was introduced in 2020.'}],
    )
    assert 'numeric_mismatch_with_evidence' in result['rule_flags']
    assert result['score'] <= 0.35


def test_performance_comparison_claim_gets_rule_flag() -> None:
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        'RAG models can perform better than traditional generative models.',
        [{'text': 'RAG combines retrieval with generation and uses retrieved evidence.'}],
    )
    assert 'performance_comparison_not_in_evidence' in result['rule_flags']
    assert result['score'] <= 0.35


def test_interpretability_claim_gets_rule_flag() -> None:
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        'The retrieval step provides a clear understanding of what information is used.',
        [{'text': 'RAG retrieves evidence and uses it as context for generation.'}],
    )
    assert 'interpretability_not_in_evidence' in result['rule_flags']
    assert result['score'] <= 0.35


def test_unsupported_year_without_evidence_year_gets_rule_flag() -> None:
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        'RAG was introduced in 2021.',
        [{'text': 'RAG combines retrieval with language generation.'}],
    )
    assert 'claim_year_not_supported_by_evidence' in result['rule_flags']
    assert result['score'] <= 0.35


def test_citation_number_not_used_as_factual_number() -> None:
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        'RAG combines retrieval and generation [Evidence-1].',
        [{'text': 'RAG combines retrieval and generation.'}],
    )
    assert 'claim_numeric_not_supported_by_evidence' not in result['rule_flags']
    assert 'claim_year_not_supported_by_evidence' not in result['rule_flags']
