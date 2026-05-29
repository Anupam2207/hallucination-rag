import numpy as np

from src.detection.rule_flags import categorize_rule_flags, legacy_flags_for_category
from src.detection.support_scorer import SupportScorer


class FakeEmbedder:
    def encode(self, texts, normalize=True):
        return np.array([[1.0, 0.0] for _ in texts])


def test_legacy_flags_map_to_generic_categories():
    categories = categorize_rule_flags([
        "fine_tuning_not_in_evidence",
        "claim_year_not_supported_by_evidence",
        "entity_mismatch_with_evidence",
    ])
    assert categories == [
        "unsupported_method_claim",
        "temporal_mismatch",
        "entity_mismatch",
    ]
    assert "fine_tuning_not_in_evidence" in legacy_flags_for_category("unsupported_method_claim")


def test_old_demo_rule_flag_still_exposed_with_generic_category():
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        "RAG fine-tunes a generator model on retrieved passages for each query.",
        [{"text": "RAG uses retrieved passages as context while generating an answer."}],
    )
    assert "fine_tuning_not_in_evidence" in result["rule_flags"]
    assert "unsupported_method_claim" in result["rule_categories"]
    assert result["score"] <= 0.35
