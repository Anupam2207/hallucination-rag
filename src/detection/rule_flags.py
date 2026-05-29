"""Rule-flag taxonomy for paper-friendly hallucination analysis.

The detector historically exposed narrow rule flags such as
``fine_tuning_not_in_evidence``.  Those flags remain stable for tests and UI
compatibility, while this module maps them into generic categories that are
simpler to report in a methodology section.
"""

from __future__ import annotations

from typing import Iterable


GENERIC_RULE_CATEGORIES = {
    "numeric_mismatch",
    "temporal_mismatch",
    "entity_mismatch",
    "definition_mismatch",
    "unsupported_method_claim",
    "unsupported_task_claim",
    "unsupported_application_claim",
    "unsupported_performance_claim",
    "unsupported_training_claim",
}

OLD_TO_GENERIC_RULE_FLAGS: dict[str, str] = {
    "numeric_mismatch_with_evidence": "numeric_mismatch",
    "claim_numeric_not_supported_by_evidence": "numeric_mismatch",
    "claim_year_not_supported_by_evidence": "temporal_mismatch",
    "claim_date_not_supported_by_evidence": "temporal_mismatch",
    "entity_mismatch_with_evidence": "entity_mismatch",
    "acronym_expansion_mismatch": "definition_mismatch",
    "definition_mismatch_with_evidence": "definition_mismatch",
    "collaborative_bert_not_in_evidence": "definition_mismatch",
    "fine_tuning_not_in_evidence": "unsupported_method_claim",
    "explicit_knowledge_representation_not_in_evidence": "unsupported_method_claim",
    "collaborative_filtering_not_in_evidence": "unsupported_method_claim",
    "open_source_library_not_in_evidence": "unsupported_application_claim",
    "unsupported_task_example_not_in_evidence": "unsupported_task_claim",
    "machine_translation_not_in_evidence": "unsupported_task_claim",
    "sentiment_analysis_not_in_evidence": "unsupported_task_claim",
    "text_classification_not_in_evidence": "unsupported_task_claim",
    "document_classification_not_in_evidence": "unsupported_task_claim",
    "summarization_not_in_evidence": "unsupported_task_claim",
    "content_generation_not_in_evidence": "unsupported_task_claim",
    "dialogue_systems_not_in_evidence": "unsupported_task_claim",
    "ecommerce_not_in_evidence": "unsupported_application_claim",
    "product_review_not_in_evidence": "unsupported_application_claim",
    "recommendation_system_not_in_evidence": "unsupported_application_claim",
    "performance_comparison_not_in_evidence": "unsupported_performance_claim",
    "interpretability_not_in_evidence": "unsupported_performance_claim",
    "training_data_requirement_not_in_evidence": "unsupported_training_claim",
    "style_tone_generation_not_in_evidence": "unsupported_task_claim",
    "nli_contradiction": "entity_mismatch",
}

GENERIC_TO_LEGACY_RULE_FLAGS: dict[str, set[str]] = {}
for old_flag, category in OLD_TO_GENERIC_RULE_FLAGS.items():
    GENERIC_TO_LEGACY_RULE_FLAGS.setdefault(category, set()).add(old_flag)


def categorize_rule_flag(flag: str) -> str:
    """Return a generic category for a legacy or generic rule flag."""
    if flag in GENERIC_RULE_CATEGORIES:
        return flag
    return OLD_TO_GENERIC_RULE_FLAGS.get(flag, flag)


def categorize_rule_flags(flags: Iterable[str] | None) -> list[str]:
    """Map flags to stable generic categories, preserving insertion order."""
    categories: list[str] = []
    for flag in flags or []:
        category = categorize_rule_flag(str(flag))
        if category and category not in categories:
            categories.append(category)
    return categories


def legacy_flags_for_category(category: str) -> set[str]:
    """Return old flags mapped to a generic category."""
    return set(GENERIC_TO_LEGACY_RULE_FLAGS.get(category, set()))
