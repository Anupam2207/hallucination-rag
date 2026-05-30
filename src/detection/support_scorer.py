import re
from typing import Any, Dict, List, Tuple

from sklearn.metrics.pairwise import cosine_similarity

from src.detection.factual_consistency import (
    extract_capitalized_entities,
    extract_years,
    run_factual_consistency_checks,
)
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.query_focus import ALIAS_MAP, normalize_text, text_contains_known_topic
from src.utils.text_cleaning import normalize_for_detection


CONCEPT_MAP: Dict[str, List[str]] = {
    "rag": ["retrieval", "generation", "retriever", "generator", "external knowledge", "knowledge source", "passages", "grounded", "context"],
    "colbert": ["late interaction", "bert", "token", "embeddings", "retrieval", "passage", "query", "document"],
    "mfa": ["factor", "password", "pin", "phone", "token", "biometric", "identity", "authentication", "security"],
    "formula 1": ["formula", "drivers", "constructors", "championship", "grand prix", "season", "racing"],
    "vehicle safety": ["seat belt", "airbag", "brake", "collision", "safety", "driver", "passenger"],
    "smartphone": ["mobile", "touchscreen", "app", "internet", "camera", "processor", "communication"],
    "ai newsroom": ["newsroom", "journalism", "editor", "article", "automation", "reporting", "news"],
    "ancient indian architecture": ["temple", "stupa", "rock cut", "architecture", "indian", "monument", "stone"],
    "hallucination": ["unsupported", "false", "claim", "evidence", "model", "factual", "generated"],
    "fact verification": ["claim", "evidence", "support", "nli", "entailment", "contradiction", "verification"],
    "black hole": ["gravity", "light", "escape", "event horizon", "spacetime", "massive stars"],
    "crop rotation": ["different crops", "same field", "planned sequence", "seasons", "soil fertility", "pest", "yields"],
    "crispr": ["gene editing", "dna", "cas9", "guide rna", "targeted changes", "bacterial immune"],
    "eiffel tower": ["iron lattice", "paris", "gustave eiffel", "1889", "exposition universelle", "champ de mars"],
    "kubernetes": ["orchestrate containers", "containers", "clusters"],
    "phishing": ["suspicious links", "attachments", "sender addresses", "verification codes", "spear phishing"],
}

CRITICAL_FLAGS = {
    "numeric_mismatch_with_evidence",
    "claim_year_not_supported_by_evidence",
    "claim_date_not_supported_by_evidence",
    "claim_numeric_not_supported_by_evidence",
    "entity_mismatch_with_evidence",
    "fine_tuning_not_in_evidence",
    "unsupported_task_example_not_in_evidence",
    "training_data_requirement_not_in_evidence",
    "performance_comparison_not_in_evidence",
    "interpretability_not_in_evidence",
    "acronym_expansion_mismatch",
    "definition_mismatch_with_evidence",
    "collaborative_filtering_not_in_evidence",
    "recommendation_system_not_in_evidence",
    "collaborative_bert_not_in_evidence",
    "open_source_library_not_in_evidence",
    "ecommerce_not_in_evidence",
    "product_review_not_in_evidence",
}


class SupportScorer:
    """Claim-evidence scorer with explicit semantic and lexical components.

    This class intentionally does not perform NLI. It returns a clean evidence
    support profile that the detector fuses with NLI:
        FinalSupport = 0.30 semantic + 0.25 lexical + 0.45 entailment
    Rule flags are returned separately so hard factual mismatches can override
    otherwise high semantic similarity.
    """

    def __init__(self, embedder: EmbeddingModel | None = None) -> None:
        self.embedder = embedder or EmbeddingModel()

    @staticmethod
    def _extract_text(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return str(item.get("text", ""))
        return str(item)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "and", "or", "of", "to", "in", "on", "for", "with", "by", "as", "at",
            "from", "that", "this", "it", "its", "into", "than", "then", "while",
            "using", "use", "used", "uses", "both", "these", "those", "can", "could",
            "would", "should", "typically", "several", "various", "including", "include",
            "common", "commonly", "also", "more", "most", "such", "called", "known",
        }
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}|\b(?:19|20)\d{2}\b", normalize_for_detection(text).lower())
        return {word.strip("-") for word in words if word not in stopwords and len(word.strip("-")) > 1}

    @staticmethod
    def _exact_support_score(claim: str, evidence: str) -> float:
        clean_claim = normalize_for_detection(claim).lower().strip(" .")
        clean_evidence = normalize_for_detection(evidence).lower().strip()
        if not clean_claim or not clean_evidence:
            return 0.0
        if clean_claim in clean_evidence:
            return 1.0
        claim_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z]{2,}|\b(?:19|20)\d{2}\b", clean_claim))
        evidence_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z]{2,}|\b(?:19|20)\d{2}\b", clean_evidence))
        if len(claim_tokens) >= 5 and claim_tokens.issubset(evidence_tokens):
            return 0.92
        return 0.0

    def lexical_overlap_score(self, claim: str, evidence: str) -> float:
        claim_tokens = self._tokens(claim)
        evidence_tokens = self._tokens(evidence)
        if not claim_tokens or not evidence_tokens:
            return 0.0
        overlap = claim_tokens.intersection(evidence_tokens)
        if len(overlap) < 2 and len(claim_tokens) > 3:
            return 0.0
        recall = len(overlap) / len(claim_tokens)
        precision = len(overlap) / len(evidence_tokens)
        # Evidence chunks are longer than claims, so recall is more important.
        # Precision is compressed instead of allowed to dominate the score.
        precision_component = min(1.0, precision * 3.0)
        return max(0.0, min(1.0, 0.72 * recall + 0.28 * precision_component))

    @staticmethod
    def _missing_specific_evidence_flags(claim: str, evidence_texts: list[str]) -> list[str]:
        clean_claim = normalize_for_detection(claim)
        evidence_lower = " ".join(normalize_for_detection(text) for text in evidence_texts).lower()
        claim_lower = clean_claim.lower()
        flags: list[str] = []

        phrase_groups = {
            "fine_tuning_not_in_evidence": ["fine-tun", "fine tun", "finetun", "fine tuned", "fine-tuned"],
            "machine_translation_not_in_evidence": ["machine translation", "language translation"],
            "sentiment_analysis_not_in_evidence": ["sentiment analysis"],
            "text_classification_not_in_evidence": ["text classification"],
            "document_classification_not_in_evidence": ["document classification"],
            "summarization_not_in_evidence": ["text summarization", "summarization"],
            "content_generation_not_in_evidence": ["content generation"],
            "dialogue_systems_not_in_evidence": ["conversational dialogue", "dialogue system", "dialogue systems", "conversational ai"],
            "training_data_requirement_not_in_evidence": [
                "training data requirement", "less training data", "reduced training data",
                "smaller amounts of labeled data", "large amounts of training data", "large amount of training data",
                "requires training data", "require training data", "labeled data",
            ],
            "explicit_knowledge_representation_not_in_evidence": ["explicit knowledge representation", "knowledge representation"],
            "style_tone_generation_not_in_evidence": ["style and tone", "tone and style"],
            "performance_comparison_not_in_evidence": [
                "better than", "outperform", "improved performance", "higher accuracy than", "more efficient than",
                "perform better than", "superior to", "higher performance",
            ],
            "interpretability_not_in_evidence": [
                "interpretability", "interpretable", "explainability", "explainable", "transparent", "transparency",
                "clear understanding", "traceability",
            ],
            "collaborative_filtering_not_in_evidence": ["collaborative filtering"],
            "recommendation_system_not_in_evidence": ["recommendation system", "recommendation systems", "recommendation accuracy"],
            "ecommerce_not_in_evidence": ["e-commerce", "ecommerce"],
            "product_review_not_in_evidence": ["product descriptions", "reviews", "product reviews"],
            "collaborative_bert_not_in_evidence": ["collaborative bert"],
            "open_source_library_not_in_evidence": ["open-source library", "open source library"],
        }

        task_flags = {
            "machine_translation_not_in_evidence", "sentiment_analysis_not_in_evidence",
            "text_classification_not_in_evidence", "document_classification_not_in_evidence",
            "summarization_not_in_evidence", "content_generation_not_in_evidence", "dialogue_systems_not_in_evidence",
        }
        for flag, phrases in phrase_groups.items():
            claim_mentions = any(phrase in claim_lower for phrase in phrases)
            evidence_mentions = any(phrase in evidence_lower for phrase in phrases)
            if claim_mentions and not evidence_mentions:
                flags.append(flag)
        if any(flag in task_flags for flag in flags):
            flags.append("unsupported_task_example_not_in_evidence")
        return list(dict.fromkeys(flags))

    @staticmethod
    def _is_definition_like(text: str) -> bool:
        lower = normalize_for_detection(text).lower()
        return bool(re.search(r"\b(is|are|refers to|means|requires|combines|connects|has|have|uses)\b", lower))

    @staticmethod
    def _concept_overlap_score(claim: str, evidence: str) -> tuple[float, list[str]]:
        claim_norm = normalize_text(claim)
        evidence_norm = normalize_text(evidence)
        claim_topics = text_contains_known_topic(claim)
        evidence_topics = text_contains_known_topic(evidence)
        common_topics = claim_topics.intersection(evidence_topics)
        if not common_topics:
            # Alias matching may fail when the claim is pronoun-based; use any
            # topic present in evidence if claim terms overlap strongly.
            common_topics = evidence_topics
        matched: list[str] = []
        for topic in common_topics:
            for concept in CONCEPT_MAP.get(topic, []):
                c = normalize_text(concept)
                if c in claim_norm and c in evidence_norm:
                    matched.append(concept)
                else:
                    tokens = {tok for tok in re.findall(r"[a-z0-9]+", c) if len(tok) > 2}
                    if tokens and tokens.issubset(set(re.findall(r"[a-z0-9]+", claim_norm))) and tokens.issubset(set(re.findall(r"[a-z0-9]+", evidence_norm))):
                        matched.append(concept)
        matched = list(dict.fromkeys(matched))
        if len(matched) >= 3:
            return 0.74, matched
        if len(matched) == 2:
            return 0.58, matched
        return 0.0, matched

    @staticmethod
    def _apply_rule_caps(score: float, flags: list[str]) -> float:
        if not flags:
            return score
        if "numeric_mismatch_with_evidence" in flags:
            return min(score, 0.25)
        if "entity_mismatch_with_evidence" in flags:
            return min(score, 0.30)
        if any(flag in flags for flag in ["acronym_expansion_mismatch", "definition_mismatch_with_evidence"]):
            return min(score, 0.30)
        if any(flag in flags for flag in ["claim_year_not_supported_by_evidence", "claim_date_not_supported_by_evidence", "claim_numeric_not_supported_by_evidence"]):
            return min(score, 0.35)
        if any(flag in CRITICAL_FLAGS for flag in flags):
            return min(score, 0.35)
        return min(score, 0.45)

    @staticmethod
    def _factual_exact_match(claim: str, evidence: str, score: float, flags: list[str]) -> bool:
        if any(flag in CRITICAL_FLAGS for flag in flags):
            return False
        clean_claim = normalize_for_detection(claim)
        clean_evidence = normalize_for_detection(evidence)
        if score < 0.45 or not clean_claim or not clean_evidence:
            return False
        claim_years = set(extract_years(clean_claim))
        evidence_years = set(extract_years(clean_evidence))
        if claim_years and not claim_years.issubset(evidence_years):
            return False
        claim_entities = set(extract_capitalized_entities(clean_claim))
        evidence_entities = set(extract_capitalized_entities(clean_evidence))
        if claim_entities:
            important_entities = {e for e in claim_entities if e.lower() not in {"colbert", "rag", "bert", "mfa"}}
            if important_entities and not important_entities.issubset(evidence_entities):
                return False
        return bool(claim_years or claim_entities) and score >= 0.45

    def _score_pair(self, claim: str, evidence: str, claim_embedding=None) -> Dict[str, float]:
        if claim_embedding is None:
            claim_embedding = self.embedder.encode([claim])
        evidence_embedding = self.embedder.encode([evidence])
        semantic = float(cosine_similarity(claim_embedding, evidence_embedding)[0][0])
        semantic = max(0.0, min(1.0, semantic))
        lexical = self.lexical_overlap_score(claim, evidence)
        exact = self._exact_support_score(claim, evidence)
        evidence_support = max(exact, semantic, 0.62 * semantic + 0.38 * lexical)
        if exact >= 0.90:
            evidence_support = max(evidence_support, exact)
        return {
            "semantic_score": round(semantic, 6),
            "lexical_score": round(lexical, 6),
            "exact_score": round(exact, 6),
            "evidence_support_score": round(max(0.0, min(1.0, evidence_support)), 6),
        }

    def _empty_result(self, claim: str) -> dict[str, Any]:
        return {
            "claim": claim,
            "score": 0.0,
            "raw_similarity_score": 0.0,
            "semantic_score": 0.0,
            "lexical_score": 0.0,
            "exact_score": 0.0,
            "evidence_support_score": 0.0,
            "best_evidence_index": None,
            "best_evidence": None,
            "nli_evidence_text": "",
            "rule_flags": [],
            "factual_consistency": {"flags": [], "details": {}},
            "best_single_score": 0.0,
            "combined_context_score": 0.0,
            "used_combined_evidence": False,
            "factual_exact_match": False,
            "definition_concept_matches": [],
        }

    def _score_claim_internal(self, claim: str, evidence_list: list[Any]) -> dict[str, Any]:
        if not claim.strip() or not evidence_list:
            return self._empty_result(claim)

        evidence_texts = [self._extract_text(item) for item in evidence_list]
        evidence_texts = [text for text in evidence_texts if text and text.strip()]
        if not evidence_texts:
            return self._empty_result(claim)

        clean_claim = normalize_for_detection(claim)
        clean_evidence_texts = [normalize_for_detection(text) for text in evidence_texts]
        claim_embedding = self.embedder.encode([clean_claim])

        best_index: int | None = None
        best_pair: Dict[str, float] | None = None
        for index, evidence_text in enumerate(clean_evidence_texts):
            pair = self._score_pair(clean_claim, evidence_text, claim_embedding=claim_embedding)
            if best_pair is None or pair["evidence_support_score"] > best_pair["evidence_support_score"]:
                best_pair = pair
                best_index = index

        best_pair = best_pair or {"semantic_score": 0.0, "lexical_score": 0.0, "exact_score": 0.0, "evidence_support_score": 0.0}
        best_evidence = evidence_texts[best_index] if best_index is not None else ""
        best_evidence_clean = clean_evidence_texts[best_index] if best_index is not None else ""

        # Combined evidence uses top-3 retrieved chunks, which is enough for most
        # claims but avoids noisy long context.
        combined_text = " ".join(clean_evidence_texts[: min(3, len(clean_evidence_texts))])[:3500]
        combined_pair = self._score_pair(clean_claim, combined_text, claim_embedding=claim_embedding) if combined_text else best_pair
        used_combined = combined_pair["evidence_support_score"] > best_pair["evidence_support_score"] + 0.03
        scoring_evidence = combined_text if used_combined else best_evidence_clean
        nli_evidence_text = scoring_evidence or best_evidence_clean

        semantic_score = max(best_pair["semantic_score"], combined_pair["semantic_score"] if used_combined else best_pair["semantic_score"])
        lexical_score = max(best_pair["lexical_score"], combined_pair["lexical_score"] if used_combined else best_pair["lexical_score"])
        exact_score = max(best_pair["exact_score"], combined_pair["exact_score"] if used_combined else best_pair["exact_score"])
        evidence_support = max(best_pair["evidence_support_score"], combined_pair["evidence_support_score"] if used_combined else best_pair["evidence_support_score"])

        flags = self._missing_specific_evidence_flags(clean_claim, [scoring_evidence or best_evidence_clean])
        factual_result = run_factual_consistency_checks(clean_claim, scoring_evidence or best_evidence_clean)
        flags.extend(list(factual_result.get("flags", [])))
        flags = list(dict.fromkeys(flags))

        definition_matches: list[str] = []
        if self._is_definition_like(clean_claim):
            boost, definition_matches = self._concept_overlap_score(clean_claim, scoring_evidence or best_evidence_clean)
            if boost > 0 and not any(flag in CRITICAL_FLAGS for flag in flags):
                evidence_support = max(evidence_support, boost)
                lexical_score = max(lexical_score, min(0.72, boost))

        capped_score = self._apply_rule_caps(evidence_support, flags)
        factual_exact_match = self._factual_exact_match(clean_claim, scoring_evidence or best_evidence_clean, evidence_support, flags)
        if factual_exact_match:
            capped_score = max(capped_score, 0.72)

        return {
            "claim": claim,
            "score": round(float(capped_score), 4),
            "raw_similarity_score": round(float(evidence_support), 4),
            "semantic_score": round(float(semantic_score), 4),
            "lexical_score": round(float(lexical_score), 4),
            "exact_score": round(float(exact_score), 4),
            "evidence_support_score": round(float(evidence_support), 4),
            "best_evidence_index": best_index,
            "best_evidence": best_evidence,
            "nli_evidence_text": nli_evidence_text,
            "rule_flags": flags,
            "factual_consistency": factual_result,
            "best_single_score": round(float(best_pair["evidence_support_score"]), 4),
            "combined_context_score": round(float(combined_pair["evidence_support_score"]), 4),
            "used_combined_evidence": bool(used_combined),
            "factual_exact_match": bool(factual_exact_match),
            "definition_concept_matches": definition_matches,
        }

    def score_claim(self, claim: str, evidence_list: list[Any]) -> tuple[float, int | None]:
        result = self._score_claim_internal(claim, evidence_list)
        return float(result["score"]), result["best_evidence_index"]

    def score_claim_against_evidence(self, claim: str, evidence_list: list[Any]) -> dict[str, Any]:
        return self._score_claim_internal(claim, evidence_list)
