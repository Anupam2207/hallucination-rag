import re
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from src.detection.factual_consistency import (
    extract_capitalized_entities,
    extract_years,
    run_factual_consistency_checks,
)
from src.detection.rule_flags import categorize_rule_flags
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.evidence_intent import (
    FACTUAL,
    annotate_evidence,
    is_credible_support_evidence,
)
from src.utils.text_cleaning import normalize_for_detection


class SupportScorer:
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
            "would", "should", "typically", "several", "various", "including",
        }
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
        return {word for word in words if word not in stopwords}

    @staticmethod
    def _exact_support_score(claim: str, evidence: str) -> float:
        clean_claim = normalize_for_detection(claim).lower().strip(" .")
        clean_evidence = normalize_for_detection(evidence).lower().strip()
        if not clean_claim or not clean_evidence:
            return 0.0
        if clean_claim in clean_evidence:
            return 0.96
        # Useful when the evidence sentence is slightly shorter than the claim.
        claim_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z]{2,}", clean_claim))
        evidence_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z]{2,}", clean_evidence))
        if len(claim_tokens) >= 6 and claim_tokens.issubset(evidence_tokens):
            return 0.90
        return 0.0

    def lexical_overlap_score(self, claim: str, evidence: str) -> float:
        claim_tokens = self._tokens(normalize_for_detection(claim))
        evidence_tokens = self._tokens(normalize_for_detection(evidence))
        if not claim_tokens or not evidence_tokens:
            return 0.0
        overlap = claim_tokens.intersection(evidence_tokens)
        if len(overlap) < 3:
            return 0.0
        recall = len(overlap) / len(claim_tokens)
        precision = len(overlap) / len(evidence_tokens)
        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)


    @staticmethod
    def _concepts(text: str) -> set[str]:
        """Return lightweight meaning concepts used for paraphrase support.

        This intentionally stays small and deterministic.  It is not a new
        model; it just maps common equivalent phrases in the project corpus to
        the same concept so valid paraphrases are not marked unsupported only
        because wording differs.
        """
        lower = normalize_for_detection(text).lower()
        concepts: set[str] = set()

        if re.search(r"\b(?:mfa|multi[- ]factor authentication)\b", lower):
            concepts.add("mfa_topic")

        if re.search(r"\b(?:two or more|multiple|more than one)\b.{0,60}\b(?:factor|factors|verification|verifications|forms?)\b", lower):
            concepts.add("more_than_one_factor")
        if re.search(r"\b(?:prove identity|provide verification|forms? of verification|verification factors?)\b", lower):
            concepts.add("identity_verification")

        if "something you know" in lower or re.search(r"\b(?:passwords?|pins?|passphrases?|security questions?)\b", lower):
            concepts.add("knowledge_factor")
        if "something you have" in lower or re.search(r"\b(?:phones?|smartphones?|mobile devices?|tokens?|smartcards?|smart cards?|hardware tokens?|physical devices?)\b", lower):
            concepts.add("possession_factor")
        if "something you are" in lower or re.search(r"\b(?:biometrics?|biometric traits?|fingerprints?|face|facial recognition|voice recognition|voice)\b", lower):
            concepts.add("biometric_factor")
        if "somewhere you are" in lower or re.search(r"\b(?:location[- ]based|geolocation|location factor)\b", lower):
            concepts.add("location_factor")

        if re.search(r"\b(?:stolen passwords?|stolen credentials|unauthorized access|extra layer|additional layer|security improvement|improves security|protect sensitive|harder for attackers|difficult for attackers|not enough to access|defeat the second factor)\b", lower):
            concepts.add("security_improvement")

        if re.search(r"\b(?:authenticator apps?|hardware security keys?|sms codes?|email codes?|biometric prompts?)\b", lower):
            concepts.add("mfa_methods")

        if re.search(r"\b(?:online banking|email accounts?|enterprise networks?)\b", lower):
            concepts.add("specific_applications")

        return concepts

    @classmethod
    def concept_coverage_score(cls, claim: str, evidence: str) -> float:
        claim_concepts = cls._concepts(claim)
        evidence_concepts = cls._concepts(evidence)
        # Topic-only matches are not enough to support a factual claim.
        claim_specific = {c for c in claim_concepts if c != "mfa_topic"}
        evidence_specific = {c for c in evidence_concepts if c != "mfa_topic"}
        if not claim_specific:
            return 0.0
        return len(claim_specific & evidence_specific) / len(claim_specific)

    @staticmethod
    def _missing_specific_evidence_flags(claim: str, evidence_texts: list[str]) -> list[str]:
        clean_claim = normalize_for_detection(claim)
        clean_evidence_texts = [normalize_for_detection(text) for text in evidence_texts]
        claim_lower = clean_claim.lower()
        evidence_lower = " ".join(clean_evidence_texts).lower()
        flags: list[str] = []

        phrase_groups = {
            "fine_tuning_not_in_evidence": [
                "fine-tun", "fine tun", "finetun", "fine tuned", "fine-tuned",
            ],
            "machine_translation_not_in_evidence": [
                "machine translation", "language translation",
            ],
            "sentiment_analysis_not_in_evidence": ["sentiment analysis"],
            "text_classification_not_in_evidence": ["text classification"],
            "document_classification_not_in_evidence": ["document classification"],
            "summarization_not_in_evidence": ["text summarization", "summarization"],
            "content_generation_not_in_evidence": ["content generation"],
            "dialogue_systems_not_in_evidence": [
                "conversational dialogue", "dialogue system", "dialogue systems", "conversational ai",
            ],
            "training_data_requirement_not_in_evidence": [
                "training data requirement", "less training data", "reduced training data",
                "smaller amounts of labeled data", "large amounts of training data",
                "large amount of training data", "requires training data", "require training data",
                "labeled data",
            ],
            "explicit_knowledge_representation_not_in_evidence": [
                "explicit knowledge representation", "knowledge representation",
            ],
            "style_tone_generation_not_in_evidence": ["style and tone", "tone and style"],
            "performance_comparison_not_in_evidence": [
                "better than", "outperform", "improved performance", "higher accuracy than",
                "more efficient than", "perform better than", "superior to", "higher performance",
            ],
            "interpretability_not_in_evidence": [
                "interpretability", "interpretable", "explainability", "explainable",
                "transparent", "transparency", "clear understanding", "traceability",
            ],
            "collaborative_filtering_not_in_evidence": ["collaborative filtering"],
            "recommendation_system_not_in_evidence": [
                "recommendation system", "recommendation systems", "recommendation accuracy",
            ],
            "ecommerce_not_in_evidence": ["e-commerce", "ecommerce"],
            "product_review_not_in_evidence": ["product descriptions", "reviews", "product reviews"],
            "collaborative_bert_not_in_evidence": ["collaborative bert"],
            "open_source_library_not_in_evidence": ["open-source library", "open source library"],
            "location_based_authentication_not_in_evidence": [
                "location-based authentication", "location based authentication",
                "somewhere you are", "location factor", "geolocation",
            ],
            "specific_application_not_in_evidence": [
                "online banking", "email accounts", "enterprise networks",
            ],
        }

        task_flags = {
            "machine_translation_not_in_evidence",
            "sentiment_analysis_not_in_evidence",
            "text_classification_not_in_evidence",
            "document_classification_not_in_evidence",
            "summarization_not_in_evidence",
            "content_generation_not_in_evidence",
            "dialogue_systems_not_in_evidence",
        }

        for flag, phrases in phrase_groups.items():
            claim_mentions = any(phrase in claim_lower for phrase in phrases)
            evidence_mentions = any(phrase in evidence_lower for phrase in phrases)
            if claim_mentions and not evidence_mentions:
                flags.append(flag)

        if any(flag in task_flags for flag in flags):
            flags.append("unsupported_task_example_not_in_evidence")

        return list(dict.fromkeys(flags))

    def _apply_rule_caps(
        self,
        claim: str,
        all_evidence_texts: list[str],
        best_evidence: str,
        score: float,
    ) -> tuple[float, list[str], dict[str, Any]]:
        flags = self._missing_specific_evidence_flags(claim, all_evidence_texts)
        factual_result = run_factual_consistency_checks(claim, best_evidence or "")
        factual_flags = list(factual_result.get("flags", []))
        flags.extend(factual_flags)
        flags = list(dict.fromkeys(flags))

        if not flags:
            return score, flags, factual_result

        if "numeric_mismatch_with_evidence" in flags:
            return min(score, 0.25), flags, factual_result
        if "entity_mismatch_with_evidence" in flags:
            return min(score, 0.30), flags, factual_result
        if any(flag in flags for flag in [
            "acronym_expansion_mismatch",
            "definition_mismatch_with_evidence",
        ]):
            return min(score, 0.30), flags, factual_result
        if any(flag in flags for flag in [
            "claim_year_not_supported_by_evidence", "claim_date_not_supported_by_evidence", "claim_numeric_not_supported_by_evidence"
        ]):
            return min(score, 0.35), flags, factual_result

        return min(score, 0.35), flags, factual_result

    def _score_text_pair(self, clean_claim: str, clean_evidence: str, claim_embedding=None) -> tuple[float, float, float]:
        exact_score = self._exact_support_score(clean_claim, clean_evidence)
        lexical_score = self.lexical_overlap_score(clean_claim, clean_evidence)
        concept_score = self.concept_coverage_score(clean_claim, clean_evidence)
        if claim_embedding is None:
            claim_embedding = self.embedder.encode([clean_claim])
        evidence_embedding = self.embedder.encode([clean_evidence])
        semantic_score = float(cosine_similarity(claim_embedding, evidence_embedding)[0][0])
        hybrid_score = max(semantic_score, exact_score)
        if concept_score >= 0.65:
            hybrid_score = max(hybrid_score, 0.78)
        elif concept_score >= 0.40:
            hybrid_score = max(hybrid_score, 0.50)
        if lexical_score >= 0.55:
            hybrid_score = max(hybrid_score, lexical_score)
        return hybrid_score, semantic_score, lexical_score


    @staticmethod
    def _factual_exact_match(claim: str, evidence: str, score: float, flags: list[str]) -> bool:
        """Return True when explicit factual values/entities in the claim are present in evidence.

        This prevents optional NLI ``neutral`` from downgrading claims where the
        evidence directly contains the same year/person/entity facts.
        """
        if flags:
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
        # Entity extraction is conservative and noisy; require either strong entity
        # containment or a high lexical/semantic score.
        if claim_entities:
            important_entities = {e for e in claim_entities if e.lower() not in {"colbert", "rag", "bert"}}
            if important_entities and not important_entities.issubset(evidence_entities):
                return False
        return bool(claim_years or claim_entities) and score >= 0.45

    @staticmethod
    def _rank_evidence_for_support(item: dict[str, Any]) -> tuple[float, float, float]:
        def as_float(key: str) -> float:
            try:
                return float(item.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        return (
            as_float("evidence_credibility_score"),
            as_float("final_score"),
            as_float("factual_assertion_score"),
        )

    def _prepare_support_evidence(self, evidence_list: list[Any]) -> list[tuple[int, dict[str, Any], str]]:
        prepared: list[tuple[int, dict[str, Any], str]] = []
        for original_index, raw_item in enumerate(evidence_list):
            if isinstance(raw_item, dict):
                item = dict(raw_item)
                if "evidence_type" not in item or "evidence_credibility_score" not in item:
                    item = annotate_evidence(item)
                text = self._extract_text(item).strip()
            else:
                text = self._extract_text(raw_item).strip()
                item = annotate_evidence({"text": text, "metadata": {}})
            if not text:
                continue
            if not is_credible_support_evidence(item, min_credibility=0.15):
                continue
            prepared.append((original_index, item, text))
        prepared.sort(key=lambda entry: self._rank_evidence_for_support(entry[1]), reverse=True)
        return prepared

    def _select_combined_support_items(
        self,
        prepared: list[tuple[int, dict[str, Any], str]],
        limit: int = 3,
    ) -> list[tuple[int, dict[str, Any], str]]:
        factual = [entry for entry in prepared if entry[1].get("evidence_type") == FACTUAL]
        source = factual if factual else prepared
        return source[: max(1, limit)]

    def _score_claim_internal(self, claim: str, evidence_list: list[Any]) -> dict[str, Any]:
        empty_result = {
            "claim": claim,
            "score": 0.0,
            "raw_similarity_score": 0.0,
            "semantic_score": 0.0,
            "lexical_score": 0.0,
            "best_evidence_index": None,
            "best_evidence": None,
            "rule_flags": [],
            "rule_categories": [],
            "factual_consistency": {"flags": [], "details": {}},
            "best_single_score": 0.0,
            "combined_context_score": 0.0,
            "combined_evidence": "",
            "combined_evidence_indices": [],
            "combined_evidence_types": [],
            "used_combined_evidence": False,
            "factual_exact_match": False,
            "evidence_confidence": 0.0,
            "no_factual_evidence": True,
        }
        if not claim.strip() or not evidence_list:
            return dict(empty_result)

        prepared = self._prepare_support_evidence(evidence_list)
        if not prepared:
            result = dict(empty_result)
            result["rule_flags"] = ["no_factual_evidence"]
            result["rule_categories"] = ["no_factual_evidence"]
            return result

        clean_claim = normalize_for_detection(claim)
        evidence_texts = [entry[2] for entry in prepared]
        clean_evidence_texts = [normalize_for_detection(text) for text in evidence_texts]
        claim_embedding = self.embedder.encode([clean_claim])
        evidence_embeddings = self.embedder.encode(clean_evidence_texts)
        similarities = cosine_similarity(claim_embedding, evidence_embeddings)[0]

        best_score = 0.0
        best_semantic_score = 0.0
        best_lexical_score = 0.0
        best_prepared_index: int | None = None
        for index, semantic_score in enumerate(similarities):
            lexical_score = self.lexical_overlap_score(clean_claim, clean_evidence_texts[index])
            exact_score = self._exact_support_score(clean_claim, clean_evidence_texts[index])
            concept_score = self.concept_coverage_score(clean_claim, clean_evidence_texts[index])
            hybrid_score = max(float(semantic_score), exact_score)
            if concept_score >= 0.65:
                hybrid_score = max(hybrid_score, 0.78)
            elif concept_score >= 0.40:
                hybrid_score = max(hybrid_score, 0.50)
            if lexical_score >= 0.55:
                hybrid_score = max(hybrid_score, lexical_score)
            if hybrid_score > best_score:
                best_score = hybrid_score
                best_semantic_score = float(semantic_score)
                best_lexical_score = float(lexical_score)
                best_prepared_index = index

        combined_items = self._select_combined_support_items(prepared, limit=3)
        combined_text = " ".join(normalize_for_detection(entry[2]) for entry in combined_items)[:4000]
        combined_score = 0.0
        combined_semantic_score = 0.0
        combined_lexical_score = 0.0
        if combined_text:
            # Combined verification is explicitly limited to the top factual
            # chunks when available, preventing examples/tutorials from leaking
            # into support decisions.
            exact_combined = self._exact_support_score(clean_claim, combined_text)
            combined_lexical_score = self.lexical_overlap_score(clean_claim, combined_text)
            combined_concept_score = self.concept_coverage_score(clean_claim, combined_text)
            combined_score = max(exact_combined, combined_lexical_score)
            if combined_concept_score >= 0.65:
                combined_score = max(combined_score, 0.78)
            elif combined_concept_score >= 0.40:
                combined_score = max(combined_score, 0.50)
            if combined_score < 0.55:
                combined_pair_score, combined_semantic_score, combined_pair_lexical = self._score_text_pair(
                    clean_claim,
                    combined_text,
                    claim_embedding=claim_embedding,
                )
                combined_score = max(combined_score, combined_pair_score)
                combined_lexical_score = max(combined_lexical_score, combined_pair_lexical)
            else:
                # Still compute the semantic component once for the final support
                # formula when lexical/exact support already shows grounding.
                _, combined_semantic_score, _ = self._score_text_pair(
                    clean_claim,
                    combined_text,
                    claim_embedding=claim_embedding,
                )

        used_combined = combined_score > best_score
        raw_support_score = max(best_score, combined_score)
        best_evidence = evidence_texts[best_prepared_index] if best_prepared_index is not None else ""
        best_evidence_clean = clean_evidence_texts[best_prepared_index] if best_prepared_index is not None else ""
        best_original_index = prepared[best_prepared_index][0] if best_prepared_index is not None else None
        evidence_for_rules = combined_text if used_combined and combined_text else best_evidence_clean
        capped_score, flags, factual_result = self._apply_rule_caps(
            clean_claim,
            clean_evidence_texts,
            evidence_for_rules,
            raw_support_score,
        )
        factual_exact_match = self._factual_exact_match(clean_claim, evidence_for_rules, raw_support_score, flags)
        if factual_exact_match:
            capped_score = max(capped_score, 0.72)

        selected_confidences: list[float] = []
        for entry in combined_items:
            try:
                selected_confidences.append(float(entry[1].get("evidence_credibility_score", 0.0) or 0.0))
            except (TypeError, ValueError):
                selected_confidences.append(0.0)
        if best_prepared_index is not None:
            try:
                selected_confidences.append(float(prepared[best_prepared_index][1].get("evidence_credibility_score", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
        evidence_confidence = max([raw_support_score, *selected_confidences], default=0.0)

        semantic_component = max(best_semantic_score, combined_semantic_score)
        lexical_component = max(best_lexical_score, combined_lexical_score)

        return {
            "claim": claim,
            "score": round(float(capped_score), 4),
            "raw_similarity_score": round(float(raw_support_score), 4),
            "semantic_score": round(float(semantic_component), 4),
            "lexical_score": round(float(lexical_component), 4),
            "best_evidence_index": best_original_index,
            "best_evidence": best_evidence,
            "rule_flags": flags,
            "rule_categories": categorize_rule_flags(flags),
            "factual_consistency": factual_result,
            "best_single_score": round(float(best_score), 4),
            "combined_context_score": round(float(combined_score), 4),
            "combined_evidence": " ".join(entry[2] for entry in combined_items),
            "combined_evidence_indices": [entry[0] for entry in combined_items],
            "combined_evidence_types": [entry[1].get("evidence_type") for entry in combined_items],
            "used_combined_evidence": bool(used_combined),
            "factual_exact_match": bool(factual_exact_match),
            "evidence_confidence": round(float(max(0.0, min(1.0, evidence_confidence))), 4),
            "no_factual_evidence": not any(entry[1].get("evidence_type") == FACTUAL for entry in prepared),
        }

    def score_claim(self, claim: str, evidence_list: list[Any]) -> tuple[float, int | None]:
        result = self._score_claim_internal(claim, evidence_list)
        return float(result["score"]), result["best_evidence_index"]

    def score_claim_against_evidence(self, claim: str, evidence_list: list[Any]) -> dict[str, Any]:
        return self._score_claim_internal(claim, evidence_list)
