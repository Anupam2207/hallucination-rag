import re
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from src.detection.factual_consistency import run_factual_consistency_checks
from src.retrieval.embedder import EmbeddingModel


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

    def lexical_overlap_score(self, claim: str, evidence: str) -> float:
        claim_tokens = self._tokens(claim)
        evidence_tokens = self._tokens(evidence)
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
    def _missing_specific_evidence_flags(claim: str, evidence_texts: list[str]) -> list[str]:
        claim_lower = claim.lower()
        evidence_lower = " ".join(evidence_texts).lower()
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
                "conversational dialogue", "dialogue system", "dialogue systems",
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

        # Numeric/entity contradictions are stronger than unsupported examples.
        if any(flag in flags for flag in ["numeric_mismatch_with_evidence", "entity_mismatch_with_evidence"]):
            return min(score, 0.35), flags, factual_result

        return min(score, 0.35), flags, factual_result

    def _score_claim_internal(self, claim: str, evidence_list: list[Any]) -> dict[str, Any]:
        if not claim.strip() or not evidence_list:
            return {
                "claim": claim,
                "score": 0.0,
                "raw_similarity_score": 0.0,
                "best_evidence_index": None,
                "best_evidence": None,
                "rule_flags": [],
                "factual_consistency": {"flags": [], "details": {}},
            }

        evidence_texts = [self._extract_text(item) for item in evidence_list]
        evidence_texts = [text for text in evidence_texts if text.strip()]
        if not evidence_texts:
            return {
                "claim": claim,
                "score": 0.0,
                "raw_similarity_score": 0.0,
                "best_evidence_index": None,
                "best_evidence": None,
                "rule_flags": [],
                "factual_consistency": {"flags": [], "details": {}},
            }

        claim_embedding = self.embedder.encode([claim])
        evidence_embeddings = self.embedder.encode(evidence_texts)
        similarities = cosine_similarity(claim_embedding, evidence_embeddings)[0]

        best_score = 0.0
        best_raw_score = 0.0
        best_index: int | None = None
        for index, semantic_score in enumerate(similarities):
            lexical_score = self.lexical_overlap_score(claim, evidence_texts[index])
            if lexical_score >= 0.60:
                hybrid_score = max(float(semantic_score), float(lexical_score))
            else:
                hybrid_score = float(semantic_score)
            if hybrid_score > best_score:
                best_score = hybrid_score
                best_raw_score = hybrid_score
                best_index = index

        best_evidence = evidence_texts[best_index] if best_index is not None else ""
        capped_score, flags, factual_result = self._apply_rule_caps(
            claim,
            evidence_texts,
            best_evidence,
            best_score,
        )

        return {
            "claim": claim,
            "score": round(float(capped_score), 4),
            "raw_similarity_score": round(float(best_raw_score), 4),
            "best_evidence_index": best_index,
            "best_evidence": best_evidence,
            "rule_flags": flags,
            "factual_consistency": factual_result,
        }

    def score_claim(self, claim: str, evidence_list: list[Any]) -> tuple[float, int | None]:
        result = self._score_claim_internal(claim, evidence_list)
        return float(result["score"]), result["best_evidence_index"]

    def score_claim_against_evidence(self, claim: str, evidence_list: list[Any]) -> dict[str, Any]:
        return self._score_claim_internal(claim, evidence_list)
