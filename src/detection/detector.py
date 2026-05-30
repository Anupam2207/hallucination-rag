from typing import Any
import re

from src.config import get_config_value
from src.detection.claim_extractor import ClaimExtractor
from src.detection.nli_verifier import NLIVerifier
from src.detection.span_highlighter import highlight_hallucinated_spans
from src.detection.support_scorer import CRITICAL_FLAGS, SupportScorer
from src.retrieval.embedder import EmbeddingModel
from src.utils.text_cleaning import normalize_for_detection


class HallucinationDetector:
    def __init__(
        self,
        extractor: ClaimExtractor | None = None,
        scorer: SupportScorer | None = None,
        support_scorer: SupportScorer | None = None,
        nli_verifier: NLIVerifier | None = None,
    ) -> None:
        self.extractor = extractor or ClaimExtractor()
        if scorer is not None:
            self.scorer = scorer
        elif support_scorer is not None:
            self.scorer = support_scorer
        else:
            self.scorer = SupportScorer(embedder=EmbeddingModel())

        self.nli_verifier = nli_verifier or NLIVerifier()
        self.support_threshold = float(
            get_config_value("settings", "detection", "similarity_support_threshold", default=0.70)
        )
        self.warning_threshold = float(
            get_config_value("settings", "detection", "similarity_warning_threshold", default=0.45)
        )
        self.nli_entailment_threshold = float(
            get_config_value("settings", "detection", "nli_entailment_threshold", default=0.60)
        )
        self.nli_contradiction_threshold = float(
            get_config_value("settings", "detection", "nli_contradiction_threshold", default=0.60)
        )
        self.nli_min_similarity_to_run = float(
            get_config_value("settings", "verification", "nli_min_similarity_to_run", default=0.0)
        )
        self.nli_contradiction_relevance_threshold = float(
            get_config_value("settings", "verification", "nli_contradiction_relevance_threshold", default=0.35)
        )

    def _label_from_similarity(self, score: float) -> str:
        if score >= self.support_threshold:
            return "supported"
        if score >= self.warning_threshold:
            return "weak_support"
        return "unsupported"

    @staticmethod
    def critical_rule_flags() -> set[str]:
        return set(CRITICAL_FLAGS).union({"nli_contradiction"})

    @staticmethod
    def _shared_named_tokens(claim: str, evidence: str) -> bool:
        def caps(text: str) -> set[str]:
            return set(re.findall(r"\b[A-Z][A-Za-z]{2,}\b|\b[A-Z]{2,}\b", text or ""))
        stop = {"The", "This", "That", "Evidence", "Claim"}
        return bool((caps(claim) - stop) & (caps(evidence) - stop))

    def _nli_support_value(
        self,
        adjusted_label: str | None,
        nli_result: dict,
        evidence_support: float = 0.0,
        semantic_score: float = 0.0,
        lexical_score: float = 0.0,
    ) -> float:
        scores = nli_result.get("scores") or {}
        available = bool(nli_result.get("available"))
        if adjusted_label == "entailment":
            return max(0.0, min(1.0, float(scores.get("entailment", nli_result.get("score") or 0.85))))
        if adjusted_label == "contradiction":
            return 0.0
        if adjusted_label == "neutral":
            # Neutral is common for paraphrases, so keep it moderate rather than
            # treating it as failure.
            return max(0.50, min(0.68, 0.50 + 0.18 * max(semantic_score, lexical_score)))
        if not available or adjusted_label in {"unavailable", "not_run", "error", "not_enough_input", None}:
            # Transparent fallback when the NLI model is unavailable. The output
            # still reports nli_available=false, but support does not collapse to
            # unsupported solely because the local model is missing.
            return max(0.45, min(0.72, 0.35 + 0.45 * evidence_support + 0.15 * lexical_score))
        return 0.50

    def _adjust_nli_label(
        self,
        nli_label: str | None,
        nli_score: float | None,
        raw_similarity_score: float,
        claim: str,
        evidence: str,
        fused_flags: list[str],
        lexical_score: float | None = None,
    ) -> tuple[str | None, str | None, list[str]]:
        if nli_label != "contradiction" or nli_score is None or nli_score < self.nli_contradiction_threshold:
            return nli_label, None, fused_flags

        lexical = lexical_score if lexical_score is not None else self.scorer.lexical_overlap_score(claim, evidence)
        topically_related = (
            raw_similarity_score >= self.nli_contradiction_relevance_threshold
            or lexical >= 0.20
            or self._shared_named_tokens(claim, evidence)
        )
        if topically_related:
            return "contradiction", None, fused_flags

        if "nli_contradiction_low_relevance_ignored" not in fused_flags:
            fused_flags.append("nli_contradiction_low_relevance_ignored")
        return "neutral", "low_topical_relevance", fused_flags

    def _fuse_decision(
        self,
        score: float,
        raw_similarity_score: float,
        similarity_label: str,
        rule_flags: list[str],
        nli_result: dict,
        claim: str,
        evidence: str,
        factual_exact_match: bool = False,
        semantic_score: float | None = None,
        lexical_score: float | None = None,
        exact_score: float | None = None,
    ) -> tuple[str, float, list[str], str | None, str | None, float | None]:
        fused_flags = list(dict.fromkeys(rule_flags))
        evidence_support = max(0.0, min(1.0, float(score)))
        semantic_component = max(0.0, min(1.0, float(semantic_score if semantic_score is not None else raw_similarity_score)))
        lexical_component = max(0.0, min(1.0, float(lexical_score if lexical_score is not None else self.scorer.lexical_overlap_score(claim, evidence))))
        exact_component = max(0.0, min(1.0, float(exact_score if exact_score is not None else 0.0)))
        nli_label = nli_result.get("label")
        nli_score = nli_result.get("score")
        nli_score = float(nli_score) if nli_score is not None else None

        adjusted_label, adjustment_reason, fused_flags = self._adjust_nli_label(
            nli_label,
            nli_score,
            raw_similarity_score,
            claim,
            evidence,
            fused_flags,
            lexical_component,
        )

        critical_flags = self.critical_rule_flags()
        if adjusted_label == "contradiction":
            if factual_exact_match and not any(flag in critical_flags for flag in fused_flags):
                if "nli_contradiction_factual_match_ignored" not in fused_flags:
                    fused_flags.append("nli_contradiction_factual_match_ignored")
                adjusted_label = "neutral"
                adjustment_reason = "factual_exact_match"
            else:
                if "nli_contradiction" not in fused_flags:
                    fused_flags.append("nli_contradiction")
                return "unsupported", min(evidence_support, 0.20), fused_flags, adjusted_label, adjustment_reason, None

        if any(flag in critical_flags for flag in fused_flags):
            if "numeric_mismatch_with_evidence" in fused_flags:
                return "unsupported", min(evidence_support, 0.25), fused_flags, adjusted_label, adjustment_reason, None
            if "entity_mismatch_with_evidence" in fused_flags:
                return "unsupported", min(evidence_support, 0.30), fused_flags, adjusted_label, adjustment_reason, None
            return "unsupported", min(evidence_support, 0.35), fused_flags, adjusted_label, adjustment_reason, None

        non_adjustment_flags = [
            flag for flag in fused_flags
            if flag not in {"nli_contradiction_low_relevance_ignored", "nli_contradiction_factual_match_ignored"}
        ]
        if non_adjustment_flags:
            return "unsupported", min(evidence_support, 0.40), fused_flags, adjusted_label, adjustment_reason, None

        nli_support = self._nli_support_value(
            adjusted_label,
            nli_result,
            evidence_support=evidence_support,
            semantic_score=semantic_component,
            lexical_score=lexical_component,
        )
        composite_score = round(
            0.30 * semantic_component + 0.25 * lexical_component + 0.45 * nli_support,
            4,
        )

        # Floors for very strong direct evidence. These do not bypass mismatch
        # flags; they only prevent neutral/unavailable NLI from creating false
        # unsupported labels for obvious evidence matches.
        if exact_component >= 0.90 or factual_exact_match:
            composite_score = max(composite_score, 0.82)
        elif evidence_support >= 0.78 and lexical_component >= 0.35:
            composite_score = max(composite_score, 0.72)
        elif evidence_support >= 0.60 and lexical_component >= 0.25:
            composite_score = max(composite_score, 0.50)

        label = self._label_from_similarity(composite_score)
        if adjusted_label == "neutral" and "nli_neutral" not in fused_flags:
            fused_flags.append("nli_neutral")
        if adjusted_label in {"unavailable", "error", "not_run", "not_enough_input"} and "nli_unavailable_or_not_run" not in fused_flags:
            fused_flags.append("nli_unavailable_or_not_run")
        if adjusted_label == "entailment" and nli_score is not None and nli_score >= self.nli_entailment_threshold:
            label = "supported" if composite_score >= self.warning_threshold else "weak_support"

        return label, composite_score, fused_flags, adjusted_label, adjustment_reason, composite_score

    def detect(self, answer: str, evidence_list: list[Any]) -> dict:
        claims = self.extractor.extract_claims(answer)
        claim_results: list[dict] = []

        for claim in claims:
            score_info = self.scorer.score_claim_against_evidence(claim, evidence_list)
            score = float(score_info["score"])
            raw_similarity_score = float(score_info.get("raw_similarity_score") or score)
            semantic_score = float(score_info.get("semantic_score") or raw_similarity_score)
            lexical_score = float(score_info.get("lexical_score") or 0.0)
            exact_score = float(score_info.get("exact_score") or 0.0)
            best_index = score_info["best_evidence_index"]
            best_evidence_text = score_info.get("best_evidence") or ""
            nli_evidence_text = score_info.get("nli_evidence_text") or best_evidence_text
            rule_flags = list(score_info.get("rule_flags", []))

            similarity_label = self._label_from_similarity(score)
            if getattr(self.nli_verifier, "enabled", False) and (raw_similarity_score >= self.nli_min_similarity_to_run or rule_flags):
                nli_result = self.nli_verifier.verify(
                    normalize_for_detection(claim),
                    normalize_for_detection(nli_evidence_text),
                )
            else:
                nli_result = {
                    "enabled": getattr(self.nli_verifier, "enabled", False),
                    "available": False,
                    "label": "not_run",
                    "score": None,
                    "scores": {},
                    "error": None,
                }
            final_label, final_score, fused_flags, adjusted_nli_label, adjustment_reason, composite_score = self._fuse_decision(
                score,
                raw_similarity_score,
                similarity_label,
                rule_flags,
                nli_result,
                normalize_for_detection(claim),
                normalize_for_detection(nli_evidence_text),
                bool(score_info.get("factual_exact_match")),
                semantic_score=semantic_score,
                lexical_score=lexical_score,
                exact_score=exact_score,
            )
            span_result = highlight_hallucinated_spans(claim, best_evidence_text, fused_flags)
            entailment_score = self._nli_support_value(
                adjusted_nli_label,
                nli_result,
                evidence_support=score,
                semantic_score=semantic_score,
                lexical_score=lexical_score,
            )

            claim_results.append(
                {
                    "claim": claim,
                    "support_score": round(float(final_score), 4),
                    "raw_similarity_score": score_info.get("raw_similarity_score"),
                    "semantic_score": round(float(semantic_score), 4),
                    "lexical_score": round(float(lexical_score), 4),
                    "entailment_score": round(float(entailment_score), 4),
                    "exact_score": round(float(exact_score), 4),
                    "evidence_support_score": score_info.get("evidence_support_score"),
                    "composite_support_score": composite_score,
                    "label": final_label,
                    "similarity_label": similarity_label,
                    "best_evidence_index": best_index,
                    "best_evidence_text": best_evidence_text,
                    "nli_evidence_text": nli_evidence_text,
                    "rule_flags": fused_flags,
                    "factual_consistency": score_info.get("factual_consistency"),
                    "nli_label": nli_result.get("label"),
                    "nli_adjusted_label": adjusted_nli_label,
                    "nli_adjustment_reason": adjustment_reason,
                    "nli_score": nli_result.get("score"),
                    "best_single_score": score_info.get("best_single_score"),
                    "combined_context_score": score_info.get("combined_context_score"),
                    "used_combined_evidence": score_info.get("used_combined_evidence"),
                    "factual_exact_match": score_info.get("factual_exact_match"),
                    "definition_concept_matches": score_info.get("definition_concept_matches", []),
                    "nli_available": nli_result.get("available"),
                    "nli_error": nli_result.get("error"),
                    "highlighted_claim": span_result["highlighted_claim"],
                    "hallucinated_spans": span_result["hallucinated_spans"],
                }
            )

        claim_count = len(claim_results)
        supported = sum(item["label"] == "supported" for item in claim_results)
        weak = sum(item["label"] == "weak_support" for item in claim_results)
        unsupported = sum(item["label"] == "unsupported" for item in claim_results)
        support_ratio = round(supported / claim_count, 4) if claim_count else 0.0
        hallucination_rate = round(unsupported / claim_count, 4) if claim_count else 0.0
        average_score = round(sum(item["support_score"] for item in claim_results) / claim_count, 4) if claim_count else 0.0

        return {
            "claims": claim_results,
            "claim_count": claim_count,
            "supported_count": supported,
            "weak_count": weak,
            "unsupported_count": unsupported,
            "support_ratio": support_ratio,
            "hallucination_rate": hallucination_rate,
            "average_support_score": average_score,
        }
