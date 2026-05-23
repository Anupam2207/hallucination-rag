from typing import Any

from src.config import get_config_value
from src.detection.claim_extractor import ClaimExtractor
from src.detection.nli_verifier import NLIVerifier
from src.detection.span_highlighter import highlight_hallucinated_spans
from src.detection.support_scorer import SupportScorer
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
            get_config_value("settings", "detection", "similarity_support_threshold", default=0.55)
        )
        self.warning_threshold = float(
            get_config_value("settings", "detection", "similarity_warning_threshold", default=0.40)
        )
        self.nli_entailment_threshold = float(
            get_config_value("settings", "detection", "nli_entailment_threshold", default=0.60)
        )
        self.nli_contradiction_threshold = float(
            get_config_value("settings", "detection", "nli_contradiction_threshold", default=0.60)
        )
        self.nli_min_similarity_to_run = float(
            get_config_value("settings", "verification", "nli_min_similarity_to_run", default=0.35)
        )

    def _label_from_similarity(self, score: float) -> str:
        if score >= self.support_threshold:
            return "supported"
        if score >= self.warning_threshold:
            return "weak_support"
        return "unsupported"

    @staticmethod
    def critical_rule_flags() -> set[str]:
        return {
            "nli_contradiction",
            "numeric_mismatch_with_evidence",
            "claim_year_not_supported_by_evidence",
            "claim_date_not_supported_by_evidence",
            "claim_numeric_not_supported_by_evidence",
            "entity_mismatch_with_evidence",
            "fine_tuning_not_in_evidence",
            "unsupported_task_example_not_in_evidence",
            "training_data_requirement_not_in_evidence",
        }

    def _fuse_decision(self, score: float, similarity_label: str, rule_flags: list[str], nli_result: dict) -> tuple[str, float, list[str]]:
        fused_flags = list(dict.fromkeys(rule_flags))
        final_score = float(score)
        nli_label = nli_result.get("label")
        nli_score = nli_result.get("score")
        nli_score = float(nli_score) if nli_score is not None else None

        # 1. NLI contradiction is the strongest signal when available.
        if nli_label == "contradiction" and nli_score is not None and nli_score >= self.nli_contradiction_threshold:
            if "nli_contradiction" not in fused_flags:
                fused_flags.append("nli_contradiction")
            return "unsupported", min(final_score, 0.20), fused_flags

        # 2. Critical factual/rule flags override cosine similarity.
        if any(flag in self.critical_rule_flags() for flag in fused_flags):
            if "numeric_mismatch_with_evidence" in fused_flags:
                return "unsupported", min(final_score, 0.25), fused_flags
            if "entity_mismatch_with_evidence" in fused_flags:
                return "unsupported", min(final_score, 0.30), fused_flags
            return "unsupported", min(final_score, 0.35), fused_flags

        # 3. Other rule flags indicate unsupported details.
        if fused_flags:
            return "unsupported", min(final_score, 0.35), fused_flags

        # 4. Entailment can promote a weak semantic match.
        if nli_label == "entailment" and nli_score is not None and nli_score >= self.nli_entailment_threshold:
            if similarity_label in {"supported", "weak_support"}:
                return "supported", max(final_score, 0.70), fused_flags

        # 5. Neutral means related but not proven.
        if nli_label == "neutral":
            if "nli_neutral" not in fused_flags:
                fused_flags.append("nli_neutral")
            if similarity_label == "supported":
                return "weak_support", min(final_score, 0.49), fused_flags
            return similarity_label, min(final_score, 0.49), fused_flags

        return similarity_label, final_score, fused_flags

    def detect(self, answer: str, evidence_list: list[Any]) -> dict:
        claims = self.extractor.extract_claims(answer)
        claim_results: list[dict] = []

        for claim in claims:
            score_info = self.scorer.score_claim_against_evidence(claim, evidence_list)
            score = float(score_info["score"])
            best_index = score_info["best_evidence_index"]
            best_evidence_text = score_info.get("best_evidence") or ""
            rule_flags = list(score_info.get("rule_flags", []))

            similarity_label = self._label_from_similarity(score)
            if getattr(self.nli_verifier, "enabled", False) and (score >= self.nli_min_similarity_to_run or rule_flags):
                nli_result = self.nli_verifier.verify(normalize_for_detection(claim), normalize_for_detection(best_evidence_text))
            else:
                nli_result = {"enabled": getattr(self.nli_verifier, "enabled", False), "available": False, "label": "not_run", "score": None, "scores": {}, "error": None}
            final_label, final_score, fused_flags = self._fuse_decision(score, similarity_label, rule_flags, nli_result)
            span_result = highlight_hallucinated_spans(claim, best_evidence_text, fused_flags)

            claim_results.append(
                {
                    "claim": claim,
                    "support_score": round(float(final_score), 4),
                    "raw_similarity_score": score_info.get("raw_similarity_score"),
                    "label": final_label,
                    "similarity_label": similarity_label,
                    "best_evidence_index": best_index,
                    "best_evidence_text": best_evidence_text,
                    "rule_flags": fused_flags,
                    "factual_consistency": score_info.get("factual_consistency"),
                    "nli_label": nli_result.get("label"),
                    "nli_score": nli_result.get("score"),
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
