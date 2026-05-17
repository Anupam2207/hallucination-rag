from typing import Any

from src.config import get_config_value
from src.detection.claim_extractor import ClaimExtractor
from src.detection.nli_verifier import NLIVerifier
from src.detection.support_scorer import SupportScorer
from src.retrieval.embedder import EmbeddingModel


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
            get_config_value(
                "settings",
                "detection",
                "similarity_support_threshold",
                default=0.55,
            )
        )
        self.warning_threshold = float(
            get_config_value(
                "settings",
                "detection",
                "similarity_warning_threshold",
                default=0.40,
            )
        )
        self.nli_entailment_threshold = float(
            get_config_value(
                "settings",
                "detection",
                "nli_entailment_threshold",
                default=0.60,
            )
        )
        self.nli_contradiction_threshold = float(
            get_config_value(
                "settings",
                "detection",
                "nli_contradiction_threshold",
                default=0.60,
            )
        )

    def _label_from_similarity(self, score: float) -> str:
        if score >= self.support_threshold:
            return "supported"
        if score >= self.warning_threshold:
            return "weak_support"
        return "unsupported"

    def _fuse_label(self, similarity_label: str, rule_flags: list[str], nli_result: dict) -> str:
        if rule_flags:
            return "unsupported"

        nli_label = nli_result.get("label")
        nli_score = nli_result.get("score")
        nli_score = float(nli_score) if nli_score is not None else None

        if nli_label == "contradiction" and nli_score is not None and nli_score >= self.nli_contradiction_threshold:
            return "unsupported"
        if nli_label == "entailment" and nli_score is not None and nli_score >= self.nli_entailment_threshold:
            # NLI confirms support; promote weak related evidence to supported.
            if similarity_label in {"supported", "weak_support"}:
                return "supported"
        if nli_label == "neutral" and similarity_label == "supported":
            # Similarity says related, but NLI cannot entail it.
            return "weak_support"

        return similarity_label

    def detect(self, answer: str, evidence_list: list[Any]) -> dict:
        claims = self.extractor.extract_claims(answer)
        claim_results: list[dict] = []

        for claim in claims:
            score_info = self.scorer.score_claim_against_evidence(claim, evidence_list)
            score = float(score_info["score"])
            best_index = score_info["best_evidence_index"]
            best_evidence_text = score_info.get("best_evidence") or ""
            rule_flags = score_info.get("rule_flags", [])

            similarity_label = self._label_from_similarity(score)
            nli_result = self.nli_verifier.verify(claim, best_evidence_text)
            final_label = self._fuse_label(similarity_label, rule_flags, nli_result)

            claim_results.append(
                {
                    "claim": claim,
                    "support_score": round(float(score), 4),
                    "raw_similarity_score": score_info.get("raw_similarity_score"),
                    "label": final_label,
                    "similarity_label": similarity_label,
                    "best_evidence_index": best_index,
                    "best_evidence_text": best_evidence_text,
                    "rule_flags": rule_flags,
                    "nli_label": nli_result.get("label"),
                    "nli_score": nli_result.get("score"),
                    "nli_available": nli_result.get("available"),
                    "nli_error": nli_result.get("error"),
                }
            )

        claim_count = len(claim_results)
        supported = sum(item["label"] == "supported" for item in claim_results)
        weak = sum(item["label"] == "weak_support" for item in claim_results)
        unsupported = sum(item["label"] == "unsupported" for item in claim_results)

        support_ratio = round(supported / claim_count, 4) if claim_count else 0.0
        hallucination_rate = round(unsupported / claim_count, 4) if claim_count else 0.0
        average_score = (
            round(sum(item["support_score"] for item in claim_results) / claim_count, 4)
            if claim_count
            else 0.0
        )

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
