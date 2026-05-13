from typing import Dict


class HallucinationMetrics:
    """Lightweight before/after metrics for claim-level hallucination detection.

    Labels expected from the detector:
    - supported: fully supported by retrieved evidence
    - weak_support: related evidence exists, but support is not strong
    - unsupported: no sufficient retrieved support
    """

    @staticmethod
    def support_ratio(detection_result: Dict) -> float:
        claim_count = detection_result.get("claim_count", 0)
        if claim_count == 0:
            return 0.0
        supported = detection_result.get("supported_count", 0)
        return round(supported / claim_count, 4)

    @staticmethod
    def weak_support_ratio(detection_result: Dict) -> float:
        claim_count = detection_result.get("claim_count", 0)
        if claim_count == 0:
            return 0.0
        weak = detection_result.get("weak_count", 0)
        return round(weak / claim_count, 4)

    @staticmethod
    def weighted_support_ratio(detection_result: Dict) -> float:
        """Weighted support score.

        supported claims get full credit, weakly supported claims get partial
        credit, and unsupported claims get no credit. This is more informative
        than strict support ratio for a similarity-based baseline detector.
        """
        claim_count = detection_result.get("claim_count", 0)
        if claim_count == 0:
            return 0.0
        supported = detection_result.get("supported_count", 0)
        weak = detection_result.get("weak_count", 0)
        weighted_score = (supported + 0.5 * weak) / claim_count
        return round(weighted_score, 4)

    @staticmethod
    def hallucination_rate(detection_result: Dict) -> float:
        claim_count = detection_result.get("claim_count", 0)
        if claim_count == 0:
            return 0.0
        unsupported = detection_result.get("unsupported_count", 0)
        return round(unsupported / claim_count, 4)

    @staticmethod
    def claim_count_reduction(raw_detection: Dict, corrected_detection: Dict) -> int:
        raw_count = raw_detection.get("claim_count", 0)
        corrected_count = corrected_detection.get("claim_count", 0)
        return raw_count - corrected_count

    @staticmethod
    def factual_improvement(raw_detection: Dict, corrected_detection: Dict) -> float:
        raw_score = HallucinationMetrics.weighted_support_ratio(raw_detection)
        corrected_score = HallucinationMetrics.weighted_support_ratio(corrected_detection)
        return round(corrected_score - raw_score, 4)

    @staticmethod
    def summarize(raw_detection: Dict, corrected_detection: Dict) -> Dict:
        raw_support = HallucinationMetrics.support_ratio(raw_detection)
        corrected_support = HallucinationMetrics.support_ratio(corrected_detection)
        raw_weak_support = HallucinationMetrics.weak_support_ratio(raw_detection)
        corrected_weak_support = HallucinationMetrics.weak_support_ratio(corrected_detection)

        raw_weighted = HallucinationMetrics.weighted_support_ratio(raw_detection)
        corrected_weighted = HallucinationMetrics.weighted_support_ratio(corrected_detection)

        raw_hallucination = HallucinationMetrics.hallucination_rate(raw_detection)
        corrected_hallucination = HallucinationMetrics.hallucination_rate(corrected_detection)

        improvement = round(corrected_weighted - raw_weighted, 4)
        hallucination_reduction = round(raw_hallucination - corrected_hallucination, 4)
        claim_reduction = HallucinationMetrics.claim_count_reduction(raw_detection, corrected_detection)

        # The correction is successful when it improves weighted support, reduces
        # unsupported claims, or preserves support while producing a more concise
        # answer with no increase in hallucination rate.
        correction_success = (
            improvement > 0
            or hallucination_reduction > 0
            or (
                corrected_hallucination <= raw_hallucination
                and corrected_weighted >= raw_weighted
                and claim_reduction >= 0
            )
        )

        return {
            "raw_support_ratio": raw_support,
            "corrected_support_ratio": corrected_support,
            "raw_weak_support_ratio": raw_weak_support,
            "corrected_weak_support_ratio": corrected_weak_support,
            "raw_weighted_support_ratio": raw_weighted,
            "corrected_weighted_support_ratio": corrected_weighted,
            "raw_hallucination_rate": raw_hallucination,
            "corrected_hallucination_rate": corrected_hallucination,
            "hallucination_reduction": hallucination_reduction,
            "claim_count_reduction": claim_reduction,
            "factual_improvement": improvement,
            "correction_success": bool(correction_success),
        }

    # Backward-compatible alias. Older pipeline code called summary().
    @staticmethod
    def summary(raw_detection: Dict, corrected_detection: Dict) -> Dict:
        return HallucinationMetrics.summarize(raw_detection, corrected_detection)


def compare_detection_results(raw_detection: Dict, corrected_detection: Dict) -> Dict:
    """Functional wrapper used by the batch evaluator."""
    return HallucinationMetrics.summarize(raw_detection, corrected_detection)
