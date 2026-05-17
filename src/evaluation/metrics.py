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
        raw_count = detection_result_count(raw_detection)
        corrected_count = detection_result_count(corrected_detection)
        return raw_count - corrected_count

    @staticmethod
    def factual_improvement(raw_detection: Dict, corrected_detection: Dict) -> float:
        raw_score = HallucinationMetrics.weighted_support_ratio(raw_detection)
        corrected_score = HallucinationMetrics.weighted_support_ratio(corrected_detection)
        return round(corrected_score - raw_score, 4)

    @staticmethod
    def correction_status(
        factual_improvement: float,
        hallucination_reduction: float,
    ) -> str:
        """Return a human-readable correction status.

        The UI should not show strong success when both key metrics are unchanged.
        True success only means factual support improved or hallucination rate
        reduced. If one metric improves but the other worsens, call it partial.
        """
        eps = 1e-9
        improved_support = factual_improvement > eps
        reduced_hallucination = hallucination_reduction > eps
        worsened_support = factual_improvement < -eps
        increased_hallucination = hallucination_reduction < -eps

        if (improved_support or reduced_hallucination) and not (
            worsened_support or increased_hallucination
        ):
            return "improved"
        if improved_support or reduced_hallucination:
            return "partially_improved"
        if not worsened_support and not increased_hallucination:
            return "no_change"
        return "worsened"

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
        status = HallucinationMetrics.correction_status(improvement, hallucination_reduction)

        # True success only when factual grounding improves or hallucination rate drops.
        correction_success = status in {"improved", "partially_improved"}

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
            "correction_status": status,
            "correction_success": bool(correction_success),
        }

    # Backward-compatible alias. Older pipeline code called summary().
    @staticmethod
    def summary(raw_detection: Dict, corrected_detection: Dict) -> Dict:
        return HallucinationMetrics.summarize(raw_detection, corrected_detection)


def detection_result_count(detection_result: Dict) -> int:
    return int(detection_result.get("claim_count", 0) or 0)


def compare_detection_results(raw_detection: Dict, corrected_detection: Dict) -> Dict:
    """Functional wrapper used by the batch evaluator."""
    return HallucinationMetrics.summarize(raw_detection, corrected_detection)
