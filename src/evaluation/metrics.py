from typing import Dict, Iterable


CRITICAL_UNSUPPORTED_FLAGS = {
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


class HallucinationMetrics:
    """Before/after metrics for claim-level hallucination detection."""

    @staticmethod
    def support_ratio(detection_result: Dict) -> float:
        claim_count = detection_result.get("claim_count", 0)
        if claim_count == 0:
            return 0.0
        return round(detection_result.get("supported_count", 0) / claim_count, 4)

    @staticmethod
    def weak_support_ratio(detection_result: Dict) -> float:
        claim_count = detection_result.get("claim_count", 0)
        if claim_count == 0:
            return 0.0
        return round(detection_result.get("weak_count", 0) / claim_count, 4)

    @staticmethod
    def weighted_support_ratio(detection_result: Dict) -> float:
        claim_count = detection_result.get("claim_count", 0)
        if claim_count == 0:
            return 0.0
        supported = detection_result.get("supported_count", 0)
        weak = detection_result.get("weak_count", 0)
        return round((supported + 0.5 * weak) / claim_count, 4)

    @staticmethod
    def hallucination_rate(detection_result: Dict) -> float:
        claim_count = detection_result.get("claim_count", 0)
        if claim_count == 0:
            return 0.0
        return round(detection_result.get("unsupported_count", 0) / claim_count, 4)

    @staticmethod
    def critical_unsupported_count(detection_result: Dict) -> int:
        count = 0
        for claim in detection_result.get("claims", []) or []:
            flags = set(claim.get("rule_flags", []) or [])
            if claim.get("label") == "unsupported" and flags.intersection(CRITICAL_UNSUPPORTED_FLAGS):
                count += 1
        return count

    @staticmethod
    def claim_count_reduction(raw_detection: Dict, corrected_detection: Dict) -> int:
        return detection_result_count(raw_detection) - detection_result_count(corrected_detection)

    @staticmethod
    def factual_improvement(raw_detection: Dict, corrected_detection: Dict) -> float:
        return round(
            HallucinationMetrics.weighted_support_ratio(corrected_detection)
            - HallucinationMetrics.weighted_support_ratio(raw_detection),
            4,
        )

    @staticmethod
    def correction_status(
        factual_improvement: float,
        hallucination_reduction: float,
        corrected_critical_unsupported_count: int = 0,
        raw_critical_unsupported_count: int = 0,
    ) -> str:
        eps = 1e-9
        if corrected_critical_unsupported_count > 0:
            return "unsafe"

        improved_support = factual_improvement > eps
        reduced_hallucination = hallucination_reduction > eps
        worsened_support = factual_improvement < -eps
        increased_hallucination = hallucination_reduction < -eps
        reduced_critical = raw_critical_unsupported_count > 0 and corrected_critical_unsupported_count == 0

        if (improved_support or reduced_hallucination or reduced_critical) and not (
            worsened_support or increased_hallucination
        ):
            return "improved"
        if improved_support or reduced_hallucination or reduced_critical:
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
        raw_critical = HallucinationMetrics.critical_unsupported_count(raw_detection)
        corrected_critical = HallucinationMetrics.critical_unsupported_count(corrected_detection)
        status = HallucinationMetrics.correction_status(
            improvement,
            hallucination_reduction,
            corrected_critical,
            raw_critical,
        )
        correction_safety_passed = corrected_critical == 0
        correction_success = status in {"improved", "partially_improved"} and correction_safety_passed

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
            "raw_critical_unsupported_count": raw_critical,
            "corrected_critical_unsupported_count": corrected_critical,
            "critical_unsupported_count": corrected_critical,
            "correction_safety_passed": correction_safety_passed,
            "correction_status": status,
            "correction_success": bool(correction_success),
        }

    @staticmethod
    def summary(raw_detection: Dict, corrected_detection: Dict) -> Dict:
        return HallucinationMetrics.summarize(raw_detection, corrected_detection)


def detection_result_count(detection_result: Dict) -> int:
    return int(detection_result.get("claim_count", 0) or 0)


def compare_detection_results(raw_detection: Dict, corrected_detection: Dict) -> Dict:
    return HallucinationMetrics.summarize(raw_detection, corrected_detection)
