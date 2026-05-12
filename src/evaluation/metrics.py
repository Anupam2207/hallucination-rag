from typing import Any, Dict


class HallucinationMetrics:
    @staticmethod
    def support_ratio(result: Dict[str, Any]) -> float:
        return float(result.get('support_ratio', 0.0))

    @staticmethod
    def hallucination_rate(result: Dict[str, Any]) -> float:
        return float(result.get('hallucination_rate', 0.0))

    @staticmethod
    def factual_improvement(before: Dict[str, Any], after: Dict[str, Any]) -> float:
        return HallucinationMetrics.support_ratio(after) - HallucinationMetrics.support_ratio(before)

    @staticmethod
    def correction_success(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
        improved_support = HallucinationMetrics.support_ratio(after) > HallucinationMetrics.support_ratio(before)
        reduced_hallucination = HallucinationMetrics.hallucination_rate(after) < HallucinationMetrics.hallucination_rate(before)
        return improved_support or reduced_hallucination

    @staticmethod
    def summary(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'raw_support_ratio': HallucinationMetrics.support_ratio(before),
            'corrected_support_ratio': HallucinationMetrics.support_ratio(after),
            'raw_hallucination_rate': HallucinationMetrics.hallucination_rate(before),
            'corrected_hallucination_rate': HallucinationMetrics.hallucination_rate(after),
            'factual_improvement': HallucinationMetrics.factual_improvement(before, after),
            'correction_success': HallucinationMetrics.correction_success(before, after),
        }


def compare_detection_results(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    summary = HallucinationMetrics.summary(before, after)
    summary['support_improvement'] = summary['factual_improvement']
    summary['hallucination_reduction'] = summary['raw_hallucination_rate'] - summary['corrected_hallucination_rate']
    return summary
