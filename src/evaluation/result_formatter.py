from typing import Any, Dict


def flatten_pipeline_result(result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result.get("metrics", {})
    return {
        "query": result.get("query", ""),
        "raw_answer": result.get("raw_answer", ""),
        "corrected_answer": result.get("corrected_answer", ""),
        "evidence_count": len(result.get("evidence", [])),
        "raw_support_ratio": metrics.get("raw_support_ratio", 0.0),
        "corrected_support_ratio": metrics.get("corrected_support_ratio", 0.0),
        "raw_hallucination_rate": metrics.get("raw_hallucination_rate", 0.0),
        "corrected_hallucination_rate": metrics.get("corrected_hallucination_rate", 0.0),
        "factual_improvement": metrics.get("factual_improvement", 0.0),
        "correction_success": metrics.get("correction_success", False),
    }
