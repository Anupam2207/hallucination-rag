from src.evaluation.metrics import HallucinationMetrics


def test_all_weak_corrected_answer_not_success():
    raw = {"claims": [{"label": "unsupported", "rule_flags": []}], "claim_count": 1, "supported_count": 0, "weak_count": 0, "unsupported_count": 1}
    corrected = {"claims": [{"label": "weak_support", "rule_flags": []}], "claim_count": 1, "supported_count": 0, "weak_count": 1, "unsupported_count": 0}
    metrics = HallucinationMetrics.summarize(raw, corrected)
    assert metrics["correction_status"] == "partially_improved"
    assert metrics["correction_success"] is False
