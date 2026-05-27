from src.evaluation.metrics import HallucinationMetrics


def test_mostly_weak_corrected_answer_is_partially_improved() -> None:
    raw = {
        "claims": [{"label": "unsupported", "rule_flags": []}],
        "claim_count": 1,
        "supported_count": 0,
        "weak_count": 0,
        "unsupported_count": 1,
    }
    corrected = {
        "claims": [
            {"label": "weak_support", "rule_flags": []},
            {"label": "weak_support", "rule_flags": []},
            {"label": "supported", "rule_flags": []},
        ],
        "claim_count": 3,
        "supported_count": 1,
        "weak_count": 2,
        "unsupported_count": 0,
    }
    metrics = HallucinationMetrics.summarize(raw, corrected)
    assert metrics["correction_status"] == "partially_improved"
    assert metrics["correction_success"] is False


def test_critical_unsupported_corrected_answer_is_unsafe() -> None:
    raw = {"claims": [], "claim_count": 0, "supported_count": 0, "weak_count": 0, "unsupported_count": 0}
    corrected = {
        "claims": [{"label": "unsupported", "rule_flags": ["collaborative_filtering_not_in_evidence"]}],
        "claim_count": 1,
        "supported_count": 0,
        "weak_count": 0,
        "unsupported_count": 1,
    }
    metrics = HallucinationMetrics.summarize(raw, corrected)
    assert metrics["correction_status"] == "unsafe"
    assert metrics["correction_success"] is False
