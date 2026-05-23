from src.evaluation.metrics import HallucinationMetrics


def test_unsafe_correction_status_when_critical_claim_remains() -> None:
    raw = {"claims": [], "claim_count": 0, "supported_count": 0, "weak_count": 0, "unsupported_count": 0}
    corrected = {
        "claims": [
            {"label": "unsupported", "rule_flags": ["claim_year_not_supported_by_evidence"]}
        ],
        "claim_count": 1,
        "supported_count": 0,
        "weak_count": 0,
        "unsupported_count": 1,
    }
    metrics = HallucinationMetrics.summarize(raw, corrected)
    assert metrics["correction_status"] == "unsafe"
    assert metrics["correction_success"] is False
    assert metrics["corrected_critical_unsupported_count"] == 1


def test_no_change_is_not_success() -> None:
    raw = {"claims": [], "claim_count": 0, "supported_count": 0, "weak_count": 0, "unsupported_count": 0}
    corrected = {"claims": [], "claim_count": 0, "supported_count": 0, "weak_count": 0, "unsupported_count": 0}
    metrics = HallucinationMetrics.summarize(raw, corrected)
    assert metrics["correction_status"] == "no_change"
    assert metrics["correction_success"] is False
