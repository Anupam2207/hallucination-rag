from pathlib import Path

from scripts.evaluation_common import (
    evaluate_correction_records,
    evaluate_detection_records,
    load_jsonl,
    write_confusion_matrix_png,
)


def test_detection_evaluation_reports_required_metrics(tmp_path):
    records = load_jsonl(Path("data/evaluation/final_eval_set.jsonl"))[:5]
    rows, summary = evaluate_detection_records(records, enable_nli=False, rules=True)
    assert rows
    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "confusion_matrix",
        "unsupported_recall",
        "false_positive_rate",
        "false_negative_rate",
    ]:
        assert key in summary
    out = tmp_path / "confusion_matrix.png"
    write_confusion_matrix_png(summary["confusion_matrix"], out)
    assert out.exists() and out.stat().st_size > 0


def test_correction_evaluation_reports_required_metrics():
    records = load_jsonl(Path("data/evaluation/final_eval_set.jsonl"))[:5]
    rows, summary = evaluate_correction_records(records, enable_nli=False)
    assert rows
    for key in [
        "correction_success_rate",
        "hallucination_reduction",
        "support_improvement",
        "unsafe_correction_rate",
        "corrected_unsupported_claim_count",
        "corrected_critical_unsupported_claim_count",
        "no_change_rate",
        "malformed_correction_rate",
    ]:
        assert key in summary
