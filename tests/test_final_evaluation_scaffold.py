import json
from pathlib import Path


def test_final_eval_set_has_minimum_schema_and_coverage():
    path = Path("data/evaluation/final_eval_set.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 15
    topics = " ".join(row["query"].lower() for row in rows)
    for keyword in ["rag", "colbert", "truthfulqa", "hallucination", "vehicle", "smartphone", "newsroom"]:
        assert keyword in topics
    required = {
        "query",
        "expected_answer_type",
        "gold_supported_claims",
        "gold_unsupported_claims",
        "gold_hallucinated_spans",
        "expected_evidence_keywords",
    }
    for row in rows:
        assert required.issubset(row)
        assert isinstance(row["gold_supported_claims"], list)
        assert isinstance(row["gold_unsupported_claims"], list)
        assert isinstance(row["gold_hallucinated_spans"], list)


def test_run_final_evaluation_script_exists():
    path = Path("scripts/run_final_evaluation.py")
    text = path.read_text(encoding="utf-8")
    assert "--retrieval" in text
    assert "--nli" in text
    assert "--correction" in text
    assert "span_iou" in text
