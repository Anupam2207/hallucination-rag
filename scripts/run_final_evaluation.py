"""Run final local evaluation and generate paper-ready artifacts.

This script keeps the original CLI flags while defaulting to reproducible
manual-answer evaluation from ``data/evaluation/final_eval_set.jsonl``. It does
not require Ollama for the final tables.
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_retrieval import evaluate_retrieval_records
from src.evaluation.span_metrics import span_iou
from scripts.evaluation_common import (
    evaluate_correction_records,
    evaluate_detection_records,
    load_jsonl,
    write_confusion_matrix_png,
    write_csv,
    write_json,
)


def write_qualitative_cases(detection_rows: List[Dict[str, Any]], correction_rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    successes = [row for row in correction_rows if row.get("correction_success") or float(row.get("hallucination_reduction", 0.0) or 0.0) > 0]
    failures = [row for row in correction_rows if row.get("unsafe_correction") or row.get("malformed_correction") or float(row.get("hallucination_reduction", 0.0) or 0.0) < 0]
    if not failures:
        failures = [row for row in correction_rows if not row.get("correction_success")][:5]

    success_lines = ["# Qualitative success cases", ""]
    for row in successes[:8]:
        success_lines.extend(
            [
                f"## {row.get('id', '')}: {row.get('query', '')}",
                "",
                f"Raw answer: {row.get('raw_answer', '')}",
                "",
                f"Corrected answer: {row.get('corrected_answer', '')}",
                "",
                f"Hallucination reduction: {row.get('hallucination_reduction', 0.0)}",
                "",
            ]
        )
    (output_dir / "qualitative_success_cases.md").write_text("\n".join(success_lines), encoding="utf-8")

    failure_lines = ["# Qualitative failure cases", ""]
    for row in failures[:8]:
        failure_lines.extend(
            [
                f"## {row.get('id', '')}: {row.get('query', '')}",
                "",
                f"Raw answer: {row.get('raw_answer', '')}",
                "",
                f"Corrected answer: {row.get('corrected_answer', '')}",
                "",
                f"Status: {row.get('correction_status', '')}",
                "",
            ]
        )
    (output_dir / "qualitative_failure_cases.md").write_text("\n".join(failure_lines), encoding="utf-8")


def main() -> None:
    parser = ArgumentParser(description="Run final benchmark and generate paper artifacts.")
    parser.add_argument("--input", default="data/evaluation/final_eval_set.jsonl")
    parser.add_argument("--output-dir", default="paper_artifacts")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retrieval", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--nli", choices=["on", "off"], default="off")
    parser.add_argument("--correction", choices=["on", "off"], default="on")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    records = load_jsonl(PROJECT_ROOT / args.input)
    if args.limit is not None:
        records = records[: args.limit]

    enable_nli = args.nli == "on"
    detection_rows, detection_summary = evaluate_detection_records(records, enable_nli=enable_nli, rules=True)
    # Kept for backward-compatible evaluation scaffolds. The current detector is
    # claim-level, so span_iou is reported only when explicit predicted spans
    # are added by downstream experiments.
    detection_summary["span_iou"] = round(
        sum(span_iou([], record.get("gold_hallucinated_spans", []) or []) for record in records) / len(records), 4
    ) if records else 0.0
    retrieval_rows, retrieval_summary = evaluate_retrieval_records(records, mode=args.retrieval, top_k=args.top_k)
    correction_rows: list[dict[str, Any]] = []
    correction_summary: dict[str, Any] = {"records": len(records), "correction": args.correction}
    if args.correction == "on":
        correction_rows, correction_summary = evaluate_correction_records(records, enable_nli=enable_nli)

    artifact_dir = PROJECT_ROOT / args.output_dir
    tables_dir = artifact_dir / "tables"
    figures_dir = artifact_dir / "figures"
    qualitative_dir = artifact_dir / "qualitative_examples"
    metrics_dir = artifact_dir / "metrics"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    for directory in (tables_dir, figures_dir, qualitative_dir, metrics_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    write_csv(tables_dir / "final_detection_results.csv", detection_rows)
    write_json(metrics_dir / "final_detection_summary.json", detection_summary)
    write_csv(tables_dir / "retrieval_results.csv", retrieval_rows)
    write_json(metrics_dir / "retrieval_summary.json", retrieval_summary)
    write_confusion_matrix_png(detection_summary["confusion_matrix"], figures_dir / "confusion_matrix.png")
    write_csv(reports_dir / "final_detection_results.csv", detection_rows)
    write_json(reports_dir / "final_detection_summary.json", detection_summary)
    write_csv(reports_dir / "retrieval_results.csv", retrieval_rows)
    write_json(reports_dir / "retrieval_summary.json", retrieval_summary)

    if args.correction == "on":
        write_csv(tables_dir / "final_correction_results.csv", correction_rows)
        write_json(metrics_dir / "final_correction_summary.json", correction_summary)
        write_csv(reports_dir / "final_correction_results.csv", correction_rows)
        write_json(reports_dir / "final_correction_summary.json", correction_summary)
        write_qualitative_cases(detection_rows, correction_rows, qualitative_dir)

    combined_summary = {
        "records": len(records),
        "retrieval": retrieval_summary,
        "detection": detection_summary,
        "correction": correction_summary,
    }
    write_json(metrics_dir / "final_evaluation_summary.json", combined_summary)
    print(json.dumps(combined_summary, indent=2))


if __name__ == "__main__":
    main()
