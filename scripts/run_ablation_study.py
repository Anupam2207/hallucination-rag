"""Run local ablation experiments for retrieval, detection, NLI, rules, and correction."""
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
from scripts.evaluation_common import (
    evaluate_correction_records,
    evaluate_detection_records,
    load_jsonl,
    write_csv,
    write_json,
)


def main() -> None:
    parser = ArgumentParser(description="Run ablation study and write paper-ready tables.")
    parser.add_argument("--input", default="data/evaluation/final_eval_set.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--enable-real-nli", action="store_true", help="Load the configured NLI model for NLI ablations. Heavy on laptops.")
    parser.add_argument("--output-dir", default="paper_artifacts")
    args = parser.parse_args()

    records = load_jsonl(PROJECT_ROOT / args.input)
    if args.limit is not None:
        records = records[: args.limit]

    rows: List[Dict[str, Any]] = []

    for retrieval_mode in ("bm25", "dense", "hybrid"):
        retrieval_rows, retrieval_summary = evaluate_retrieval_records(records, mode=retrieval_mode, top_k=args.top_k)
        rows.append(
            {
                "component": "retrieval",
                "variant": f"{retrieval_mode}-only retrieval" if retrieval_mode != "hybrid" else "hybrid retrieval",
                "recall_at_k": retrieval_summary.get("recall_at_k", 0.0),
                "mrr": retrieval_summary.get("mrr", 0.0),
                "evidence_keyword_hit_rate": retrieval_summary.get("evidence_keyword_hit_rate", 0.0),
                "answerability_retrieval_success": retrieval_summary.get("answerability_retrieval_success", 0.0),
                "notes": "",
            }
        )

    detection_variants = [
        ("similarity-only detection", False, False),
        ("similarity + factual rules", False, True),
        ("similarity + NLI", args.enable_real_nli, False),
        ("similarity + factual rules + NLI", args.enable_real_nli, True),
    ]
    for name, enable_nli, rules in detection_variants:
        _det_rows, summary = evaluate_detection_records(records, enable_nli=enable_nli, rules=rules)
        rows.append(
            {
                "component": "detection",
                "variant": name,
                "claim_accuracy": summary.get("accuracy", 0.0),
                "precision": summary.get("precision", 0.0),
                "recall": summary.get("recall", 0.0),
                "f1": summary.get("f1", 0.0),
                "macro_f1": summary.get("macro_f1", 0.0),
                "unsupported_recall": summary.get("unsupported_recall", 0.0),
                "notes": "real NLI enabled" if enable_nli else ("NLI disabled/fallback for demo" if "NLI" in name else ""),
            }
        )

    _correction_rows, correction_summary = evaluate_correction_records(records, enable_nli=args.enable_real_nli)
    rows.append(
        {
            "component": "full_system",
            "variant": "full system with correction",
            "correction_success_rate": correction_summary.get("correction_success_rate", 0.0),
            "hallucination_reduction": correction_summary.get("hallucination_reduction", 0.0),
            "support_improvement": correction_summary.get("support_improvement", 0.0),
            "unsafe_correction_rate": correction_summary.get("unsafe_correction_rate", 0.0),
            "no_change_rate": correction_summary.get("no_change_rate", 0.0),
            "notes": "deterministic evidence-only correction for reproducible local evaluation",
        }
    )

    artifact_dir = PROJECT_ROOT / args.output_dir
    tables_dir = artifact_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "records": len(records),
        "real_nli_enabled": bool(args.enable_real_nli),
        "variants": rows,
        "notes": "By default, NLI ablations use the safe disabled/fallback path to keep demo mode laptop-friendly. Use --enable-real-nli for research runs.",
    }
    write_csv(tables_dir / "ablation_results.csv", rows)
    write_json(tables_dir / "ablation_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
