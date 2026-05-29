"""Evaluate deterministic evidence-only correction safety and improvement."""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation_common import evaluate_correction_records, load_jsonl, write_csv, write_json


def main() -> None:
    parser = ArgumentParser(description="Evaluate correction success, safety, and post-correction support.")
    parser.add_argument("--input", default="data/evaluation/final_eval_set.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--nli", choices=["on", "off"], default="off")
    parser.add_argument("--output-dir", default="paper_artifacts")
    args = parser.parse_args()

    records = load_jsonl(PROJECT_ROOT / args.input)
    if args.limit is not None:
        records = records[: args.limit]
    rows, summary = evaluate_correction_records(records, enable_nli=(args.nli == "on"))

    artifact_dir = PROJECT_ROOT / args.output_dir
    tables_dir = artifact_dir / "tables"
    metrics_dir = artifact_dir / "metrics"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    for path in (tables_dir, metrics_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    write_csv(tables_dir / "final_correction_results.csv", rows)
    write_json(metrics_dir / "final_correction_summary.json", summary)
    write_csv(reports_dir / "final_correction_results.csv", rows)
    write_json(reports_dir / "final_correction_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
