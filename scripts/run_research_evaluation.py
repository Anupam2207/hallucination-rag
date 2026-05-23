import csv
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.classification_metrics import faithfulness_score
from src.evaluation.span_metrics import span_iou
from src.pipeline import HallucinationRAGPipeline


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = ArgumentParser(description="Run lightweight research-style evaluation.")
    parser.add_argument("--input", default="data/evaluation/research_eval_set.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    records = load_jsonl(PROJECT_ROOT / args.input)
    if args.limit:
        records = records[: args.limit]

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "research_eval_results.csv"
    summary_path = output_dir / "research_eval_summary.json"

    pipeline = HallucinationRAGPipeline()
    rows = []
    for index, row in enumerate(records, start=1):
        query = row.get("query", "")
        result = pipeline.run(query)
        corrected_detection = result.get("corrected_detection", {})
        raw_detection = result.get("raw_detection", {})
        predicted_spans = []
        for claim in raw_detection.get("claims", []):
            predicted_spans.extend(claim.get("hallucinated_spans", []))
        gold_spans = row.get("gold_hallucinated_spans", [])
        rows.append({
            "id": row.get("id", index),
            "query": query,
            "retrieval_mode": result.get("retrieval_mode"),
            "raw_hallucination_rate": result.get("metrics", {}).get("raw_hallucination_rate"),
            "corrected_hallucination_rate": result.get("metrics", {}).get("corrected_hallucination_rate"),
            "factual_improvement": result.get("metrics", {}).get("factual_improvement"),
            "faithfulness": faithfulness_score(corrected_detection),
            "span_iou": span_iou(predicted_spans, gold_spans),
            "correction_status": result.get("metrics", {}).get("correction_status"),
        })
        print(f"[{index}/{len(records)}] {query} -> {rows[-1]['correction_status']}")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "count": len(rows),
        "avg_factual_improvement": round(sum(float(r.get("factual_improvement") or 0) for r in rows) / len(rows), 4) if rows else 0.0,
        "avg_span_iou": round(sum(float(r.get("span_iou") or 0) for r in rows) / len(rows), 4) if rows else 0.0,
        "avg_faithfulness": round(sum(float(r.get("faithfulness") or 0) for r in rows) / len(rows), 4) if rows else 0.0,
        "baseline_modes_supported": [
            "similarity_only",
            "similarity_plus_rules",
            "similarity_plus_rules_plus_nli",
            "hybrid_plus_nli",
            "hybrid_plus_nli_plus_correction",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote:", csv_path)
    print("Wrote:", summary_path)


if __name__ == "__main__":
    main()
