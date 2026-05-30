"""Run the lightweight final evaluation set for the RAG hallucination project.

This script is intentionally small and local-friendly. It does not download any
external benchmark. It evaluates the existing pipeline on a curated JSONL set and
writes CSV/JSON summaries for report tables and ablation studies.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.span_metrics import span_iou
from src.pipeline import HallucinationRAGPipeline
from src.utils.text_cleaning import normalize_for_detection


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,}", normalize_for_detection(text).lower()))


def similar_enough(a: str, b: str, threshold: float = 0.45) -> bool:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return False
    score = len(ta & tb) / max(1, len(ta | tb))
    return score >= threshold or ta.issubset(tb) or tb.issubset(ta)


def collect_predicted_unsupported(detection: Dict[str, Any]) -> List[str]:
    claims = detection.get("claims", []) or []
    return [str(item.get("claim", "")) for item in claims if item.get("label") == "unsupported"]


def claim_prf(gold_unsupported: Iterable[str], predicted_unsupported: Iterable[str]) -> Dict[str, float]:
    gold = list(gold_unsupported or [])
    pred = list(predicted_unsupported or [])
    matched_gold: set[int] = set()
    tp = 0
    for p in pred:
        hit = None
        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            if similar_enough(p, g):
                hit = i
                break
        if hit is not None:
            matched_gold.add(hit)
            tp += 1
    fp = max(0, len(pred) - tp)
    fn = max(0, len(gold) - tp)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def collect_predicted_spans(detection: Dict[str, Any]) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    for claim in detection.get("claims", []) or []:
        spans.extend(claim.get("hallucinated_spans", []) or [])
    return spans


def evidence_keyword_hit_rate(evidence: List[Dict[str, Any]], expected_keywords: Iterable[str]) -> float:
    keywords = [normalize_for_detection(str(k)).lower() for k in (expected_keywords or []) if str(k).strip()]
    if not keywords:
        return 0.0
    text = normalize_for_detection(" ".join(str(item.get("text", "")) for item in evidence)).lower()
    hits = sum(1 for keyword in keywords if keyword in text)
    return round(hits / len(keywords), 4)


def average(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> None:
    parser = ArgumentParser(description="Run final lightweight benchmark evaluation.")
    parser.add_argument("--input", default="data/evaluation/final_eval_set.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retrieval", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--nli", choices=["on", "off"], default="off")
    parser.add_argument("--correction", choices=["on", "off"], default="on")
    args = parser.parse_args()

    records = load_jsonl(PROJECT_ROOT / args.input)
    if args.limit is not None:
        records = records[: args.limit]

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = HallucinationRAGPipeline(
        retrieval_mode=args.retrieval,
        enable_nli=(args.nli == "on"),
    )
    correction_enabled = args.correction == "on"

    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        query = str(record.get("query", "")).strip()
        print(f"[{index}/{len(records)}] {query}")
        result = pipeline.run(query=query, correction_enabled=correction_enabled)
        raw_detection = result.get("raw_detection", {}) or {}
        corrected_detection = result.get("corrected_detection", {}) or {}
        metrics = result.get("metrics", {}) or {}
        predicted_unsupported = collect_predicted_unsupported(raw_detection)
        prf = claim_prf(record.get("gold_unsupported_claims", []), predicted_unsupported)
        predicted_spans = collect_predicted_spans(raw_detection)
        span_score = span_iou(predicted_spans, record.get("gold_hallucinated_spans", []))
        row = {
            "id": record.get("id", index),
            "query": query,
            "expected_answer_type": record.get("expected_answer_type", ""),
            "retrieval_mode": result.get("retrieval_mode"),
            "nli": args.nli,
            "correction": args.correction,
            "answerability_status": result.get("answerability_status", ""),
            "raw_support_ratio": metrics.get("raw_support_ratio", 0.0),
            "corrected_support_ratio": metrics.get("corrected_support_ratio", 0.0),
            "raw_hallucination_rate": metrics.get("raw_hallucination_rate", 0.0),
            "corrected_hallucination_rate": metrics.get("corrected_hallucination_rate", 0.0),
            "hallucination_reduction": metrics.get("hallucination_reduction", 0.0),
            "factual_improvement": metrics.get("factual_improvement", 0.0),
            "average_corrected_support_score": corrected_detection.get("average_support_score", 0.0),
            "correction_status": metrics.get("correction_status", ""),
            "correction_success": metrics.get("correction_success", False),
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "span_iou": span_score,
            "evidence_keyword_hit_rate": evidence_keyword_hit_rate(result.get("evidence", []), record.get("expected_evidence_keywords", [])),
        }
        rows.append(row)

    suffix = f"{args.retrieval}_nli-{args.nli}_correction-{args.correction}"
    csv_path = output_dir / f"final_eval_results_{suffix}.csv"
    summary_path = output_dir / f"final_eval_summary_{suffix}.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "count": len(rows),
        "retrieval": args.retrieval,
        "nli": args.nli,
        "correction": args.correction,
        "avg_precision": average(float(r["precision"]) for r in rows),
        "avg_recall": average(float(r["recall"]) for r in rows),
        "avg_f1": average(float(r["f1"]) for r in rows),
        "avg_span_iou": average(float(r["span_iou"]) for r in rows),
        "avg_support_score": average(float(r["average_corrected_support_score"] or 0.0) for r in rows),
        "avg_hallucination_reduction": average(float(r["hallucination_reduction"] or 0.0) for r in rows),
        "correction_success_rate": average(1.0 if r["correction_success"] in {True, "True", "true", 1} else 0.0 for r in rows),
        "avg_evidence_keyword_hit_rate": average(float(r["evidence_keyword_hit_rate"] or 0.0) for r in rows),
        "notes": "Precision/recall/F1 are approximate claim-overlap metrics for lightweight local evaluation. Span IoU uses gold character spans when available.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
