"""Evaluate retrieval quality for the local hallucination-RAG benchmark."""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation_common import load_jsonl, write_csv, write_json
from src.config import get_first_config_value
from src.evaluation.retrieval_metrics import (
    answerability_retrieval_success,
    average,
    first_relevant_rank,
    keyword_hit_rate,
    mean_reciprocal_rank,
    recall_at_k,
)
from src.pipeline import HallucinationRAGPipeline
from src.retrieval.evidence_intent import annotate_evidence_list
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.sparse_retriever import BM25SparseRetriever


def build_retriever(mode: str):
    mode = mode.lower()
    if mode in {"bm25", "sparse", "bm25-only"}:
        return BM25SparseRetriever(), "bm25"
    if mode in {"dense", "dense-only"}:
        return SemanticRetriever(), "dense"
    return HybridRetriever(), "hybrid"


def _safe_retrieve(retriever: Any, query: str, top_k: int) -> tuple[list[dict[str, Any]], str]:
    try:
        return list(retriever.retrieve(query, top_k=top_k)), ""
    except Exception as exc:
        return [], f"retrieval_failed: {exc}"


def evaluate_retrieval_records(records: List[Dict[str, Any]], mode: str = "hybrid", top_k: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    top_k = int(top_k or get_first_config_value(("settings", "retrieval", "top_k"), default=4))
    retriever, resolved_mode = build_retriever(mode)
    threshold = float(
        get_first_config_value(
            ("settings", "retrieval", "answerability_threshold"),
            ("settings", "detection", "answerability_threshold"),
            default=0.45,
        )
    )
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        query = str(record.get("query", "")).strip()
        expected_keywords = record.get("expected_evidence_keywords", []) or []
        retrieved, warning = _safe_retrieve(retriever, query, top_k=top_k)
        annotated = annotate_evidence_list(retrieved, query=query)
        first_rank = first_relevant_rank(annotated, expected_keywords)
        answerable_by_score = HallucinationRAGPipeline._is_answerable_by_score(annotated, threshold, strict_factual_mode=bool(record.get("expected_answer_type") == "specific_fact_answerability"))
        specific_supported, _reason = HallucinationRAGPipeline._specific_fact_supported_by_evidence(query, annotated)
        predicted_answerable = bool(answerable_by_score and specific_supported)
        gold_answerable = bool(record.get("answerable", True))
        rows.append(
            {
                "id": record.get("id", index),
                "category": record.get("category", ""),
                "query": query,
                "retrieval_mode": resolved_mode,
                "top_k": top_k,
                "retrieved_count": len(annotated),
                "first_relevant_rank": first_rank or "",
                "recall_at_k_hit": bool(first_rank and first_rank <= top_k),
                "mrr": round(0.0 if not first_rank else 1.0 / int(first_rank), 4),
                "evidence_keyword_hit_rate": keyword_hit_rate(annotated, expected_keywords),
                "predicted_answerable": predicted_answerable,
                "gold_answerable": gold_answerable,
                "answerability_retrieval_success": predicted_answerable == gold_answerable,
                "top_chunk_id": annotated[0].get("chunk_id", "") if annotated else "",
                "top_source": (annotated[0].get("metadata", {}) or {}).get("source_rel", "") if annotated else "",
                "warning": warning,
            }
        )
    summary = {
        "records": len(records),
        "retrieval_mode": resolved_mode,
        "top_k": top_k,
        "recall_at_k": recall_at_k(rows, top_k),
        "mrr": mean_reciprocal_rank(rows),
        "evidence_keyword_hit_rate": average(float(row["evidence_keyword_hit_rate"] or 0.0) for row in rows),
        "answerability_retrieval_success": answerability_retrieval_success(rows),
        "failed_retrievals": sum(1 for row in rows if row.get("warning")),
    }
    return rows, summary


def main() -> None:
    parser = ArgumentParser(description="Evaluate retrieval recall, MRR, keyword hits, and answerability success.")
    parser.add_argument("--input", default="data/evaluation/final_eval_set.jsonl")
    parser.add_argument("--retrieval", "--mode", dest="mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="paper_artifacts")
    args = parser.parse_args()

    records = load_jsonl(PROJECT_ROOT / args.input)
    if args.limit is not None:
        records = records[: args.limit]
    rows, summary = evaluate_retrieval_records(records, mode=args.mode, top_k=args.top_k)

    artifact_dir = PROJECT_ROOT / args.output_dir
    tables_dir = artifact_dir / "tables"
    metrics_dir = artifact_dir / "metrics"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    for path in (tables_dir, metrics_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    write_csv(tables_dir / "retrieval_results.csv", rows)
    write_json(metrics_dir / "retrieval_summary.json", summary)
    write_csv(reports_dir / "retrieval_results.csv", rows)
    write_json(reports_dir / "retrieval_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
