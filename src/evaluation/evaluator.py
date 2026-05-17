import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import compare_detection_results


class BenchmarkEvaluator:
    """Run the full pipeline on a JSONL benchmark file.

    Expected benchmark JSONL fields:
    - query_id: optional string/id
    - query: required user query
    - ground_truth: optional reference answer
    """

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def run(self, benchmark_path: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
        records: list[dict[str, Any]] = []
        with open(benchmark_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not item.get("query"):
                    continue

                result = self.pipeline.run(item["query"])
                metrics = compare_detection_results(
                    result["raw_detection"],
                    result["corrected_detection"],
                )
                records.append(
                    {
                        "query_id": item.get("query_id") or item.get("id"),
                        "query": item["query"],
                        "ground_truth": item.get("ground_truth", ""),
                        "raw_answer": result.get("raw_answer", ""),
                        "corrected_answer": result.get("corrected_answer", ""),
                        "evidence_count": len(result.get("evidence", [])),
                        **metrics,
                    }
                )

        if not records:
            return [], {}

        df = pd.DataFrame(records)
        summary = {
            "num_queries": int(len(df)),
            "avg_raw_support_ratio": round(float(df["raw_support_ratio"].mean()), 4),
            "avg_corrected_support_ratio": round(float(df["corrected_support_ratio"].mean()), 4),
            "avg_raw_weighted_support_ratio": round(float(df["raw_weighted_support_ratio"].mean()), 4),
            "avg_corrected_weighted_support_ratio": round(float(df["corrected_weighted_support_ratio"].mean()), 4),
            "avg_factual_improvement": round(float(df["factual_improvement"].mean()), 4),
            "avg_raw_hallucination_rate": round(float(df["raw_hallucination_rate"].mean()), 4),
            "avg_corrected_hallucination_rate": round(float(df["corrected_hallucination_rate"].mean()), 4),
            "avg_hallucination_reduction": round(float(df["hallucination_reduction"].mean()), 4),
            "avg_claim_count_reduction": round(float(df["claim_count_reduction"].mean()), 4),
            "correction_success_rate": round(float(df["correction_success"].mean()), 4),
        }
        return records, summary
