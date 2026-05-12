import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import compare_detection_results


class BenchmarkEvaluator:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def run(self, benchmark_path: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
        records: list[dict[str, Any]] = []
        with open(benchmark_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                result = self.pipeline.run(item['query'])
                metrics = compare_detection_results(result['raw_detection'], result['corrected_detection'])
                records.append(
                    {
                        'query_id': item.get('query_id'),
                        'query': item['query'],
                        'ground_truth': item.get('ground_truth', ''),
                        **metrics,
                    }
                )

        if not records:
            return [], {}

        df = pd.DataFrame(records)
        summary = {
            'num_queries': int(len(df)),
            'avg_raw_support_ratio': round(float(df['raw_support_ratio'].mean()), 4),
            'avg_corrected_support_ratio': round(float(df['corrected_support_ratio'].mean()), 4),
            'avg_support_improvement': round(float(df['support_improvement'].mean()), 4),
            'avg_raw_hallucination_rate': round(float(df['raw_hallucination_rate'].mean()), 4),
            'avg_corrected_hallucination_rate': round(float(df['corrected_hallucination_rate'].mean()), 4),
            'avg_hallucination_reduction': round(float(df['hallucination_reduction'].mean()), 4),
        }
        return records, summary
