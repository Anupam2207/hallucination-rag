import json
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config_value
from src.evaluation.metrics import compare_detection_results
from src.logger import get_logger
from src.paths import EVAL_DIR, PREDICTIONS_DIR, REPORTS_DIR, ensure_directories
from src.pipeline import HallucinationRAGPipeline
from src.utils.json_utils import write_jsonl


logger = get_logger('run_batch_evaluation')


def load_benchmark(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get('query'):
                records.append(item)
            if limit is not None and len(records) >= limit:
                break
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}

    def avg(field: str) -> float:
        values = [float(record.get(field, 0.0) or 0.0) for record in records]
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        'num_queries': len(records),
        'avg_raw_support_ratio': avg('raw_support_ratio'),
        'avg_corrected_support_ratio': avg('corrected_support_ratio'),
        'avg_raw_weighted_support_ratio': avg('raw_weighted_support_ratio'),
        'avg_corrected_weighted_support_ratio': avg('corrected_weighted_support_ratio'),
        'avg_factual_improvement': avg('factual_improvement'),
        'avg_raw_hallucination_rate': avg('raw_hallucination_rate'),
        'avg_corrected_hallucination_rate': avg('corrected_hallucination_rate'),
        'avg_hallucination_reduction': avg('hallucination_reduction'),
        'avg_claim_count_reduction': avg('claim_count_reduction'),
        'correction_success_rate': avg('correction_success'),
    }


def main() -> None:
    parser = ArgumentParser(description='Run bounded batch evaluation over benchmark_queries.jsonl.')
    parser.add_argument('--benchmark', type=str, default=str(EVAL_DIR / 'benchmark_queries.jsonl'))
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of benchmark queries to run.')
    parser.add_argument('--max-workers', type=int, default=int(get_config_value('settings', 'runtime', 'max_batch_workers', default=1)))
    parser.add_argument('--max-llm-workers', type=int, default=int(get_config_value('settings', 'runtime', 'max_llm_workers', default=1)))
    parser.add_argument('--max-nli-workers', type=int, default=int(get_config_value('settings', 'runtime', 'max_nli_workers', default=1)))
    args = parser.parse_args()

    ensure_directories()
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        raise FileNotFoundError(f'Benchmark file not found: {benchmark_path}')

    items = load_benchmark(benchmark_path, limit=args.limit)
    if not items:
        raise RuntimeError(f'No benchmark queries found in {benchmark_path}')

    max_workers = max(1, int(args.max_workers))
    llm_semaphore = BoundedSemaphore(max(1, int(args.max_llm_workers)))

    logger.info(
        'Running %s queries with max_workers=%s, max_llm_workers=%s, max_nli_workers=%s',
        len(items),
        max_workers,
        args.max_llm_workers,
        args.max_nli_workers,
    )

    # Reuse a single pipeline by default to avoid multiple model instances on
    # low-resource laptops. For max_workers > 1, a small bounded pool is used;
    # the LLM section is still protected by llm_semaphore.
    shared_pipeline = HallucinationRAGPipeline()

    def run_one(item: dict[str, Any]) -> dict[str, Any]:
        with llm_semaphore:
            result = shared_pipeline.run(item['query'])
        metrics = compare_detection_results(result['raw_detection'], result['corrected_detection'])
        return {
            'query_id': item.get('query_id') or item.get('id'),
            'query': item['query'],
            'ground_truth': item.get('ground_truth', ''),
            'raw_answer': result.get('raw_answer', ''),
            'corrected_answer': result.get('corrected_answer', ''),
            'evidence_count': len(result.get('evidence', [])),
            'retrieval_mode': result.get('retrieval_mode'),
            **metrics,
        }

    records: list[dict[str, Any]] = []
    if max_workers == 1:
        for index, item in enumerate(items, start=1):
            logger.info('Evaluating query %s/%s: %s', index, len(items), item['query'])
            records.append(run_one(item))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(run_one, item): item for item in items}
            for index, future in enumerate(as_completed(future_to_item), start=1):
                item = future_to_item[future]
                try:
                    records.append(future.result())
                    logger.info('Completed query %s/%s: %s', index, len(items), item['query'])
                except Exception as exc:
                    logger.error('Failed query %s: %s', item.get('query'), exc)
                    records.append({'query': item.get('query'), 'error': str(exc)})

    summary = summarize_records([record for record in records if 'error' not in record])
    predictions_path = PREDICTIONS_DIR / 'batch_results.jsonl'
    write_jsonl(predictions_path, records)

    summary_path = REPORTS_DIR / 'batch_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    logger.info('Saved %s evaluation records to %s', len(records), predictions_path)
    logger.info('Saved summary to %s', summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
