import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import BenchmarkEvaluator
from src.logger import get_logger
from src.paths import EVAL_DIR, PREDICTIONS_DIR, REPORTS_DIR, ensure_directories
from src.pipeline import HallucinationRAGPipeline
from src.utils.json_utils import write_jsonl


logger = get_logger('run_batch_evaluation')


def main() -> None:
    ensure_directories()
    benchmark_path = EVAL_DIR / 'benchmark_queries.jsonl'
    if not benchmark_path.exists():
        raise FileNotFoundError(f'Benchmark file not found: {benchmark_path}')

    evaluator = BenchmarkEvaluator(HallucinationRAGPipeline())
    records, summary = evaluator.run(benchmark_path)
    predictions_path = PREDICTIONS_DIR / 'batch_results.jsonl'
    write_jsonl(predictions_path, records)

    summary_path = REPORTS_DIR / 'batch_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    logger.info('Saved %s evaluation records to %s', len(records), predictions_path)
    logger.info('Saved summary to %s', summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
