import json
import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import HallucinationRAGPipeline


def main() -> None:
    parser = ArgumentParser(description='Run a single query through the full hallucination-detection pipeline.')
    parser.add_argument('--query', type=str, help='User query to process.')
    parser.add_argument('--top-k', type=int, default=None, help='Override retrieval top-k.')
    args = parser.parse_args()

    query = args.query or input('Enter your query: ').strip()
    pipeline = HallucinationRAGPipeline()
    result = pipeline.run(query=query, top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
