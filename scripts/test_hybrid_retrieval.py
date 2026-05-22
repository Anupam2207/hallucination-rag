import json
import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import SemanticRetriever


def compact(rows: list[dict]) -> list[dict]:
    compacted = []
    for row in rows:
        compacted.append(
            {
                'chunk_id': row.get('chunk_id'),
                'source': (row.get('metadata') or {}).get('source_rel'),
                'dense_rank': row.get('dense_rank'),
                'sparse_rank': row.get('sparse_rank'),
                'dense_similarity': row.get('dense_similarity', row.get('similarity')),
                'sparse_score': row.get('sparse_score'),
                'rrf_score': row.get('rrf_score'),
                'text_preview': row.get('text', '')[:180],
            }
        )
    return compacted


def main() -> None:
    parser = ArgumentParser(description='Compare dense-only and hybrid retrieval for one query.')
    parser.add_argument('--query', default='What is retrieval-augmented generation?')
    parser.add_argument('--top-k', type=int, default=4)
    args = parser.parse_args()

    dense = SemanticRetriever()
    hybrid = HybridRetriever()

    dense_results = dense.retrieve(args.query, top_k=args.top_k)
    hybrid_results = hybrid.retrieve(args.query, top_k=args.top_k)

    print('\nDENSE RESULTS')
    print(json.dumps(compact(dense_results), indent=2))

    print('\nHYBRID RESULTS')
    print(json.dumps(compact(hybrid_results), indent=2))


if __name__ == '__main__':
    main()
