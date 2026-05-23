import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


def main() -> None:
    parser = ArgumentParser(description="Test optional cross-encoder reranker.")
    parser.add_argument("--query", default="What is retrieval-augmented generation?")
    parser.add_argument("--enable", action="store_true", help="Load the real reranker model.")
    args = parser.parse_args()

    retriever = HybridRetriever()
    evidence = retriever.retrieve(args.query, top_k=8)
    reranker = CrossEncoderReranker(enabled=args.enable)
    reranked = reranker.rerank(args.query, evidence, top_k=5)
    print("Reranker enabled:", reranker.enabled)
    print("Reranker available:", reranker.available)
    if reranker.error:
        print("Reranker error:", reranker.error)
    for item in reranked:
        print({
            "chunk_id": item.get("chunk_id"),
            "rerank_score": item.get("rerank_score"),
            "rrf_score": item.get("rrf_score"),
            "preview": item.get("text", "")[:120],
        })


if __name__ == "__main__":
    main()
