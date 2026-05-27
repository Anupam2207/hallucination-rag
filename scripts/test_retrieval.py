import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test hybrid retrieval for one query.")
    parser.add_argument("--query", type=str, default="What is retrieval-augmented generation?")
    args = parser.parse_args()
    retriever = HybridRetriever()
    results = retriever.retrieve(args.query)
    print(json.dumps({"query": args.query, "evidence": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
