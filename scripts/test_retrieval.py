import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import SemanticRetriever


def main() -> None:
    query = "What is retrieval-augmented generation?"
    retriever = SemanticRetriever()
    results = retriever.retrieve(query)
    print(json.dumps({"query": query, "evidence": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
