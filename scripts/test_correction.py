import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import HallucinationRAGPipeline


def main() -> None:
    query = "What is retrieval-augmented generation?"
    result = HallucinationRAGPipeline().run(query)
    print(json.dumps({
        "query": result["query"],
        "raw_answer": result["raw_answer"],
        "corrected_answer": result["corrected_answer"],
        "metrics": result["metrics"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
