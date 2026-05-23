import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.span_highlighter import highlight_hallucinated_spans


def main() -> None:
    result = highlight_hallucinated_spans(
        "RAG was introduced in 2021.",
        "RAG was introduced in 2020.",
        ["numeric_mismatch_with_evidence"],
    )
    print(result)
    assert "[2021]" in result["highlighted_claim"]
    print("Span highlighter smoke test passed.")


if __name__ == "__main__":
    main()
