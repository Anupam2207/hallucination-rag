import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.span_highlighter import highlight_hallucinated_spans


def main() -> None:
    cases = [
        (
            "RAG was introduced in 2021.",
            "RAG was introduced in 2020.",
            ["numeric_mismatch_with_evidence"],
        ),
        (
            "RAG was introduced in 2021.",
            "RAG combines retrieval and generation.",
            ["claim_year_not_supported_by_evidence"],
        ),
        (
            "RAG fine-tunes a generator model on retrieved passages.",
            "RAG uses retrieved passages as context.",
            ["fine_tuning_not_in_evidence"],
        ),
    ]
    for claim, evidence, flags in cases:
        result = highlight_hallucinated_spans(claim, evidence, flags)
        print(result)
        assert result["hallucinated_spans"]
    citation_result = highlight_hallucinated_spans(
        "RAG combines retrieval and generation [Evidence-1].",
        "RAG combines retrieval and generation.",
        [],
    )
    print(citation_result)
    assert citation_result["hallucinated_spans"] == []
    print("Span highlighter smoke test passed.")


if __name__ == "__main__":
    main()
