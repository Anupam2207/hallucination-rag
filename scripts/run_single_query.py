import json
import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import HallucinationRAGPipeline


def _shorten(text: str, limit: int = 280) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _compact_result(result: dict) -> dict:
    evidence = []
    for item in result.get("evidence", [])[:4]:
        meta = item.get("metadata", {}) or {}
        evidence.append({
            "id": item.get("evidence_id"),
            "source": meta.get("source_rel") or meta.get("file_name"),
            "score": round(float(item.get("final_score") or item.get("weighted_score") or item.get("similarity") or 0.0), 4),
            "text": _shorten(item.get("text", ""), 220),
        })

    claims = []
    for claim in result.get("raw_detection", {}).get("claims", []):
        claims.append({
            "claim": claim.get("claim"),
            "label": claim.get("label"),
            "score": round(float(claim.get("support_score") or 0.0), 4),
        })

    metrics = result.get("metrics", {}) or {}
    return {
        "query": result.get("query"),
        "answerable": result.get("answerable"),
        "raw_answer": result.get("raw_answer"),
        "retrieved_evidence": evidence,
        "claim_verification": claims,
        "corrected_answer": result.get("corrected_answer"),
        "summary": {
            "supported_claims": result.get("raw_detection", {}).get("supported_count", 0),
            "weak_claims": result.get("raw_detection", {}).get("weak_count", 0),
            "unsupported_claims": result.get("raw_detection", {}).get("unsupported_count", 0),
            "raw_support_ratio": metrics.get("raw_support_ratio"),
            "corrected_support_ratio": metrics.get("corrected_support_ratio"),
            "correction_action": metrics.get("correction_action"),
            "correction_status": metrics.get("correction_status"),
            "correction_success": metrics.get("correction_success"),
            "warnings": result.get("warnings", []),
        },
    }


def _print_readable(compact: dict) -> None:
    print("\nQUERY")
    print(compact["query"])
    print("\nRAW LLM ANSWER")
    print(compact.get("raw_answer") or "")

    print("\nRETRIEVED EVIDENCE")
    for i, ev in enumerate(compact.get("retrieved_evidence", []), start=1):
        print(f"{i}. {ev['source']}  score={ev['score']}")
        print(f"   {ev['text']}")

    print("\nCLAIM VERIFICATION")
    claims = compact.get("claim_verification", [])
    if not claims:
        print("No factual claims extracted.")
    for i, claim in enumerate(claims, start=1):
        print(f"{i}. [{claim['label']}] score={claim['score']} - {claim['claim']}")

    print("\nCORRECTED ANSWER")
    print(compact.get("corrected_answer") or "")

    print("\nSUMMARY")
    summary = compact.get("summary", {})
    for key in (
        "supported_claims", "weak_claims", "unsupported_claims",
        "raw_support_ratio", "corrected_support_ratio",
        "correction_action", "correction_status", "correction_success",
    ):
        print(f"{key}: {summary.get(key)}")
    if summary.get("warnings"):
        print("warnings:", "; ".join(summary["warnings"]))


def main() -> None:
    parser = ArgumentParser(description="Run one query through the simplified RAG hallucination pipeline.")
    parser.add_argument("--query", type=str, help="User query to process.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k.")
    parser.add_argument("--json", action="store_true", help="Print compact JSON instead of readable text.")
    parser.add_argument("--debug-json", action="store_true", help="Print the full internal debug JSON.")
    parser.add_argument("--no-correction", action="store_true", help="Disable answer correction.")
    args = parser.parse_args()

    query = args.query or input("Enter your query: ").strip()
    pipeline = HallucinationRAGPipeline()
    result = pipeline.run(query=query, top_k=args.top_k, correction_enabled=not args.no_correction)

    if args.debug_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        compact = _compact_result(result)
        if args.json:
            print(json.dumps(compact, indent=2, ensure_ascii=False))
        else:
            _print_readable(compact)


if __name__ == "__main__":
    main()
