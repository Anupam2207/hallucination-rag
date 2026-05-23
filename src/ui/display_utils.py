from typing import Any, Dict, List

import pandas as pd


VISIBLE_CLAIM_COLUMNS = ["claim_preview", "label", "support_score", "rule_flags", "nli_label"]


def _preview(text: Any, limit: int = 120) -> str:
    value = str(text or "").replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def claims_to_dataframe(claims: List[Dict[str, Any]], compact: bool = True) -> pd.DataFrame:
    rows = []
    for item in claims:
        row = {
            "claim_preview": _preview(item.get("highlighted_claim") or item.get("claim", "")),
            "claim": item.get("claim", ""),
            "highlighted_claim": item.get("highlighted_claim", item.get("claim", "")),
            "label": item.get("label", ""),
            "support_score": round(float(item.get("support_score", 0.0)), 3),
            "rule_flags": ", ".join(item.get("rule_flags", [])),
            "raw_similarity": round(float(item.get("raw_similarity_score") or 0.0), 3),
            "nli_label": item.get("nli_label", ""),
            "nli_score": item.get("nli_score", ""),
            "hallucinated_spans": item.get("hallucinated_spans", []),
            "best_evidence_text": item.get("best_evidence_text", ""),
        }
        rows.append(row)
    dataframe = pd.DataFrame(rows)
    if compact and not dataframe.empty:
        return dataframe[VISIBLE_CLAIM_COLUMNS]
    return dataframe


def evidence_to_dataframe(evidence_list: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(evidence_list, start=1):
        metadata = item.get("metadata", {})
        evidence_id = item.get("evidence_id") or f"Evidence-{index}"
        rows.append(
            {
                "evidence_id": evidence_id,
                "rank": index,
                "method": item.get("retrieval_method", "dense"),
                "source": metadata.get("source_rel") or metadata.get("file_name") or item.get("chunk_id", "unknown"),
                "rrf_score": _round_optional(item.get("rrf_score")),
                "dense_similarity": _round_optional(item.get("dense_similarity", item.get("similarity"))),
                "sparse_rank": item.get("sparse_rank"),
                "sparse_score": _round_optional(item.get("sparse_score")),
                "text_preview": _preview(item.get("text", ""), 160),
            }
        )
    return pd.DataFrame(rows)


def _round_optional(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None
