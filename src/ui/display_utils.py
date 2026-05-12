from typing import Any, Dict, List

import pandas as pd


def claims_to_dataframe(claims: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in claims:
        rows.append(
            {
                "claim": item.get("claim", ""),
                "label": item.get("label", ""),
                "support_score": round(float(item.get("support_score", 0.0)), 3),
            }
        )
    return pd.DataFrame(rows)


def evidence_to_dataframe(evidence_list: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(evidence_list, start=1):
        metadata = item.get("metadata", {})
        rows.append(
            {
                "rank": index,
                "source": metadata.get("source_rel") or metadata.get("file_name") or item.get("chunk_id", "unknown"),
                "similarity": round(float(item.get("similarity") or 0.0), 3),
                "text": item.get("text", ""),
            }
        )
    return pd.DataFrame(rows)
