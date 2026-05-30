from typing import Any, Dict, Iterable, List


def deduplicate_evidence(evidence_list: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_texts: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for item in evidence_list:
        text = str(item.get("text", "")).strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        copied = dict(item)
        copied.setdefault("evidence_id", f"Evidence-{len(deduped) + 1}")
        deduped.append(copied)
    return deduped


def format_evidence_block(
    evidence_list: Iterable[Dict[str, Any]], max_items: int | None = None
) -> str:
    """Convert retrieved evidence into a prompt-friendly text block."""
    formatted: List[str] = []
    for index, item in enumerate(deduplicate_evidence(evidence_list)):
        if max_items is not None and index >= max_items:
            break
        metadata = item.get("metadata", {})
        source = (
            metadata.get("source_rel")
            or metadata.get("file_name")
            or item.get("chunk_id", "unknown")
        )
        evidence_id = item.get("evidence_id") or f"Evidence-{index + 1}"
        text = item.get("text", "").strip()
        if not text:
            continue
        formatted.append(f"[{evidence_id}] Source: {source}\n{text}")
    return "\n\n".join(formatted)
