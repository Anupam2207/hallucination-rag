from typing import Any, Dict, Iterable, List

from src.retrieval.evidence_intent import annotate_evidence, is_credible_support_evidence


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
    evidence_list: Iterable[Dict[str, Any]],
    max_items: int | None = None,
    support_only: bool = True,
) -> str:
    """Convert retrieved evidence into a prompt-friendly text block.

    Correction prompts should not receive examples, references, tutorials, or
    malformed chunks as grounding evidence.  The original list is preserved in
    pipeline outputs; this formatter only controls what the correction LLM sees.
    """
    formatted: List[str] = []
    for item in deduplicate_evidence(evidence_list):
        if max_items is not None and len(formatted) >= max_items:
            break
        annotated = annotate_evidence(item) if "evidence_type" not in item else dict(item)
        if support_only and not is_credible_support_evidence(annotated, min_credibility=0.20):
            continue
        metadata = annotated.get("metadata", {})
        source = (
            metadata.get("source_rel")
            or metadata.get("file_name")
            or annotated.get("chunk_id", "unknown")
        )
        evidence_id = annotated.get("evidence_id") or f"Evidence-{len(formatted) + 1}"
        text = annotated.get("text", "").strip()
        if not text:
            continue
        evidence_type = annotated.get("evidence_type", "UNKNOWN")
        confidence = annotated.get("evidence_credibility_score", "")
        formatted.append(
            f"[{evidence_id}] Type: {evidence_type}; Confidence: {confidence}; Source: {source}\n{text}"
        )
    return "\n\n".join(formatted)
