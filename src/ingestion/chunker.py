import re
from typing import Any, Dict, List

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SECTION_HEADING_RE = re.compile(
    r"^(abstract|introduction|background|related work|method|methodology|approach|experiment|experiments|results|discussion|conclusion|future work|references)\b.*",
    re.IGNORECASE,
)

_SECTION_IMPORTANCE = {
    "abstract": 5,
    "introduction": 4,
    "background": 3,
    "related work": 2,
    "method": 3,
    "methodology": 3,
    "approach": 3,
    "experiment": 1,
    "experiments": 1,
    "results": 1,
    "discussion": 1,
    "conclusion": 2,
    "future work": 1,
    "references": -5,
}


def _section_name_from_line(line: str) -> str | None:
    clean = line.strip().strip("#").strip()
    match = _SECTION_HEADING_RE.match(clean)
    if not match:
        return None
    key = match.group(1).lower()
    return key.title()


def _section_key(section_name: str | None) -> str:
    if not section_name:
        return "body"
    lower = section_name.lower()
    for key in _SECTION_IMPORTANCE:
        if lower.startswith(key):
            return key
    return lower


def section_importance_score(section_name: str | None, text: str = "") -> int:
    key = _section_key(section_name)
    if key in _SECTION_IMPORTANCE:
        return _SECTION_IMPORTANCE[key]
    # Definitions and overview chunks are valuable for RAG evidence.
    lower = (text or "").lower()
    if " is " in lower or " refers to " in lower or " defined as " in lower:
        return 2
    return 0


def _prepare_sectioned_sentences(text: str) -> List[tuple[int, str, str | None]]:
    rows: List[tuple[int, str, str | None]] = []
    current_section: str | None = None
    sentence_index = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        detected_section = _section_name_from_line(line)
        if detected_section:
            current_section = detected_section
            # Keep section title as context only if it has additional content.
            if len(line.split()) <= 4:
                continue
        for sentence in split_sentences(line):
            rows.append((sentence_index, sentence, current_section))
            sentence_index += 1
    if not rows and text.strip():
        for sentence in split_sentences(text):
            rows.append((sentence_index, sentence, current_section))
            sentence_index += 1
    return rows


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return sentences or [text]


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    return [record["text"] for record in chunk_text_with_metadata(text, chunk_size, chunk_overlap)]


def chunk_text_with_metadata(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sectioned_sentences = _prepare_sectioned_sentences(text)
    if not sectioned_sentences:
        return []

    chunks: List[Dict[str, Any]] = []
    current: List[tuple[int, str, str | None]] = []
    current_len = 0

    def current_section_name() -> str | None:
        sections = [section for _, _, section in current if section]
        return sections[-1] if sections else None

    def emit_current() -> None:
        if not current:
            return
        chunk_value = " ".join(sentence for _, sentence, _ in current).strip()
        if not chunk_value:
            return
        start_index = current[0][0]
        end_index = current[-1][0]
        section_name = current_section_name()
        chunks.append(
            {
                "text": chunk_value,
                "sentence_count": len(current),
                "start_sentence_index": start_index,
                "end_sentence_index": end_index,
                "section_name": section_name or "Body",
                "importance_score": section_importance_score(section_name, chunk_value),
            }
        )

    def keep_overlap() -> tuple[List[tuple[int, str, str | None]], int]:
        overlap_items: List[tuple[int, str, str | None]] = []
        overlap_len = 0
        for item in reversed(current):
            overlap_items.insert(0, item)
            overlap_len += len(item[1]) + 1
            if overlap_len >= chunk_overlap:
                break
        return overlap_items, sum(len(sentence) + 1 for _, sentence, _ in overlap_items)

    for sent_index, sentence, section_name in sectioned_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            emit_current()
            current = []
            current_len = 0
            start = 0
            step = max(1, chunk_size - chunk_overlap)
            piece_index = 0
            while start < len(sentence):
                piece = sentence[start : start + chunk_size].strip()
                if piece:
                    chunks.append(
                        {
                            "text": piece,
                            "sentence_count": 1,
                            "start_sentence_index": sent_index,
                            "end_sentence_index": sent_index,
                            "long_sentence_piece_index": piece_index,
                            "section_name": section_name or "Body",
                            "importance_score": section_importance_score(section_name, piece),
                        }
                    )
                    piece_index += 1
                start += step
            continue

        projected_len = current_len + len(sentence) + (1 if current else 0)
        if current and projected_len > chunk_size:
            emit_current()
            current, current_len = keep_overlap()

        current.append((sent_index, sentence, section_name))
        current_len += len(sentence) + (1 if len(current) > 1 else 0)

    emit_current()
    return chunks


def build_sentence_chunk_windows(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    return chunk_text_with_metadata(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def create_chunk_records(document: Dict[str, Any], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    chunk_items = chunk_text_with_metadata(document["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    records: List[Dict[str, Any]] = []
    document_title = document.get("document_title") or document.get("file_name", "Untitled")
    title_pattern = re.compile(rf"^#*\s*{re.escape(str(document_title))}\s+", re.IGNORECASE)
    for index, chunk_item in enumerate(chunk_items):
        chunk_text_value = title_pattern.sub("", chunk_item["text"]).strip()
        chunk_id = f"{document['doc_id']}_chunk_{index}"
        records.append(
            {
                "chunk_id": chunk_id,
                "doc_id": document["doc_id"],
                "parent_doc_id": document["doc_id"],
                "document_title": document_title,
                "chunk_index": index,
                "chunk_position": index,
                "text": chunk_text_value,
                "char_length": len(chunk_text_value),
                "sentence_count": chunk_item.get("sentence_count", 0),
                "start_sentence_index": chunk_item.get("start_sentence_index"),
                "end_sentence_index": chunk_item.get("end_sentence_index"),
                "section_name": chunk_item.get("section_name", "Body"),
                "importance_score": chunk_item.get("importance_score", 0),
                "source_rel": document["source_rel"],
                "file_name": document["file_name"],
                "file_type": document["file_type"],
            }
        )
    return records


def build_chunks_for_documents(documents: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    seen_texts: set[str] = set()
    for document in documents:
        for record in create_chunk_records(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            normalized_text = re.sub(r"\s+", " ", record["text"].strip().lower())
            if not normalized_text or normalized_text in seen_texts:
                continue
            seen_texts.add(normalized_text)
            chunks.append(record)
    return chunks
