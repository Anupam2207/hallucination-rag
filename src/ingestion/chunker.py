import re
from typing import Any, Dict, List


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return sentences or [text]


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if chunk_overlap < 0:
        raise ValueError('chunk_overlap must be non-negative')
    if chunk_overlap >= chunk_size:
        raise ValueError('chunk_overlap must be smaller than chunk_size')

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    def flush_chunk() -> None:
        nonlocal current_sentences, current_len
        if not current_sentences:
            return
        chunk = ' '.join(current_sentences).strip()
        if chunk:
            chunks.append(chunk)

        overlap_sentences: List[str] = []
        overlap_len = 0
        for sentence in reversed(current_sentences):
            overlap_sentences.insert(0, sentence)
            overlap_len += len(sentence) + 1
            if overlap_len >= chunk_overlap:
                break
        current_sentences = overlap_sentences
        current_len = sum(len(sentence) + 1 for sentence in current_sentences)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > chunk_size:
            if current_sentences:
                flush_chunk()
            start = 0
            step = max(1, chunk_size - chunk_overlap)
            while start < len(sentence):
                piece = sentence[start:start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                start += step
            current_sentences = []
            current_len = 0
            continue

        projected_len = current_len + len(sentence) + (1 if current_sentences else 0)
        if current_sentences and projected_len > chunk_size:
            flush_chunk()

        current_sentences.append(sentence)
        current_len += len(sentence) + (1 if len(current_sentences) > 1 else 0)

    if current_sentences:
        chunk = ' '.join(current_sentences).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def create_chunk_records(document: Dict[str, Any], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    chunks = chunk_text(document['text'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    records: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        chunk_id = f"{document['doc_id']}_chunk_{index}"
        records.append(
            {
                'chunk_id': chunk_id,
                'doc_id': document['doc_id'],
                'chunk_index': index,
                'text': chunk,
                'char_length': len(chunk),
                'source_rel': document['source_rel'],
                'file_name': document['file_name'],
                'file_type': document['file_type'],
            }
        )
    return records


def build_chunks_for_documents(documents: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for document in documents:
        chunks.extend(create_chunk_records(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return chunks
