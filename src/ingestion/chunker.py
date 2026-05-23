import re
from typing import Any, Dict, List


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Split on sentence boundaries while preserving punctuation markers.
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return sentences or [text]


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """Sentence-aware chunking with overlap.

    This preserves sentence boundaries whenever possible. Very long sentences are
    split by character window only as a fallback.
    """
    return [record["text"] for record in chunk_text_with_metadata(text, chunk_size, chunk_overlap)]


def chunk_text_with_metadata(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if chunk_overlap < 0:
        raise ValueError('chunk_overlap must be non-negative')
    if chunk_overlap >= chunk_size:
        raise ValueError('chunk_overlap must be smaller than chunk_size')

    # Break the document into sentence-aware chunks with optional overlap.

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[Dict[str, Any]] = []
    current_sentences: List[tuple[int, str]] = []
    current_len = 0

    def emit_current() -> None:
        if not current_sentences:
            return
        chunk_text_value = ' '.join(sentence for _, sentence in current_sentences).strip()
        if not chunk_text_value:
            return
        # Finalize the current chunk before starting a new one.
        start_index = current_sentences[0][0]
        end_index = current_sentences[-1][0]
        chunks.append(
            {
                'text': chunk_text_value,
                'sentence_count': len(current_sentences),
                'start_sentence_index': start_index,
                'end_sentence_index': end_index,
            }
        )

    def keep_overlap() -> tuple[List[tuple[int, str]], int]:
        overlap_sentences: List[tuple[int, str]] = []
        overlap_len = 0
        # Preserve a slice of the previous chunk to overlap with the next chunk.
        for item in reversed(current_sentences):
            overlap_sentences.insert(0, item)
            overlap_len += len(item[1]) + 1
            if overlap_len >= chunk_overlap:
                break
        return overlap_sentences, sum(len(sentence) + 1 for _, sentence in overlap_sentences)

    for sent_index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > chunk_size:
            emit_current()
            current_sentences = []
            current_len = 0
            start = 0
            step = max(1, chunk_size - chunk_overlap)
            piece_index = 0
            # Long sentences are split into fixed-size pieces rather than
            # forcing the chunker to exceed the maximum chunk length.
            while start < len(sentence):
                piece = sentence[start:start + chunk_size].strip()
                if piece:
                    chunks.append(
                        {
                            'text': piece,
                            'sentence_count': 1,
                            'start_sentence_index': sent_index,
                            'end_sentence_index': sent_index,
                            'long_sentence_piece_index': piece_index,
                        }
                    )
                    piece_index += 1
                start += step
            continue

        projected_len = current_len + len(sentence) + (1 if current_sentences else 0)
        if current_sentences and projected_len > chunk_size:
            emit_current()
            current_sentences, current_len = keep_overlap()

        current_sentences.append((sent_index, sentence))
        current_len += len(sentence) + (1 if len(current_sentences) > 1 else 0)

    emit_current()
    return chunks




def build_sentence_chunk_windows(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """Compatibility wrapper returning semantic sentence windows with metadata."""
    return chunk_text_with_metadata(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def create_chunk_records(document: Dict[str, Any], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    chunk_items = chunk_text_with_metadata(document['text'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    records: List[Dict[str, Any]] = []
    for index, chunk_item in enumerate(chunk_items):
        chunk_text_value = chunk_item['text']
        chunk_id = f"{document['doc_id']}_chunk_{index}"
        records.append(
            {
                'chunk_id': chunk_id,
                'doc_id': document['doc_id'],
                'parent_doc_id': document['doc_id'],
                'chunk_index': index,
                'text': chunk_text_value,
                'char_length': len(chunk_text_value),
                'sentence_count': chunk_item.get('sentence_count', 0),
                'start_sentence_index': chunk_item.get('start_sentence_index'),
                'end_sentence_index': chunk_item.get('end_sentence_index'),
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
