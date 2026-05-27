import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config_value
from src.ingestion.chunker import build_chunks_for_documents
from src.ingestion.loaders import load_documents
from src.ingestion.preprocess import preprocess_documents
from src.logger import get_logger
from src.paths import CHUNKS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories
from src.utils.json_utils import write_jsonl


logger = get_logger('ingest_documents')


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with open(path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_directories()
    logger.info('Loading documents from %s', RAW_DATA_DIR)
    documents = load_documents(RAW_DATA_DIR)
    if not documents:
        raise RuntimeError(
            'No raw documents found. Add files under data/raw/txt, data/raw/md, data/raw/json, or data/raw/pdf.'
        )

    processed_documents = preprocess_documents(documents)
    logger.info('Loaded %s documents, %s remained after preprocessing.', len(documents), len(processed_documents))

    chunk_size = int(get_config_value('settings', 'retrieval', 'chunk_size', default=500))
    chunk_overlap = int(get_config_value('settings', 'retrieval', 'chunk_overlap', default=100))
    chunks = build_chunks_for_documents(processed_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info('Generated %s chunks.', len(chunks))

    processed_jsonl = PROCESSED_DATA_DIR / 'cleaned_documents.jsonl'
    processed_csv = PROCESSED_DATA_DIR / 'corpus_metadata.csv'
    chunks_jsonl = CHUNKS_DIR / 'chunks.jsonl'
    chunks_csv = CHUNKS_DIR / 'chunk_metadata.csv'

    write_jsonl(processed_jsonl, processed_documents)
    write_csv(
        processed_csv,
        [
            {
                'doc_id': doc['doc_id'],
                'file_name': doc['file_name'],
                'file_type': doc['file_type'],
                'source_rel': doc['source_rel'],
                'document_title': doc.get('document_title'),
                'text_length': doc['text_length'],
            }
            for doc in processed_documents
        ],
    )
    write_jsonl(chunks_jsonl, chunks)
    write_csv(
        chunks_csv,
        [
            {
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'chunk_index': chunk['chunk_index'],
                'file_name': chunk['file_name'],
                'file_type': chunk['file_type'],
                'source_rel': chunk['source_rel'],
                'char_length': chunk['char_length'],
                'sentence_count': chunk.get('sentence_count'),
                'start_sentence_index': chunk.get('start_sentence_index'),
                'end_sentence_index': chunk.get('end_sentence_index'),
                'parent_doc_id': chunk.get('parent_doc_id'),
                'document_title': chunk.get('document_title'),
                'section_name': chunk.get('section_name'),
                'chunk_position': chunk.get('chunk_position'),
                'importance_score': chunk.get('importance_score'),
            }
            for chunk in chunks
        ],
    )

    logger.info('Wrote processed documents to %s', processed_jsonl)
    logger.info('Wrote chunk artifacts to %s', chunks_jsonl)


if __name__ == '__main__':
    main()
