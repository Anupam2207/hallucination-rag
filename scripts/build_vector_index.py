import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.paths import CHUNKS_DIR, ensure_directories
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.json_utils import load_jsonl


def main() -> None:
    parser = ArgumentParser(description='Build or rebuild the Chroma vector index from chunk artifacts.')
    parser.add_argument('--reset', action='store_true', help='Delete the existing collection before rebuilding.')
    parser.add_argument('--rebuild', action='store_true', help='Alias for --reset.')
    args = parser.parse_args()

    logger = get_logger('build_vector_index')
    ensure_directories()

    chunks_path = CHUNKS_DIR / 'chunks.jsonl'
    chunk_records = load_jsonl(chunks_path)
    if not chunk_records:
        raise FileNotFoundError(
            f'No chunk records found at {chunks_path}. Run scripts/ingest_documents.py first.'
        )

    texts = [record['text'] for record in chunk_records]
    ids = [record['chunk_id'] for record in chunk_records]
    metadatas = [
        {
            'doc_id': record['doc_id'],
            'chunk_index': record['chunk_index'],
            'source_rel': record['source_rel'],
            'file_name': record['file_name'],
            'file_type': record['file_type'],
            'parent_doc_id': record.get('parent_doc_id', record.get('doc_id')),
            'sentence_count': record.get('sentence_count'),
            'start_sentence_index': record.get('start_sentence_index'),
            'end_sentence_index': record.get('end_sentence_index'),
            'document_title': record.get('document_title'),
            'section_name': record.get('section_name'),
            'chunk_position': record.get('chunk_position'),
            'importance_score': record.get('importance_score', 0),
        }
        for record in chunk_records
    ]

    logger.info('Encoding %s chunks...', len(texts))
    embedder = EmbeddingModel()
    embeddings = embedder.encode(texts)
    if hasattr(embeddings, 'tolist'):
        embeddings = embeddings.tolist()

    store = ChromaVectorStore()
    if args.reset or args.rebuild:
        logger.info('Resetting existing Chroma collection...')
        store.reset_collection()

    store.upsert_documents(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    logger.info('Vector index build complete. Collection size: %s', store.count())


if __name__ == '__main__':
    main()
