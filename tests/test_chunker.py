from src.ingestion.chunker import build_sentence_chunk_windows, create_chunk_records, chunk_text


def test_chunk_text_splits_long_text() -> None:
    text = 'Sentence one. Sentence two. Sentence three. Sentence four.'
    chunks = chunk_text(text, chunk_size=25, chunk_overlap=5)
    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_rejects_invalid_overlap() -> None:
    try:
        chunk_text('hello world', chunk_size=10, chunk_overlap=10)
        assert False, 'Expected ValueError'
    except ValueError:
        assert True


def test_sentence_window_metadata_exists() -> None:
    text = 'Sentence one. Sentence two. Sentence three.'
    windows = build_sentence_chunk_windows(text, chunk_size=80, chunk_overlap=10)
    assert windows
    assert {'text', 'sentence_count', 'start_sentence_index', 'end_sentence_index'} <= set(windows[0])


def test_create_chunk_records_has_parent_child_metadata() -> None:
    document = {
        'doc_id': 'doc1',
        'text': 'Sentence one. Sentence two.',
        'source_rel': 'raw/txt/demo.txt',
        'file_name': 'demo.txt',
        'file_type': 'txt',
    }
    records = create_chunk_records(document, chunk_size=80, chunk_overlap=10)
    assert records
    first = records[0]
    assert first['parent_doc_id'] == 'doc1'
    assert 'sentence_count' in first
    assert 'start_sentence_index' in first
    assert 'end_sentence_index' in first
