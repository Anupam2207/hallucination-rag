from src.ingestion.chunker import chunk_text


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
