from src.retrieval.hybrid_retriever import reciprocal_rank_fusion
from src.retrieval.sparse_retriever import BM25SparseRetriever


def test_rrf_promotes_chunk_present_in_both_lists() -> None:
    dense = [
        {'chunk_id': 'a', 'text': 'dense only'},
        {'chunk_id': 'b', 'text': 'both'},
    ]
    sparse = [
        {'chunk_id': 'b', 'text': 'both', 'sparse_score': 3.0},
        {'chunk_id': 'c', 'text': 'sparse only', 'sparse_score': 2.0},
    ]
    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=3)
    assert fused[0]['chunk_id'] == 'b'
    assert fused[0]['dense_rank'] == 2
    assert fused[0]['sparse_rank'] == 1
    assert len({item['chunk_id'] for item in fused}) == len(fused)


def test_bm25_sparse_retriever_finds_exact_keyword() -> None:
    records = [
        {'chunk_id': 'rag', 'text': 'Retrieval augmented generation uses evidence.', 'doc_id': 'd1', 'chunk_index': 0},
        {'chunk_id': 'mars', 'text': 'Mars is a planet.', 'doc_id': 'd2', 'chunk_index': 0},
    ]
    retriever = BM25SparseRetriever(records=records)
    results = retriever.retrieve('retrieval generation', top_k=1)
    assert results[0]['chunk_id'] == 'rag'
    assert results[0]['sparse_rank'] == 1
    assert results[0]['sparse_score'] > 0
