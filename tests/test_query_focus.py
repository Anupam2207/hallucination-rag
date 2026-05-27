from src.retrieval.query_focus import extract_query_focus, has_topical_match


def test_who_introduced_rag_keeps_rag_as_core_entity():
    focus = extract_query_focus("Who introduced RAG?")
    assert "rag" in focus["core_entity_terms"]
    assert "introduced" in focus["relation_terms"]
    assert "introduced" not in focus["core_entity_terms"]


def test_what_is_colbert_keeps_colbert_as_core_entity():
    focus = extract_query_focus("What is ColBERT?")
    assert "colbert" in focus["core_entity_terms"]
    assert "what" not in focus["core_entity_terms"]


def test_relation_terms_alone_do_not_satisfy_topical_match():
    focus = extract_query_focus("Who introduced RAG?")
    unrelated = "Ancient architecture developed many temple styles."
    assert not has_topical_match(unrelated, {"file_name": "architecture.pdf"}, focus)
    related = "Retrieval-Augmented Generation, also called RAG, was proposed for knowledge-intensive NLP."
    assert has_topical_match(related, {"file_name": "rag_history.pdf"}, focus)
