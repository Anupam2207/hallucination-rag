from src.retrieval.query_focus import definition_score


def test_definition_like_chunk_beats_experiment_chunk():
    query = "What is ColBERT?"
    definition = "ColBERT is a neural retrieval model that uses contextualized late interaction for ranking passages."
    rq_text = "RQ1: Beyond re-ranking, can ColBERT support retrieval directly? Datasets and metrics are discussed."
    assert definition_score(query, definition, {"section_name": "Introduction"}) > definition_score(query, rq_text, {"section_name": "Results"})
