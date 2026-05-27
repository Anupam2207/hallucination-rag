from src.retrieval.query_focus import source_quality


def test_clean_pdf_has_higher_source_quality_than_research_pdf():
    clean = source_quality({"file_name": "colbert_retrieval_model.pdf", "file_type": "pdf"})
    research = source_quality({"file_name": "ColBERT.pdf", "file_type": "pdf"})
    assert clean > research
    assert clean == 1.0


def test_text_files_are_high_quality():
    assert source_quality({"file_name": "truthfulqa.txt", "file_type": "txt"}) >= 0.9
