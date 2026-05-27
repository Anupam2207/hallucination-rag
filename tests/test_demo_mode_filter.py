from pathlib import Path

from src.ingestion.loaders import _should_skip_file


def test_demo_mode_excludes_research_pdf_by_default():
    skip, reason = _should_skip_file(Path("data/raw/pdf/ColBERT.pdf"))
    assert skip
    assert "demo_mode" in reason


def test_demo_mode_includes_clean_generated_pdf():
    skip, _reason = _should_skip_file(Path("data/raw/pdf/colbert_retrieval_model.pdf"))
    assert not skip
