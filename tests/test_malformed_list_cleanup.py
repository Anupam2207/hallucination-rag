from src.utils.text_cleaning import clean_malformed_lists


def test_clean_malformed_numbered_list_artifacts():
    text = "The causes are: 1. Training data issues 2. 3. Retrieval mismatch"
    cleaned = clean_malformed_lists(text)
    assert "2. 3." not in cleaned
    assert not cleaned.endswith("1.")
    assert "Training data issues" in cleaned
    assert "Retrieval mismatch" in cleaned
