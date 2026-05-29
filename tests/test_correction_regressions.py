from scripts.evaluation_common import deterministic_correct_answer, make_detector
from src.generation.correction import AnswerCorrector, INSUFFICIENT_EVIDENCE_RESPONSE


def test_correction_keeps_supported_answer_unchanged():
    detector = make_detector(enable_nli=False, rules=True)
    answer = "RAG combines retrieval and generation."
    corrected, _ = deterministic_correct_answer(
        detector,
        answer,
        [{"text": "RAG combines retrieval and generation.", "metadata": {"section_name": "Introduction"}}],
    )
    assert corrected == answer


def test_relation_correction_preserves_subject_for_who_question():
    answer = AnswerCorrector._deterministic_relation_answer(
        "Who introduced ColBERT?",
        [{
            "text": "ColBERT was introduced by Omar Khattab and Matei Zaharia in 2020 for neural retrieval.",
            "metadata": {"section_name": "Introduction", "file_type": "pdf", "source_rel": "ColBERT.pdf"},
            "dense_similarity": 0.95,
        }],
    )
    assert answer == "ColBERT was introduced by Omar Khattab and Matei Zaharia in 2020."


def test_malformed_or_unsupported_correction_is_detected():
    detector = make_detector(enable_nli=False, rules=True)
    corrected, _ = deterministic_correct_answer(
        detector,
        "RAG was introduced in 2021.",
        [{"text": "RAG combines retrieval with generation.", "metadata": {"section_name": "Introduction"}}],
    )
    assert corrected == INSUFFICIENT_EVIDENCE_RESPONSE
