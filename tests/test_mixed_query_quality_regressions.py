import numpy as np

from src.detection.claim_extractor import ClaimExtractor
from src.detection.support_scorer import SupportScorer
from src.generation.answer_sentence_selector import select_answer_sentences
from src.pipeline import HallucinationRAGPipeline


class LowSimilarityEmbedder:
    def encode(self, texts, normalize=True):
        vectors = []
        for index, _ in enumerate(texts):
            vectors.append([1.0, 0.0] if index == 0 else [0.0, 1.0])
        return np.array(vectors)


def _fake_pipeline():
    return object.__new__(HallucinationRAGPipeline)


def test_black_hole_definition_prefers_definition_sentence():
    evidence = [
        {
            "text": "In 2019, the Event Horizon Telescope collaboration released the first image of a black hole. The image showed the shadow of the black hole in M87.",
            "metadata": {"document_title": "Black Holes", "section_name": "Background"},
        },
        {
            "text": "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape after crossing the event horizon.",
            "metadata": {"document_title": "Black Holes", "section_name": "Introduction"},
        },
    ]
    answer = _fake_pipeline()._fallback_answer_from_evidence("What is a black hole?", evidence)
    assert answer.startswith("A black hole is a region of spacetime")
    assert not answer.startswith("In 2019")


def test_crop_rotation_does_not_include_drip_irrigation():
    evidence = [
        {
            "text": "Crop rotation is the practice of growing different crops on the same field in a planned sequence across seasons or years. It helps maintain soil fertility, reduce pest pressure, and improve long-term yields. Filters are important because small emitters can clog in drip irrigation systems.",
            "metadata": {"document_title": "Crop Rotation", "section_name": "Introduction"},
        }
    ]
    answer = _fake_pipeline()._fallback_answer_from_evidence("Explain crop rotation", evidence)
    assert "Crop rotation" in answer
    assert "soil fertility" in answer
    assert "emitters" not in answer
    assert "filters" not in answer.lower()
    assert "clog" not in answer.lower()


def test_eiffel_tower_avoids_meta_instruction_sentence():
    evidence = [
        {
            "text": "When answering questions about the Eiffel Tower, it is important not to confuse it with other Paris landmarks. The Eiffel Tower is an iron lattice tower in Paris, France. It was designed by engineer Gustave Eiffel's company for the 1889 Exposition Universelle.",
            "metadata": {"document_title": "Eiffel Tower", "section_name": "Introduction"},
        }
    ]
    answer = _fake_pipeline()._fallback_answer_from_evidence("What is the Eiffel Tower?", evidence)
    assert not answer.lower().startswith("when answering questions")
    assert "iron lattice tower in Paris" in answer


def test_heading_question_not_extracted_as_claim():
    claims = ClaimExtractor().extract_claims("What is a Black Hole?\nA black hole is a region where gravity is strong.")
    assert claims == ["A black hole is a region where gravity is strong."]


def test_crispr_definition_paraphrase_supported():
    scorer = SupportScorer(embedder=LowSimilarityEmbedder())
    result = scorer.score_claim_against_evidence(
        "CRISPR is a gene-editing method for making targeted DNA changes.",
        [{"text": "CRISPR gene editing is a biotechnology method used to make targeted changes to DNA."}],
    )
    assert result["score"] >= 0.45
    assert not result["rule_flags"]


def test_black_hole_definition_paraphrase_supported():
    scorer = SupportScorer(embedder=LowSimilarityEmbedder())
    result = scorer.score_claim_against_evidence(
        "A black hole has gravity so strong that light cannot escape.",
        [{"text": "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape after crossing the event horizon."}],
    )
    assert result["score"] >= 0.45
    assert not result["rule_flags"]


def test_kubernetes_partial_answerability():
    evidence = [
        {
            "text": "Docker popularized container workflows, while Kubernetes is commonly used to orchestrate containers across clusters.",
            "metadata": {"document_title": "Containers", "section_name": "Body"},
            "final_score": 0.7,
        }
    ]
    status, warning = _fake_pipeline()._definition_answerability_status("Explain Kubernetes", evidence)
    assert status == "partially_answerable"
    assert warning == "limited_answer_specific_evidence"
    answer = _fake_pipeline()._fallback_answer_from_evidence("Explain Kubernetes", evidence, status)
    assert answer.startswith("The knowledge base provides limited information about Kubernetes")


def test_sentence_selector_prefers_answer_centrality_over_evidence_order():
    evidence = [
        {
            "text": "In 2019, the Event Horizon Telescope collaboration released the first image of a black hole.",
            "metadata": {"document_title": "Black Holes", "section_name": "Background"},
        },
        {
            "text": "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape after crossing the event horizon.",
            "metadata": {"document_title": "Black Holes", "section_name": "Introduction"},
        },
    ]
    selected = select_answer_sentences("What is a black hole?", evidence, max_sentences=2)
    assert selected[0].startswith("A black hole is a region of spacetime")
