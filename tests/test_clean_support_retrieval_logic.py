import numpy as np

from src.detection.detector import HallucinationDetector
from src.detection.support_scorer import SupportScorer
from src.retrieval.hybrid_retriever import HybridRetriever


class FakeEmbedder:
    def encode(self, texts, normalize=True):
        vectors = []
        for text in texts:
            text_l = text.lower()
            if ("rag" in text_l or "retrieval" in text_l) and "external" in text_l:
                vectors.append(np.array([1.0, 0.0, 0.0]))
            elif "rag" in text_l or "retrieval" in text_l:
                vectors.append(np.array([0.9, 0.1, 0.0]))
            elif "colbert" in text_l:
                vectors.append(np.array([0.0, 1.0, 0.0]))
            else:
                vectors.append(np.array([0.0, 0.0, 1.0]))
        return np.array(vectors)


class EntailmentNLI:
    enabled = True
    def verify(self, claim, evidence):
        return {
            "enabled": True,
            "available": True,
            "label": "entailment",
            "score": 0.9,
            "scores": {"entailment": 0.9, "neutral": 0.08, "contradiction": 0.02},
            "error": None,
        }


class UnavailableNLI:
    enabled = True
    def verify(self, claim, evidence):
        return {
            "enabled": True,
            "available": False,
            "label": "unavailable",
            "score": None,
            "scores": {},
            "error": "model missing",
        }


class FakeExtractor:
    def extract_claims(self, answer):
        return [answer]


def test_support_scorer_exposes_semantic_and_lexical_components():
    scorer = SupportScorer(embedder=FakeEmbedder())
    result = scorer.score_claim_against_evidence(
        "Retrieval-Augmented Generation connects a language model to external knowledge.",
        [{"text": "Retrieval-Augmented Generation, or RAG, connects a language model to an external knowledge source."}],
    )
    assert result["semantic_score"] > 0.5
    assert result["lexical_score"] > 0.4
    assert result["score"] >= 0.45
    assert result["rule_flags"] == []


def test_detector_uses_clean_semantic_lexical_nli_fusion():
    detector = HallucinationDetector(
        extractor=FakeExtractor(),
        support_scorer=SupportScorer(embedder=FakeEmbedder()),
        nli_verifier=EntailmentNLI(),
    )
    detector.nli_min_similarity_to_run = 0.0
    result = detector.detect(
        "Retrieval-Augmented Generation connects a language model to external knowledge.",
        [{"text": "Retrieval-Augmented Generation, or RAG, connects a language model to an external knowledge source."}],
    )
    claim = result["claims"][0]
    assert claim["nli_label"] == "entailment"
    assert claim["entailment_score"] >= 0.85
    assert claim["support_score"] >= 0.70
    assert claim["label"] == "supported"


def test_nli_unavailable_does_not_create_false_unsupported_when_evidence_is_strong():
    detector = HallucinationDetector(
        extractor=FakeExtractor(),
        support_scorer=SupportScorer(embedder=FakeEmbedder()),
        nli_verifier=UnavailableNLI(),
    )
    detector.nli_min_similarity_to_run = 0.0
    result = detector.detect(
        "Retrieval-Augmented Generation connects a language model to external knowledge.",
        [{"text": "Retrieval-Augmented Generation, or RAG, connects a language model to an external knowledge source."}],
    )
    claim = result["claims"][0]
    assert claim["nli_label"] == "unavailable"
    assert claim["nli_available"] is False
    assert claim["label"] in {"supported", "weak_support"}


class FakeDense:
    def retrieve(self, query, top_k=4):
        return [
            {
                "chunk_id": "txt_rag",
                "text": "RAG text from old txt index.",
                "metadata": {"file_type": "txt", "source_rel": "raw/txt/rag.txt", "document_title": "RAG"},
                "dense_similarity": 0.99,
            },
            {
                "chunk_id": "pdf_rag",
                "text": "Retrieval-Augmented Generation, or RAG, connects a language model to an external knowledge source.",
                "metadata": {"file_type": "pdf", "source_rel": "raw/pdf/rag_demo.pdf", "document_title": "Retrieval-Augmented Generation"},
                "dense_similarity": 0.62,
            },
        ]


class FakeSparse:
    def retrieve(self, query, top_k=4):
        return [
            {
                "chunk_id": "pdf_rag",
                "text": "Retrieval-Augmented Generation, or RAG, connects a language model to an external knowledge source.",
                "metadata": {"file_type": "pdf", "source_rel": "raw/pdf/rag_demo.pdf", "document_title": "Retrieval-Augmented Generation"},
                "sparse_score": 5.0,
            }
        ]


def test_hybrid_retriever_filters_non_pdf_runtime_sources():
    retriever = HybridRetriever(dense_retriever=FakeDense(), sparse_retriever=FakeSparse())
    results = retriever.retrieve("What is retrieval-augmented generation?", top_k=2)
    assert results
    assert all(item["metadata"].get("file_type") == "pdf" for item in results)
    assert results[0]["chunk_id"] == "pdf_rag"
