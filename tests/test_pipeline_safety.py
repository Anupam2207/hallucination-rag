from src.pipeline import HallucinationRAGPipeline


class FakeRetriever:
    def retrieve(self, query, top_k=None):
        return [
            {
                "text": "Retrieval-Augmented Generation combines retrieval with generation.",
                "dense_similarity": 0.67,
                "metadata": {"source_rel": "raw/txt/rag.txt"},
            }
        ]


class FakeGenerator:
    def generate_answer(self, query):
        return "I couldn't find any information on RAG being introduced in 2021."


class FakeCorrector:
    def correct(self, query, raw_answer, evidence):
        return "RAG was introduced in 2021."


class FakeDetector:
    def detect(self, answer, evidence):
        if "2021" in answer and "does not provide evidence" not in answer:
            return {
                "claims": [
                    {
                        "claim": "RAG was introduced in 2021.",
                        "label": "unsupported",
                        "support_score": 0.35,
                        "rule_flags": ["claim_year_not_supported_by_evidence"],
                    }
                ],
                "claim_count": 1,
                "supported_count": 0,
                "weak_count": 0,
                "unsupported_count": 1,
                "support_ratio": 0.0,
                "hallucination_rate": 1.0,
                "average_support_score": 0.35,
            }
        return {
            "claims": [],
            "claim_count": 0,
            "supported_count": 0,
            "weak_count": 0,
            "unsupported_count": 0,
            "support_ratio": 0.0,
            "hallucination_rate": 0.0,
            "average_support_score": 0.0,
        }


def _fake_pipeline():
    pipe = object.__new__(HallucinationRAGPipeline)
    pipe.retriever = FakeRetriever()
    pipe.generator = FakeGenerator()
    pipe.corrector = FakeCorrector()
    pipe.detector = FakeDetector()
    pipe.retrieval_mode = "hybrid"
    pipe.answerability_threshold = 0.45
    return pipe


def test_specific_year_query_is_not_answerable_without_year_evidence():
    result = _fake_pipeline().run("RAG was introduced in 2021.")
    assert result["answerable"] is False
    assert "insufficient_evidence_for_specific_fact" in result["warnings"]
    assert "does not provide evidence" in result["corrected_answer"]


def test_repair_removes_critical_unsupported_corrected_sentence():
    pipe = _fake_pipeline()
    repaired, changed, removed = pipe._repair_corrected_answer(
        "RAG was introduced in 2021. RAG combines retrieval and generation.",
        {
            "claims": [
                {
                    "claim": "RAG was introduced in 2021.",
                    "label": "unsupported",
                    "rule_flags": ["claim_year_not_supported_by_evidence"],
                }
            ]
        },
    )
    assert changed is True
    assert "2021" not in repaired
    assert removed
