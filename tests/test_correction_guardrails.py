from src.detection.detector import HallucinationDetector
from src.detection.nli_verifier import NLIVerifier
from src.generation.correction import INSUFFICIENT_EVIDENCE_RESPONSE
from src.pipeline import HallucinationRAGPipeline


MFA_EVIDENCE = (
    "Multi-factor authentication, or MFA, is a security method that requires users "
    "to prove identity using more than one type of factor. Common factors include "
    "something the user knows, such as a password; something the user has, such as "
    "a phone or hardware token; and something the user is, such as a biometric trait. "
    "MFA improves security because a stolen password alone is not enough to access an account. "
    "Attackers must also defeat the second factor. Common MFA methods include authenticator apps, "
    "hardware security keys, SMS codes, email codes, and biometric prompts."
)

RAG_EVIDENCE = (
    "Retrieval-Augmented Generation, usually called RAG, connects a language model to a retrieval system. "
    "The retriever finds relevant external knowledge before the model generates an answer. "
    "The generator uses the retrieved evidence to produce a more grounded response. "
    "RAG can improve factual grounding because the answer is based on retrieved documents instead of only model parameters."
)


class _FakeRetriever:
    def __init__(self, evidence_text: str | None):
        self.evidence_text = evidence_text

    def retrieve(self, query, top_k=None):
        if not self.evidence_text:
            return []
        return [
            {
                "text": self.evidence_text,
                "metadata": {
                    "document_title": "Multi-Factor Authentication" if "MFA" in self.evidence_text else "Retrieval-Augmented Generation",
                    "file_name": "cybersecurity_mfa.txt" if "MFA" in self.evidence_text else "ai_rag_overview.txt",
                    "file_type": "txt",
                    "section_name": "Body",
                    "source_rel": "raw/txt/test.txt",
                    "importance_score": 2,
                },
                "final_score": 1.0,
                "weighted_score": 1.0,
                "sparse_score": 10.0,
                "sparse_rank": 1,
            }
        ]


class _FakeGenerator:
    def __init__(self, raw_answer: str):
        self.raw_answer = raw_answer

    def generate_answer(self, query):
        return self.raw_answer


class _FakeCorrector:
    def __init__(self, answer: str = INSUFFICIENT_EVIDENCE_RESPONSE, fail_if_called: bool = False):
        self.answer = answer
        self.fail_if_called = fail_if_called
        self.called = False

    def correct(self, query, raw_answer, evidence):
        self.called = True
        if self.fail_if_called:
            raise AssertionError("correction should not run for a fully supported raw answer")
        return self.answer


def _make_pipeline(raw_answer: str, evidence_text: str | None, corrector: _FakeCorrector) -> HallucinationRAGPipeline:
    pipe = HallucinationRAGPipeline.__new__(HallucinationRAGPipeline)
    pipe.retriever = _FakeRetriever(evidence_text)
    pipe.generator = _FakeGenerator(raw_answer)
    pipe.corrector = corrector
    pipe.detector = HallucinationDetector(nli_verifier=NLIVerifier(enabled=False))
    pipe.retrieval_mode = "hybrid"
    pipe.answerability_threshold = 0.1
    return pipe


def test_supported_mfa_raw_answer_is_preserved_not_replaced_with_insufficient():
    raw_answer = (
        "Multi-Factor Authentication (MFA) requires users to provide multiple forms of verification. "
        "Common factors include something the user knows, such as passwords or PINs; something the user has, "
        "such as a phone or hardware token; and something the user is, such as a biometric trait. "
        "MFA improves security because a stolen password alone is not enough to access an account."
    )
    corrector = _FakeCorrector(fail_if_called=True)
    result = _make_pipeline(raw_answer, MFA_EVIDENCE, corrector).run("Explain Multi-factor authentication?", top_k=6)

    assert result["raw_detection"]["unsupported_count"] == 0
    assert result["corrected_answer"] == raw_answer
    assert result["corrected_detection"] == result["raw_detection"]
    assert result["corrected_answer"] != INSUFFICIENT_EVIDENCE_RESPONSE
    assert result["metrics"]["correction_status"] == "no_change"
    assert not corrector.called


def test_supported_rag_raw_answer_is_preserved_or_remains_supported():
    raw_answer = (
        "Retrieval-Augmented Generation, or RAG, connects a language model to a retrieval system. "
        "The retriever finds relevant external knowledge before the model generates an answer. "
        "The generator uses retrieved evidence to produce a more grounded response."
    )
    corrector = _FakeCorrector(fail_if_called=True)
    result = _make_pipeline(raw_answer, RAG_EVIDENCE, corrector).run("What is retrieval-augmented generation?", top_k=4)

    assert result["raw_detection"]["unsupported_count"] == 0
    assert result["corrected_answer"] != INSUFFICIENT_EVIDENCE_RESPONSE
    assert result["corrected_detection"]["unsupported_count"] == 0
    assert result["metrics"]["correction_status"] == "no_change"
    assert not corrector.called


def test_unsupported_rag_answer_gets_evidence_fallback_instead_of_insufficient():
    raw_answer = (
        "Retrieval-Augmented Generation (RAG) combines retrieval and generation, then fine-tunes a generator "
        "on the retrieved texts. RAG has been shown to be effective for text summarization and document retrieval."
    )
    corrector = _FakeCorrector(answer=INSUFFICIENT_EVIDENCE_RESPONSE)
    result = _make_pipeline(raw_answer, RAG_EVIDENCE, corrector).run("What is retrieval-augmented generation?", top_k=4)

    assert corrector.called
    assert result["raw_detection"]["unsupported_count"] > 0
    assert result["corrected_answer"] != INSUFFICIENT_EVIDENCE_RESPONSE
    assert "fine-tun" not in result["corrected_answer"].lower()
    assert "summarization" not in result["corrected_answer"].lower()
    assert result["corrected_detection"]["unsupported_count"] == 0


def test_insufficient_evidence_query_still_returns_insufficient_response():
    raw_answer = "The answer is not supported by any retrieved evidence."
    corrector = _FakeCorrector(answer="This should not be used.")
    result = _make_pipeline(raw_answer, None, corrector).run("Who invented the nonexistent WidgetNet model?", top_k=4)

    assert result["answerable"] is False
    assert "does not provide evidence" in result["corrected_answer"]
    assert not corrector.called
