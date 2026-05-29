import json
import sys

from src.detection.claim_extractor import ClaimExtractor
from src.detection.detector import HallucinationDetector
from src.detection.nli_verifier import NLIVerifier
from src.ingestion.preprocess import preprocess_documents, preprocess_text
from src.pipeline import HallucinationRAGPipeline


MFA_EVIDENCE = (
    "Multi-factor authentication, or MFA, is a security method that requires users "
    "to prove identity using more than one type of factor. Common factors include "
    "something the user knows, such as a password; something the user has, such as "
    "a phone or hardware token; and something the user is, such as a biometric trait. "
    "MFA improves security because a stolen password alone is not enough to access "
    "an account. Attackers must also defeat the second factor. Common MFA methods "
    "include authenticator apps, hardware security keys, SMS codes, email codes, and "
    "biometric prompts."
)

COLBERT_EVIDENCE = (
    "ColBERT is a neural information retrieval model designed for accurate passage "
    "retrieval with efficient late interaction between query and document representations. "
    "The name ColBERT stands for Contextualized Late Interaction over BERT. It was "
    "introduced by Omar Khattab and Matei Zaharia in 2020."
)


def _detector() -> HallucinationDetector:
    return HallucinationDetector(nli_verifier=NLIVerifier(enabled=False))


def test_metadata_cleanup_removes_domain_labels_but_preserves_dns_phrase():
    cleaned = preprocess_text("# Multi-Factor Authentication Domain: Cybersecurity\nDomain: Cybersecurity\nMFA requires more than one factor.")
    assert "Domain: Cybersecurity" not in cleaned
    assert "MFA requires more than one factor" in cleaned

    docs = preprocess_documents([
        {
            "document_title": "Domain Name System",
            "text": "Domain Name System\nThe Domain Name System translates domain names into IP addresses.",
        }
    ])
    assert docs
    assert "Domain Name System translates domain names" in docs[0]["text"]


def test_claim_extractor_keeps_bullets_but_drops_mfa_list_heading():
    answer = """
    Common methods of MFA include:
    1. Something you know: passwords, PINs, or security questions.
    2. Something you have: smartphones, tokens, or physical devices.
    3. Something you are: fingerprints, facial recognition, or voice recognition.
    """
    claims = ClaimExtractor().extract_claims(answer)
    joined = "\n".join(claims).lower()
    assert "common methods of mfa include" not in joined
    assert any("Something you know" in claim for claim in claims)
    assert any("Something you have" in claim for claim in claims)
    assert any("Something you are" in claim for claim in claims)
    assert all(not claim.endswith(":.") for claim in claims)


def test_mfa_claims_are_supported_or_weak_individually():
    claims = [
        "Multi-Factor Authentication requires two or more forms of verification.",
        "Something you know includes passwords, PINs, or security questions.",
        "Something you have includes smartphones, tokens, or physical devices.",
        "Something you are includes fingerprints, facial recognition, or voice recognition.",
        "MFA makes stolen passwords alone insufficient for account access.",
    ]
    detector = _detector()
    for claim in claims:
        result = detector.detect(claim, [MFA_EVIDENCE])
        assert result["claim_count"] == 1
        assert result["claims"][0]["label"] in {"supported", "weak_support"}, result["claims"][0]


def test_colbert_negative_controls_remain_unsupported_individually():
    detector = _detector()
    for claim in [
        "ColBERT stands for Collaborative BERT.",
        "ColBERT uses collaborative filtering.",
    ]:
        result = detector.detect(claim, [COLBERT_EVIDENCE])
        assert result["claim_count"] == 1
        assert result["claims"][0]["label"] == "unsupported", result["claims"][0]


def test_repair_does_not_delete_supported_corrected_sentences():
    pipe = HallucinationRAGPipeline.__new__(HallucinationRAGPipeline)
    corrected_answer = (
        "Multi-factor authentication requires more than one type of factor. "
        "Common factors include passwords, phones or hardware tokens, and biometric traits. "
        "MFA improves security because a stolen password alone is not enough to access an account."
    )
    corrected_detection = _detector().detect(corrected_answer, [MFA_EVIDENCE])
    repaired, changed, removed = pipe._repair_corrected_answer(
        corrected_answer,
        corrected_detection,
        query="Explain Multi-factor authentication?",
    )
    assert not changed
    assert removed == []
    assert repaired == corrected_answer


def test_run_single_query_cli_nli_modes_forward_safely(monkeypatch, capsys):
    import scripts.run_single_query as run_single_query

    seen = []

    class DummyPipeline:
        def __init__(self, enable_nli=None):
            seen.append(enable_nli)

        def run(self, query, top_k=None):
            return {
                "query": query,
                "top_k": top_k,
                "raw_detection": {"claims": [{"nli_label": "not_run", "nli_error": None}]},
            }

    monkeypatch.setattr(run_single_query, "HallucinationRAGPipeline", DummyPipeline)
    for mode, expected in [("off", False), ("auto", None), ("on", True)]:
        monkeypatch.setattr(sys, "argv", ["run_single_query.py", "--query", "What is ColBERT?", "--top-k", "6", "--nli", mode])
        run_single_query.main()
        payload = json.loads(capsys.readouterr().out)
        assert payload["query"] == "What is ColBERT?"
        assert payload["top_k"] == 6
        assert seen[-1] is expected
