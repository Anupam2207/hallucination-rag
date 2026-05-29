import re

from src.detection.claim_extractor import ClaimExtractor
from src.detection.detector import HallucinationDetector
from src.detection.nli_verifier import NLIVerifier
from src.detection.support_scorer import SupportScorer
from src.pipeline import HallucinationRAGPipeline
from src.retrieval.evidence_intent import annotate_evidence
from src.retrieval.hybrid_retriever import HybridRetriever


class _FakeRetriever:
    def __init__(self, rows):
        self.rows = rows

    def retrieve(self, query, top_k=None):
        return list(self.rows)[: top_k or len(self.rows)]


def _row(chunk_id, text, title, file_name, *, doc_id=None, sparse_score=0.0, similarity=0.0, rank=1, pos=0):
    doc_id = doc_id or file_name.replace('.', '_')
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "doc_id": doc_id,
            "parent_doc_id": doc_id,
            "document_title": title,
            "file_name": file_name,
            "file_type": file_name.rsplit('.', 1)[-1],
            "source_rel": f"raw/txt/{file_name}",
            "section_name": "Body",
            "chunk_position": pos,
            "importance_score": 0,
        },
        "dense_rank": rank if similarity else None,
        "dense_similarity": similarity,
        "sparse_rank": rank if sparse_score else None,
        "sparse_score": sparse_score,
        "final_score": similarity,
    }


def test_exact_title_query_ranks_dns_above_generic_system_chunks():
    dns = _row(
        "dns0",
        "The Domain Name System, or DNS, translates human-readable domain names into IP addresses.",
        "Domain Name System",
        "computer_networks_dns.txt",
        sparse_score=12.0,
        rank=1,
    )
    vehicle = _row(
        "vehicle0",
        "A vehicle safety system may have a name that suggests full automation.",
        "Vehicle Safety Systems",
        "vehicle_safety_systems.pdf",
        similarity=0.75,
        sparse_score=5.0,
        rank=1,
    )
    retriever = HybridRetriever(dense_retriever=_FakeRetriever([vehicle]), sparse_retriever=_FakeRetriever([dns, vehicle]))
    results = retriever.retrieve("Domain Name System", top_k=2)
    assert results[0]["metadata"]["file_name"] == "computer_networks_dns.txt"


def test_topic_lock_keeps_mfa_chunks_above_adjacent_security_chunks():
    mfa_rows = [
        _row(f"mfa{i}", f"Multi-factor authentication evidence sentence {i}. MFA uses factors.", "Multi-Factor Authentication", "cybersecurity_mfa.txt", doc_id="mfa", sparse_score=10 - i, rank=i + 1, pos=i)
        for i in range(4)
    ]
    phishing = _row(
        "phish",
        "Phishing pages may ask users for MFA codes, but this document is about phishing.",
        "Cybersecurity Phishing",
        "cybersecurity_phishing.txt",
        doc_id="phish",
        similarity=0.7,
        sparse_score=9,
        rank=1,
        pos=0,
    )
    retriever = HybridRetriever(dense_retriever=_FakeRetriever([phishing]), sparse_retriever=_FakeRetriever([*mfa_rows, phishing]))
    results = retriever.retrieve("Explain Multi-factor authentication?", top_k=4)
    assert [item["metadata"]["file_name"] for item in results[:3]] == ["cybersecurity_mfa.txt"] * 3


def test_claim_extractor_drops_list_intro_heading():
    answer = """Common methods of MFA include:\n1. Something you know: passwords or PINs.\n2. Something you have: a phone or token."""
    claims = ClaimExtractor().extract_claims(answer)
    assert all(not claim.lower().startswith("common methods of mfa include") for claim in claims)
    assert any("Something you know" in claim for claim in claims)
    assert any("Something you have" in claim for claim in claims)


def test_mfa_paraphrase_claims_are_supported_or_weak():
    evidence = [
        "Multi-factor authentication, or MFA, is a security method that requires users to prove identity using more than one type of factor. Common factors include something the user knows, such as a password; something the user has, such as a phone or hardware token; and something the user is, such as a biometric trait. MFA improves security because a stolen password alone is not enough to access an account. Attackers must also defeat the second factor."
    ]
    detector = HallucinationDetector(nli_verifier=NLIVerifier(enabled=False))
    answer = (
        "Multi-Factor Authentication requires two or more forms of verification. "
        "Something you know includes passwords, PINs, or security questions. "
        "Something you have includes smartphones, tokens, or physical devices. "
        "Something you are includes fingerprints, facial recognition, or voice recognition. "
        "MFA makes stolen passwords alone insufficient for account access."
    )
    result = detector.detect(answer, evidence)
    assert result["unsupported_count"] == 0
    assert result["supported_count"] + result["weak_count"] == result["claim_count"]


def test_colbert_hallucinated_definition_and_method_stay_unsupported():
    evidence = [
        "ColBERT is a neural information retrieval model. The name ColBERT stands for Contextualized Late Interaction over BERT. It was introduced by Omar Khattab and Matei Zaharia in 2020."
    ]
    detector = HallucinationDetector(nli_verifier=NLIVerifier(enabled=False))
    result = detector.detect("ColBERT stands for Collaborative BERT. ColBERT uses collaborative filtering.", evidence)
    assert result["unsupported_count"] == result["claim_count"]


def test_evidence_fallback_correction_is_complete_and_verifies():
    evidence = [
        annotate_evidence({"text": "Multi-factor authentication, or MFA, is a security method that requires users to prove identity using more than one type of factor. Common factors include something the user knows, such as a password; something the user has, such as a phone or hardware token; and something the user is, such as a biometric trait. MFA improves security because a stolen password alone is not enough to access an account. Attackers must also defeat the second factor.", "metadata": {"document_title": "Multi-Factor Authentication", "file_name": "cybersecurity_mfa.txt", "file_type": "txt", "section_name": "Body"}}, query="Explain Multi-factor authentication?")
    ]
    answer = HallucinationRAGPipeline._fallback_answer_from_evidence("Explain Multi-factor authentication?", evidence)
    assert 3 <= len(re.findall(r"[.!?](?:\s|$)", answer)) <= 5
    result = HallucinationDetector(nli_verifier=NLIVerifier(enabled=False)).detect(answer, evidence)
    assert result["unsupported_count"] == 0
