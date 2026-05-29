from src.detection.claim_extractor import ClaimExtractor
from src.detection.detector import HallucinationDetector

MFA_EVIDENCE = (
    "Multi-factor authentication, or MFA, is a security method that requires users to prove identity "
    "using more than one type of factor. Common factors include something the user knows, such as a password; "
    "something the user has, such as a phone or hardware token; and something the user is, such as a biometric trait. "
    "MFA improves security because a stolen password alone is not enough to access an account."
)

COLBERT_EVIDENCE = (
    "ColBERT is a neural information retrieval model designed for accurate passage retrieval with efficient "
    "late interaction between query and document representations. The name ColBERT stands for Contextualized "
    "Late Interaction over BERT."
)


def _label(claim: str, evidence: str = MFA_EVIDENCE) -> str:
    detector = HallucinationDetector()
    return detector.detect(claim, [evidence])["claims"][0]["label"]


def test_mfa_paraphrases_are_supported_or_weak_supported():
    claims = [
        "MFA requires two or more verification factors.",
        "Something you know: Password, PIN, or passphrase.",
        "Something you have: Smartcard, token, or mobile device.",
        "Something you are: Biometric data such as fingerprint, face, or voice.",
        "MFA adds security against unauthorized access or stolen credentials.",
    ]
    for claim in claims:
        assert _label(claim) in {"supported", "weak_support"}, claim


def test_mfa_location_based_factor_is_not_supported_without_evidence():
    assert _label("Somewhere you are: Location-based authentication.") == "unsupported"


def test_mfa_specific_applications_are_not_over_supported_without_evidence():
    assert _label("MFA is used in online banking, email accounts, and enterprise networks.") in {"weak_support", "unsupported"}


def test_colbert_negative_controls_remain_unsupported():
    assert _label("ColBERT stands for Collaborative BERT.", COLBERT_EVIDENCE) == "unsupported"
    assert _label("ColBERT uses collaborative filtering.", COLBERT_EVIDENCE) == "unsupported"


def test_mfa_list_headings_are_not_claims():
    answer = """
    Typical MFA methods include:
    1. Something you know: Password, PIN, or passphrase.
    2. Something you have: Smartcard, token, or mobile device.
    3. Something you are: Biometric data such as fingerprint, face, or voice.
    """
    claims = ClaimExtractor().extract_claims(answer)
    assert "Typical MFA methods include:." not in claims
    assert "Common MFA methods include:." not in claims
    assert any("Something you know" in claim for claim in claims)
    assert any("Something you have" in claim for claim in claims)
    assert any("Something you are" in claim for claim in claims)
