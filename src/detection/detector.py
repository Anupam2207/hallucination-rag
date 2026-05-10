from src.detection.claim_extractor import ClaimExtractor
from src.detection.support_scorer import SupportScorer


class HallucinationDetector:

    def __init__(
        self,
        threshold=0.45
    ):

        self.threshold = threshold

        self.extractor = ClaimExtractor()

        self.scorer = SupportScorer()

    def detect(
        self,
        answer: str,
        evidence_list
    ):

        claims = self.extractor.extract_claims(
            answer
        )

        results = []

        for claim in claims:

            score = self.scorer.score_claim(
                claim,
                evidence_list
            )

            label = (
                "SUPPORTED"
                if score >= self.threshold
                else "POTENTIAL_HALLUCINATION"
            )

            results.append({
                "claim": claim,
                "score": round(score, 3),
                "label": label
            })

        return results