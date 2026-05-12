from src.config import get_config_value
from src.detection.claim_extractor import ClaimExtractor
from src.detection.support_scorer import SupportScorer
from src.retrieval.embedder import EmbeddingModel


class HallucinationDetector:
    def __init__(self, extractor: ClaimExtractor | None = None, scorer: SupportScorer | None = None) -> None:
        shared_embedder = EmbeddingModel()
        self.extractor = extractor or ClaimExtractor()
        self.scorer = scorer or SupportScorer(embedder=shared_embedder)
        self.support_threshold = float(
            get_config_value('settings', 'detection', 'similarity_support_threshold', default=0.55)
        )
        self.warning_threshold = float(
            get_config_value('settings', 'detection', 'similarity_warning_threshold', default=0.40)
        )

    def detect(self, answer: str, evidence_list: list[str]) -> dict:
        claims = self.extractor.extract_claims(answer)
        claim_results: list[dict] = []
        for claim in claims:
            score, best_index = self.scorer.score_claim(claim, evidence_list)
            if score >= self.support_threshold:
                label = 'supported'
            elif score >= self.warning_threshold:
                label = 'weak_support'
            else:
                label = 'unsupported'
            claim_results.append(
                {
                    'claim': claim,
                    'support_score': round(score, 4),
                    'label': label,
                    'best_evidence_index': best_index,
                    'best_evidence_text': evidence_list[best_index] if best_index is not None else None,
                }
            )

        supported = sum(item['label'] == 'supported' for item in claim_results)
        unsupported = sum(item['label'] == 'unsupported' for item in claim_results)
        weak = sum(item['label'] == 'weak_support' for item in claim_results)
        claim_count = len(claim_results)
        average_score = round(sum(item['support_score'] for item in claim_results) / claim_count, 4) if claim_count else 0.0
        support_ratio = round(supported / claim_count, 4) if claim_count else 0.0
        hallucination_rate = round(unsupported / claim_count, 4) if claim_count else 0.0

        return {
            'claims': claim_results,
            'claim_count': claim_count,
            'supported_count': supported,
            'weak_count': weak,
            'unsupported_count': unsupported,
            'support_ratio': support_ratio,
            'hallucination_rate': hallucination_rate,
            'average_support_score': average_score,
        }
