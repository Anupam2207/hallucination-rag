from src.detection.detector import HallucinationDetector


class FakeExtractor:
    def extract_claims(self, answer):
        return [answer]


class FakeScorer:
    def lexical_overlap_score(self, claim, evidence):
        return 0.0

    def score_claim_against_evidence(self, claim, evidence_list):
        return {
            'claim': claim,
            'score': 0.12,
            'raw_similarity_score': 0.12,
            'best_evidence_index': 0,
            'best_evidence': evidence_list[0]['text'],
            'rule_flags': [],
            'factual_consistency': {'flags': [], 'details': {}},
        }


class FakeNLI:
    enabled = True

    def verify(self, claim, evidence):
        return {
            'enabled': True,
            'available': True,
            'label': 'contradiction',
            'score': 0.99,
            'scores': {'contradiction': 0.99, 'entailment': 0.0, 'neutral': 0.01},
            'error': None,
        }


def test_low_relevance_nli_contradiction_is_not_trusted():
    detector = HallucinationDetector(
        extractor=FakeExtractor(),
        support_scorer=FakeScorer(),
        nli_verifier=FakeNLI(),
    )
    detector.nli_min_similarity_to_run = 0.0
    result = detector.detect(
        'Formula 1 cars use hybrid power units.',
        [{'text': 'RAG combines retrieval with generation.'}],
    )
    claim = result['claims'][0]
    assert claim['nli_label'] == 'contradiction'
    assert claim['nli_adjusted_label'] == 'neutral'
    assert 'nli_contradiction_low_relevance_ignored' in claim['rule_flags']
    assert 'nli_contradiction' not in claim['rule_flags']
