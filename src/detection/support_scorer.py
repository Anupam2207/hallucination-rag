from sklearn.metrics.pairwise import cosine_similarity
from src.retrieval.embedder import EmbeddingModel


class SupportScorer:

    def __init__(self):

        self.embedder = EmbeddingModel()

    def score_claim(
        self,
        claim: str,
        evidence_list
    ):

        claim_emb = self.embedder.encode(
            claim
        )

        evidence_embs = self.embedder.encode(
            evidence_list
        )

        similarities = cosine_similarity(
            claim_emb,
            evidence_embs
        )[0]

        best_score = float(
            similarities.max()
        )

        return best_score