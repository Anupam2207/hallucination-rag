class HallucinationMetrics:

    @staticmethod
    def compute_support_ratio(
        claim_results
    ):

        total = len(
            claim_results
        )

        if total == 0:
            return 0.0

        supported = sum(
            1
            for x in claim_results
            if x["label"] == "SUPPORTED"
        )

        return round(
            supported / total,
            3
        )

    @staticmethod
    def compute_hallucination_rate(
        claim_results
    ):

        total = len(
            claim_results
        )

        if total == 0:
            return 0.0

        hallucinated = sum(
            1
            for x in claim_results
            if x["label"] == "POTENTIAL_HALLUCINATION"
        )

        return round(
            hallucinated / total,
            3
        )

    @staticmethod
    def compute_improvement(
        before,
        after
    ):

        return round(
            after - before,
            3
        )