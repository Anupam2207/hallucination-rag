from src.generation.base_answer import BaseAnswerGenerator
from src.generation.correction import AnswerCorrector

from src.retrieval.retriever import SemanticRetriever

from src.detection.detector import HallucinationDetector

from src.evaluation.metrics import HallucinationMetrics


def main():

    query = "What is retrieval augmented generation?"

    print(f"\nQUERY:\n{query}\n")

    generator = BaseAnswerGenerator()

    raw_answer = generator.generate_answer(
        query
    )

    retriever = SemanticRetriever()

    retrieval = retriever.retrieve(
        query
    )

    evidence = retrieval[
        "documents"
    ][0]

    detector = HallucinationDetector()

    raw_results = detector.detect(
        raw_answer,
        evidence
    )

    corrector = AnswerCorrector()

    corrected_answer = corrector.correct(
        query,
        evidence
    )

    corrected_results = detector.detect(
        corrected_answer,
        evidence
    )

    before_support = (
        HallucinationMetrics.compute_support_ratio(
            raw_results
        )
    )

    after_support = (
        HallucinationMetrics.compute_support_ratio(
            corrected_results
        )
    )

    hallucination_rate = (
        HallucinationMetrics.compute_hallucination_rate(
            raw_results
        )
    )

    improvement = (
        HallucinationMetrics.compute_improvement(
            before_support,
            after_support
        )
    )

    print("\nEVALUATION:\n")

    print(
        f"Before Support Ratio: {before_support}"
    )

    print(
        f"After Support Ratio: {after_support}"
    )

    print(
        f"Hallucination Rate: {hallucination_rate}"
    )

    print(
        f"Improvement: {improvement}"
    )


if __name__ == "__main__":
    main()