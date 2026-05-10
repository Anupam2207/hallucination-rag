from src.generation.base_answer import BaseAnswerGenerator
from src.retrieval.retriever import SemanticRetriever
from src.detection.detector import HallucinationDetector


def main():

    query = "What is retrieval augmented generation?"

    print(f"\nQuery: {query}\n")

    generator = BaseAnswerGenerator()

    answer = generator.generate_answer(
        query
    )

    print("RAW ANSWER:\n")
    print(answer)

    retriever = SemanticRetriever()

    retrieved = retriever.retrieve(
        query
    )

    evidence_list = retrieved[
        "documents"
    ][0]

    detector = HallucinationDetector()

    results = detector.detect(
        answer,
        evidence_list
    )

    print("\nCLAIM ANALYSIS:\n")

    for row in results:

        print(
            f"{row['label']} | "
            f"{row['score']} | "
            f"{row['claim']}"
        )


if __name__ == "__main__":
    main()