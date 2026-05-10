from src.generation.base_answer import BaseAnswerGenerator
from src.retrieval.retriever import SemanticRetriever
from src.generation.correction import AnswerCorrector


def main():

    query = "What is retrieval augmented generation?"

    print(f"\nQUERY:\n{query}\n")

    generator = BaseAnswerGenerator()

    raw_answer = generator.generate_answer(
        query
    )

    print("RAW ANSWER:\n")
    print(raw_answer)

    retriever = SemanticRetriever()

    retrieval = retriever.retrieve(
        query
    )

    evidence_list = retrieval[
        "documents"
    ][0]

    print("\nRETRIEVED EVIDENCE:\n")

    for i, evidence in enumerate(
        evidence_list
    ):

        print(
            f"{i+1}. {evidence}"
        )

    corrector = AnswerCorrector()

    corrected = corrector.correct(
        query=query,
        evidence_list=evidence_list
    )

    print("\nCORRECTED ANSWER:\n")
    print(corrected)


if __name__ == "__main__":
    main()