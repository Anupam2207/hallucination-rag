from src.generation.base_answer import BaseAnswerGenerator


def main():

    generator = BaseAnswerGenerator()

    query = "What is retrieval augmented generation?"

    print(f"\nQuery: {query}\n")

    answer = generator.generate_answer(query)

    print("Raw LLM Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()