from src.retrieval.retriever import SemanticRetriever


def main():

    retriever = SemanticRetriever()

    query = "What is retrieval augmented generation?"

    print(f"\nQuery: {query}\n")

    results = retriever.retrieve(query)

    docs = results["documents"][0]

    for i, doc in enumerate(docs):

        print(f"\nEvidence {i+1}:")
        print(doc)


if __name__ == "__main__":
    main()