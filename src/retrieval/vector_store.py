import chromadb
from pathlib import Path


class ChromaVectorStore:

    def __init__(
        self,
        persist_dir="db/chroma",
        collection_name="rag_knowledge"
    ):

        Path(persist_dir).mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=persist_dir
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(
        self,
        ids,
        texts,
        embeddings,
        metadatas
    ):

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    def count(self):
        return self.collection.count()