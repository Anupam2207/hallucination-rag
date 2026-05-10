import json
import pandas as pd
from pathlib import Path

from src.retrieval.embedder import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore


BASE_DIR = Path(__file__).resolve().parents[1]

CHUNK_FILE = BASE_DIR / "data" / "chunks" / "chunks.jsonl"
META_FILE = BASE_DIR / "data" / "chunks" / "chunk_metadata.csv"


def load_chunks():

    chunks = []

    with open(CHUNK_FILE, "r", encoding="utf-8") as f:

        for line in f:
            chunks.append(json.loads(line))

    metadata = pd.read_csv(META_FILE)

    return chunks, metadata


def main():

    print("Loading chunks...")

    chunks, metadata = load_chunks()

    texts = [x["text"] for x in chunks]

    ids = [str(x["chunk_id"]) for x in chunks]

    metadatas = metadata.to_dict("records")

    print("Generating embeddings...")

    embedder = EmbeddingModel()

    embeddings = embedder.encode(texts)

    print("Building vector store...")

    vector_store = ChromaVectorStore()

    vector_store.add_documents(
        ids=ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Index complete. Total vectors: {vector_store.count()}"
    )


if __name__ == "__main__":
    main()