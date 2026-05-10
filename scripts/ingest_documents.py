from pathlib import Path
import json
import pandas as pd

from src.ingestion.loaders import load_documents
from src.ingestion.chunker import chunk_text


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"


def main():

    PROCESSED_DIR.mkdir(exist_ok=True)
    CHUNKS_DIR.mkdir(exist_ok=True)

    print("Loading documents...")

    documents = load_documents(RAW_DIR)

    print(f"Loaded {len(documents)} documents")

    all_chunks = []
    metadata = []

    chunk_id = 0

    for doc in documents:

        chunks = chunk_text(doc["text"])

        for chunk in chunks:

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk
            })

            metadata.append({
                "chunk_id": chunk_id,
                "source": doc["source"],
                "filename": doc["filename"]
            })

            chunk_id += 1

    chunk_file = CHUNKS_DIR / "chunks.jsonl"

    with open(chunk_file, "w", encoding="utf-8") as f:

        for row in all_chunks:
            f.write(json.dumps(row) + "\n")

    metadata_df = pd.DataFrame(metadata)

    metadata_df.to_csv(
        CHUNKS_DIR / "chunk_metadata.csv",
        index=False
    )

    print(f"Created {len(all_chunks)} chunks")
    print("Ingestion complete")


if __name__ == "__main__":
    main()