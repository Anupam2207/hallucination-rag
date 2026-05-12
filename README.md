# Hallucination Detection and Correction in LLMs using RAG

A lightweight M.Tech final-year project that demonstrates how to detect and correct unsupported claims in LLM answers using retrieval-augmented generation (RAG).

## What this project does

The system performs the following pipeline:

1. Accept a user query.
2. Generate a raw LLM answer using a local Ollama model.
3. Retrieve supporting passages from a local knowledge base using dense embeddings and ChromaDB.
4. Split the answer into claim-like sentences.
5. Score each claim against retrieved evidence with cosine similarity.
6. Mark claims as supported, partially supported, or unsupported.
7. Rewrite the answer using the retrieved evidence and the original answer.
8. Show before/after support metrics.

## Project status

This version focuses on making the knowledge-base preparation, vector indexing, and single-query pipeline reproducible and runnable.

Implemented in this update:
- raw document ingestion for txt, md, json, and pdf
- lightweight preprocessing
- sentence-aware chunking
- reproducible ChromaDB index building
- single-query pipeline runner
- working Streamlit entry point
- correction that uses the original raw answer
- basic evaluation metrics
- sample corpus and demo queries

Still reserved for later phases:
- NLI verifier
- stronger batch evaluation and paper-grade experiments
- richer UI polish

## Recommended hardware/runtime

- Windows laptop
- Python 3.10+
- Ollama installed and running locally
- Small local model such as `llama3.2:3b`

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure Ollama is installed and the model is available:

```bash
ollama pull llama3.2:3b
```

## Sanity check

```bash
python scripts/check_setup.py
python scripts/test_ollama.py
```

## Build the knowledge base

### 1) Ingest raw documents

```bash
python scripts/ingest_documents.py
```

This creates:
- `data/processed/cleaned_documents.jsonl`
- `data/processed/corpus_metadata.csv`
- `data/chunks/chunks.jsonl`
- `data/chunks/chunk_metadata.csv`

### 2) Build the vector index

```bash
python scripts/build_vector_index.py --reset
```

## Run a single query in the terminal

```bash
python scripts/run_single_query.py --query "What is retrieval-augmented generation?"
```

## Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Sample data

The repository ships with a small curated demo corpus under `data/raw/` and a small benchmark under `data/eval/benchmark_queries.jsonl`.

## Important notes

- If Ollama is not running, answer generation will fail with a readable error.
- If the Chroma index has not been built yet, retrieval will return no evidence and the UI or CLI will show a warning.
- The hallucination detector currently uses similarity-based support scoring. The NLI verifier is intentionally left for a later phase.
