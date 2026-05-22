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
- correction prompt safeguards against common RAG hallucinations such as unsupported fine-tuning claims
- hybrid support scoring with semantic similarity plus narrow rule-based caps for specific unsupported details
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
- The hallucination detector currently uses similarity-based support scoring plus a small rule-based safeguard for common unsupported details. The NLI verifier is intentionally left for a later phase.

## Chroma telemetry warning on Windows

If you see messages such as:

```text
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
```

the pipeline can still run. It is caused by a Chroma/PostHog dependency mismatch in some environments. This project pins `posthog<4.0.0` in `requirements.txt` to reduce the warning. If the warning continues after updating this zip, run:

```bash
pip install "posthog<4.0.0"
```

Then rebuild or rerun the query.

## Recommended validation order

```bash
python scripts/check_setup.py
python scripts/ingest_documents.py
python scripts/build_vector_index.py --reset
python scripts/run_single_query.py --query "What is retrieval-augmented generation?"
streamlit run streamlit_app.py --server.fileWatcherType none
```

## Latest detector/UI update

This version adds the following stability improvements:

- Correction status now has four states: `improved`, `partially_improved`, `no_change`, and `worsened`.
- The UI no longer shows a green success message when factual improvement is exactly zero.
- Claim tables are compact by default and show long evidence text inside expandable detail sections.
- The detector returns rule flags and optional NLI fields for each claim.
- NLI verification is optional through `configs/settings.yaml` using `detection.enable_nli`.

By default, NLI is disabled to keep the Windows laptop demo lightweight. To test the real NLI model, run:

```bash
python scripts/test_nli_verifier.py --enable
```

If the model is unavailable or cannot be downloaded, the app falls back to similarity + rule-based verification.

## Hybrid retrieval update

This version can use hybrid retrieval when `configs/settings.yaml` contains:

```yaml
retrieval:
  mode: "hybrid"
```

Hybrid retrieval combines:

- dense semantic retrieval from ChromaDB
- lightweight BM25 sparse retrieval over `data/chunks/chunks.jsonl`
- Reciprocal Rank Fusion (RRF)

RRF score is computed as:

```text
score = 1 / (rrf_k + dense_rank) + 1 / (rrf_k + sparse_rank)
```

Use this diagnostic script after building the vector index:

```bash
python scripts/test_hybrid_retrieval.py --query "What is retrieval-augmented generation?"
```

The evidence table now includes optional retrieval fields such as `rrf_score`, `dense_similarity`, `sparse_rank`, and `sparse_score`.

## Bounded batch evaluation

The Streamlit and single-query pipeline remain sequential for stability. Batch evaluation supports conservative bounded processing:

```bash
python scripts/run_batch_evaluation.py --limit 5 --max-workers 1 --max-llm-workers 1 --max-nli-workers 1
```

Keep these values low on the Windows laptop with 8 GB RAM and GTX 1050 Ti 4 GB VRAM.

## Retrieval upgrade: hybrid dense + sparse search

The retriever now supports two modes through `configs/settings.yaml`:

```yaml
retrieval:
  mode: hybrid   # use "dense" to fall back to Chroma-only retrieval
  dense_top_k: 8
  sparse_top_k: 8
  rrf_k: 60
```

Hybrid mode keeps the existing Chroma dense retriever and adds a lightweight dependency-free BM25 retriever over `data/chunks/chunks.jsonl`. Results are fused with Reciprocal Rank Fusion (RRF):

```text
RRF score = 1 / (k + dense_rank) + 1 / (k + sparse_rank)
```

This improves evidence retrieval for exact acronyms, names, dates, and technical keywords while preserving semantic retrieval quality.

### Test hybrid retrieval

```bash
python scripts/test_hybrid_retrieval.py --query "What is retrieval-augmented generation?"
```

### Low-resource batch evaluation

Batch evaluation remains conservative by default to avoid overloading local Ollama or optional NLI models on 8 GB RAM machines:

```bash
python scripts/run_batch_evaluation.py --limit 5 --max-workers 1 --max-llm-workers 1 --max-nli-workers 1
```
