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

## Research-grade factual verification extensions

This version adds a REFIND-inspired verification layer while preserving the original lightweight RAG pipeline.

### Factual consistency checks

The detector now compares factual values in each claim against the best retrieved evidence. It flags:

- `numeric_mismatch_with_evidence` for conflicting years, dates, or numbers
- `entity_mismatch_with_evidence` for simple same-relation entity conflicts

Example:

```text
Claim: RAG was introduced in 2021.
Evidence: RAG was introduced in 2020.
Output: unsupported, highlighted span [2021]
```

Run:

```bash
python scripts/test_factual_consistency.py
python scripts/test_span_highlighter.py
```

### Optional NLI verification

NLI is implemented as an optional CPU-first layer. It treats retrieved evidence as the premise and the generated claim as the hypothesis.

Config in `configs/settings.yaml`:

```yaml
verification:
  enable_nli: false
  nli_model: "cross-encoder/nli-deberta-v3-small"
  nli_device: "cpu"
  nli_max_evidence_chars: 900
  nli_cache_enabled: true
```

For live demo on low-resource hardware, keep NLI disabled. For research evaluation, enable it and run:

```bash
python scripts/test_nli_verifier.py --enable
```

### REFIND-inspired span highlighting

Exact REFIND CSR is not implemented because Ollama does not expose reliable token-level log probabilities. Instead, the system highlights suspicious factual spans triggered by retrieved-evidence checks, such as mismatched years, unsupported task examples, and unsupported fine-tuning claims.

### Research evaluation

A lightweight research evaluation script is provided:

```bash
python scripts/run_research_evaluation.py --limit 5
```

Outputs:

```text
results/research_eval_results.csv
results/research_eval_summary.json
```

Recommended hardware settings:

- NLI disabled for Streamlit demos
- NLI enabled only for selected evaluation runs
- CPU device for NLI/reranker
- max workers = 1 on 8 GB RAM machines

## Safety Fixes Added After Factual Verification Review

The latest version adds a correction-safety layer so that the corrected answer cannot silently retain or introduce high-risk unsupported claims.

Key safeguards:

- Citation markers such as `[Evidence-1]` are stripped before claim extraction, factual consistency checks, NLI verification, and metric calculation. They are kept only in the final user-facing corrected answer.
- Non-factual assistant phrases such as "I couldn't find..." or "Could you please provide more context?" are skipped during claim scoring.
- Year/date/number claims are checked more strictly. If a claim says a year such as `2021` but the retrieved evidence does not support that year, the detector adds `claim_year_not_supported_by_evidence`.
- Critical flags such as `fine_tuning_not_in_evidence`, `numeric_mismatch_with_evidence`, `claim_year_not_supported_by_evidence`, `entity_mismatch_with_evidence`, and `nli_contradiction` force unsupported labeling.
- The pipeline includes a query-aware answerability gate. If the user asks for a specific fact that is not supported by retrieved evidence, the system returns a safe insufficient-evidence response instead of letting the LLM invent an answer.
- After correction, the corrected answer is verified again. If critical unsupported corrected claims remain, a conservative repair pass removes the unsafe sentence.
- Metrics now include `corrected_critical_unsupported_count`, `correction_safety_passed`, and an `unsafe` correction status.

Recommended safety test:

```bash
python scripts/run_single_query.py --query "RAG was introduced in 2021."
```

Expected behavior: the system should not claim that RAG was introduced in 2021 unless the knowledge base explicitly supports that year.

After adding or changing files in `data/raw/`, rebuild the knowledge base:

```bash
python scripts/ingest_documents.py
python scripts/build_vector_index.py --reset
```

## Latest Audit Improvements

This version keeps the existing architecture but hardens the main pipeline.

- Ingestion now uses PyMuPDF first for PDF extraction, falls back to pypdf, and logs loaded/skipped documents.
- Preprocessing removes common PDF noise such as URLs, DOI/arXiv strings, page numbers, bibliography sections, isolated equations, emails, and affiliation-like lines while preserving definitions, years, and section headings.
- Chunking remains sentence-aware and now stores `document_title`, `section_name`, `chunk_position`, and `importance_score` for better evidence ranking.
- Hybrid retrieval still uses dense ChromaDB + BM25 + RRF, but ranking now also considers dense similarity, normalized sparse score, and section importance.
- NLI remains optional. When enabled, contradiction decisions are protected by a topical-relevance guard so unrelated premise/claim pairs are not blindly treated as contradictions.
- Pytest is restricted to the `tests/` directory to prevent CLI scripts under `scripts/` from being collected as tests.

Recommended validation:

```bash
python -m compileall -q .
pytest -q
python scripts/ingest_documents.py
python scripts/build_vector_index.py --reset
python scripts/run_single_query.py --query "What is retrieval-augmented generation?"
```

## Latest audit fixes

- Hybrid retrieval now applies topical gating so sparse-only noise from unrelated papers is demoted. Queries such as `What is ColBERT?` must retrieve chunks that actually mention ColBERT instead of unrelated abstract/introduction sections.
- BM25 query processing removes common stopwords and avoids generic query expansion such as `abstract` and `introduction`, which previously caused irrelevant PDF sections to rank too high.
- Fused retrieval ranking now combines dense similarity, normalized sparse score, section importance, topical relevance, and RRF score.
- The pipeline has a complete safe execution path for answerable and non-answerable queries, including safe refusal for unsupported specific factual claims.
- Claim extraction now skips heading-like pseudo-claims such as `The RAG process typically consists of two main stages:`.


## Corrected-answer scoring and citation handling

Corrected answers may show evidence citations such as `[Evidence-1]` in the UI. These citations are display-only. Before claim extraction, factual consistency checks, NLI verification, and metric calculation, the system strips citation markers so citation numbers are not treated as factual numbers.

`weak_support` means the retrieved evidence is related to the claim, but the claim is not fully verified. A correction is marked fully improved only when safety checks pass and the corrected answer has sufficiently grounded claims. If most corrected claims remain weakly supported, the status becomes `partially_improved` instead of an overconfident success.

To create a clean deliverable zip without caches, local databases, generated chunks, or environment files, run:

```bash
python scripts/package_project.py --output hallucination-rag-clean.zip
```


## Latest correction-output policy

The Streamlit UI now shows retrieved evidence separately as evidence cards. The user-facing corrected answer is intentionally clean and does not include inline `[Evidence-*]` citation markers. Evidence IDs remain available as metadata and in the retrieved-evidence panel. Detection, factual consistency checks, NLI verification, and metrics strip citations before scoring so citation numbers are never treated as factual numbers.

Correction success is conservative: a result is marked as successful only when safety checks pass, hallucination does not increase, and at least one corrected claim is strongly supported. If the corrected answer is mostly weakly supported, the status is partial rather than a strong success.
