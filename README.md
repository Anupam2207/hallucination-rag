# hallucination-rag

A final-year engineering research project for **claim-level hallucination detection and correction in retrieval-augmented generation (RAG) answers**.

The system retrieves evidence, checks whether the query is answerable from that evidence, splits generated answers into factual claims, scores each claim as `supported`, `weak_support`, or `unsupported`, optionally verifies claims with NLI, corrects unsupported answers using evidence only, and verifies the corrected answer again.

## What this project does

`hallucination-rag` is designed to be both demo-friendly and paper-ready:

- It runs in low-resource demo mode on a laptop by default.
- It supports a heavier research mode with optional NLI verification.
- It reports claim-level detection metrics, retrieval metrics, correction metrics, ablations, confusion matrices, and qualitative examples.
- It avoids relying on hardcoded single-demo rules by mapping rule findings into reusable research categories such as `numeric_mismatch`, `temporal_mismatch`, `entity_mismatch`, `definition_mismatch`, `unsupported_method_claim`, and `unsupported_task_claim`.

## Architecture

```text
User query
  |
  v
Ingestion and indexing
  - load documents from data/raw
  - clean and chunk documents
  - build dense vector index and BM25 sparse index
  |
  v
Hybrid retrieval
  - dense retrieval
  - BM25 retrieval
  - reciprocal-rank fusion
  - evidence intent and credibility scoring
  |
  v
Answerability gate
  - rejects empty, example-only, reference-only, or semantically related but fact-missing evidence
  - stricter checks for who/when/where/founded/introduced/invented/date/year questions
  |
  v
Answer generation
  - Ollama LLM when available
  - evidence-only fallback in demo/offline mode
  |
  v
Claim extraction
  - fast sentence extractor
  - optional atomic splitting for bullets, numbered lists, semicolons, and conjunction-heavy facts
  |
  v
Claim verification
  - semantic/lexical support scoring
  - factual consistency rules
  - optional NLI entailment/contradiction check
  |
  v
Correction
  - rewrite unsupported answers using evidence only
  - preserve supported answers
  - repair unsafe or malformed corrections
  |
  v
Post-correction verification and metrics
```

## Repository layout

```text
configs/                 Runtime, model, and prompt configuration
configs/settings.yaml    Base config with demo-safe defaults
configs/settings_demo.yaml
configs/settings_research.yaml
data/raw/                Source documents kept in the repository
data/evaluation/         Manual-answer evaluation sets
scripts/                 Ingestion, indexing, evaluation, ablation, and utility scripts
src/                     Core package
src/detection/           Claim extraction, support scoring, NLI, factual rules
src/retrieval/           Dense, sparse, hybrid retrieval, vector store, evidence scoring
src/generation/          Base answer generation and correction
src/evaluation/          Detection, correction, retrieval, and span metrics
tests/                   Unit and regression tests
docs/                    Methodology and experiment notes
paper_artifacts/         Generated tables, figures, metrics, and qualitative examples
```

`data/processed`, `data/chunks`, `db/chroma`, `outputs`, and `results` are rebuildable artifacts and are ignored by Git.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional; do not commit .env
python scripts/check_setup.py
```

The default demo profile does not require NLI and can run without downloading sentence-transformer weights. It uses a deterministic hashing embedder unless you switch to research mode.

## Demo mode

Demo mode is the default and is configured in `configs/settings_demo.yaml`.

```bash
export HALLUCINATION_RAG_PROFILE=demo
python scripts/check_setup.py
```

Demo mode choices:

- NLI disabled by default.
- Hashing-vectorizer embedding backend for offline/low-memory runs.
- Hybrid retrieval remains enabled, but dense embeddings are lightweight.
- Ollama is optional. If the Ollama Python package or local service is unavailable, the pipeline returns an evidence-only fallback answer and records a warning.

Important demo thresholds are configurable in `configs/settings.yaml` and `configs/settings_demo.yaml`:

```yaml
retrieval:
  top_k: 4
  dense_top_k: 8
  sparse_top_k: 8
  rrf_k: 60
  answerability_threshold: 0.45

detection:
  support_threshold: 0.70
  weak_support_threshold: 0.45
  answerability_threshold: 0.45
  nli_entailment_threshold: 0.60
  nli_contradiction_threshold: 0.60
```

## Research mode

Research mode is configured in `configs/settings_research.yaml`.

```bash
export HALLUCINATION_RAG_PROFILE=research
python scripts/check_setup.py
```

Research mode choices:

- NLI enabled by default.
- Larger retrieval candidate pools.
- Research PDFs included during ingestion.
- Embedding backend set to `auto`, so sentence-transformers are used when available and the hash fallback is still allowed.

NLI is intentionally optional because cross-encoder NLI models can be slow and memory-heavy on 8 GB systems. The code falls back safely if the NLI model is unavailable.

## Ingest documents

Place source files under one of these directories:

```text
data/raw/txt
data/raw/md
data/raw/json
data/raw/pdf
```

Then run:

```bash
python scripts/ingest_documents.py
```

This creates `data/processed/cleaned_documents.jsonl` and `data/chunks/chunks.jsonl`.

## Build the index

```bash
python scripts/build_vector_index.py --reset
```

The vector store uses ChromaDB when installed. If `chromadb` is unavailable, it automatically uses a small JSONL-backed cosine-search store under `db/chroma`, so the project remains runnable on a laptop.

## Run one query

```bash
python scripts/run_single_query.py --query "What is retrieval-augmented generation?"
```

The JSON output includes retrieved evidence, answerability status, raw answer, corrected answer, claim-level labels, rule flags, rule categories, and before/after metrics.

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Run ingestion and indexing first. The app displays raw and corrected answers, evidence passages, claim labels, support scores, hallucinated spans, warnings, and correction metrics.

## Run tests

```bash
pytest -q
```

The test suite covers ingestion utilities, retrieval, evidence intent, answerability, claim extraction, support scoring, NLI fallback, correction safety, evaluation metrics, and regression cases.

## Run final evaluation

The final benchmark uses fixed manual answers in `data/evaluation/final_eval_set.jsonl`, so results are reproducible without calling an LLM.

```bash
python scripts/run_final_evaluation.py --retrieval hybrid --nli off --correction on
```

This writes:

```text
paper_artifacts/tables/final_detection_results.csv
paper_artifacts/metrics/final_detection_summary.json
paper_artifacts/tables/final_correction_results.csv
paper_artifacts/metrics/final_correction_summary.json
paper_artifacts/tables/retrieval_results.csv
paper_artifacts/metrics/retrieval_summary.json
paper_artifacts/figures/confusion_matrix.png
paper_artifacts/qualitative_examples/qualitative_success_cases.md
paper_artifacts/qualitative_examples/qualitative_failure_cases.md
```

You can also run the components separately:

```bash
python scripts/evaluate_retrieval.py --retrieval hybrid
python scripts/evaluate_detection.py --nli off --rules on
python scripts/evaluate_correction.py --nli off
```

## Reproduce paper tables and ablations

```bash
python scripts/run_ablation_study.py
```

The ablation script compares:

- BM25-only retrieval
- dense-only retrieval
- hybrid retrieval
- similarity-only detection
- similarity + factual rules
- similarity + NLI
- similarity + factual rules + NLI
- full system with correction

It writes:

```text
paper_artifacts/tables/ablation_results.csv
paper_artifacts/tables/ablation_summary.json
```

Use real NLI in research mode with:

```bash
export HALLUCINATION_RAG_PROFILE=research
python scripts/run_ablation_study.py --enable-real-nli
```

## Metrics reported

Retrieval:

- `recall@k`
- MRR
- evidence keyword hit rate
- answerability retrieval success

Detection:

- claim-level accuracy
- precision, recall, F1
- macro F1
- confusion matrix
- unsupported recall
- false positive rate
- false negative rate

Correction:

- correction success rate
- hallucination reduction
- support improvement
- unsafe correction rate
- corrected unsupported claim count
- corrected critical unsupported claim count
- no-change rate
- malformed correction rate

## Known limitations

- The default demo embedder is deterministic and lightweight, but it is not a replacement for sentence-transformer embeddings in a research-quality retrieval study.
- Rule-based factual checks catch common numeric, temporal, entity, definition, task, method, training, application, and performance hallucinations, but they are not a complete logical verifier.
- NLI verification improves contradiction checks when available, but it is disabled by default to fit low-memory systems.
- The correction stage is evidence-constrained and conservative. It may refuse instead of producing a fluent answer when evidence is incomplete.
- PDF extraction quality depends on the source PDF text layer.
- The manual 100-example dataset is suitable for a final-year project paper, but larger external benchmarks are still needed for stronger claims.

## Troubleshooting

### Ollama

If `run_single_query.py` reports an Ollama warning, the project still runs using evidence-only fallback generation. To enable local LLM generation:

```bash
pip install ollama
ollama serve
ollama pull llama3.2:3b
```

Then rerun the query script.

### ChromaDB

If ChromaDB is not installed, the project automatically uses the JSONL vector-store fallback. To use ChromaDB explicitly:

```bash
pip install chromadb
python scripts/build_vector_index.py --reset
```

### NLI

Demo mode keeps NLI off. For research mode:

```bash
export HALLUCINATION_RAG_PROFILE=research
python scripts/evaluate_detection.py --nli on
```

If model loading fails, the verifier returns `label='unavailable'` instead of crashing. Use `--nli off` for low-memory runs.

### Low-memory systems

Recommended settings for 8 GB RAM:

- Keep `HALLUCINATION_RAG_PROFILE=demo`.
- Keep `runtime.embedding_backend: hash`.
- Keep `verification.enable_nli: false`.
- Use `--nli off` for evaluation scripts.
- Keep `retrieval.top_k` between 3 and 5.
