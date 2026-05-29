# Methodology

## Problem definition

Given a user query, a retrieved evidence set, and an LLM answer, the system performs claim-level hallucination detection and correction. A claim is treated as hallucinated when the available evidence does not support the factual content of that claim. The target labels are:

- `supported`: the evidence directly supports the claim.
- `weak_support`: the evidence is related but incomplete, indirect, or below the strong support threshold.
- `unsupported`: the evidence does not support the claim or contradicts a critical fact.

The system also decides whether the query itself is answerable from the retrieved evidence. If evidence is absent, example-only, reference-only, or missing the requested specific fact, the system refuses rather than guessing.

## System architecture

```text
Documents -> preprocessing -> chunking -> dense index + BM25 index
       -> hybrid retrieval -> evidence intent and credibility scoring
       -> answerability gate -> answer generation or refusal
       -> claim extraction -> support scoring + factual rules + optional NLI
       -> correction -> post-correction verification -> metrics
```

The design separates retrieval, answerability, detection, correction, and evaluation so each stage can be tested and ablated independently.

## Retrieval method

The retriever supports three modes:

1. **BM25-only retrieval** over `data/chunks/chunks.jsonl`.
2. **Dense-only retrieval** over the vector store.
3. **Hybrid retrieval** using dense retrieval, BM25 retrieval, and reciprocal-rank fusion.

Hybrid retrieval uses configurable candidate counts:

- `retrieval.top_k`
- `retrieval.dense_top_k`
- `retrieval.sparse_top_k`
- `retrieval.rrf_k`

Dense retrieval uses ChromaDB when available. In demo mode, a deterministic hashing embedder and JSONL vector-store fallback keep the project runnable offline.

## Evidence credibility scoring

Retrieved chunks are annotated before answerability and support scoring. The evidence annotation layer classifies each chunk as one of:

- `FACTUAL`
- `EXPLANATORY`
- `EXAMPLE`
- `REFERENCE`
- `DISCUSSION`
- `NOISE`

The credibility score combines semantic score, metadata quality, section priority, source authority, factual assertion patterns, and penalties for examples, tutorials, hypotheticals, references, and noise. This prevents semantically similar examples such as `For example, "RAG was introduced in 2021"...` from being counted as evidence for the factual claim.

## Answerability

Answerability is checked before generation and correction. A query is rejected when:

- no evidence is retrieved;
- retrieved chunks are not credible support evidence;
- the question asks for a specific fact and the fact is missing;
- a who/when/where/founded/introduced/invented question retrieves only a related definition;
- a date/year/numeric question retrieves evidence with the wrong or missing value;
- the retrieved text is from example, tutorial, hypothetical, or reference sections.

This is stricter than semantic similarity alone. For example, a RAG definition can support `What is RAG?`, but it cannot support `Who introduced RAG?` unless an authorship statement is present.

## Claim extraction

The claim extractor is intentionally lightweight and CPU-friendly. It keeps the original sentence-based extraction path and adds optional atomic splitting. It handles:

- normal factual sentences;
- bullet lists;
- numbered lists;
- semicolon-separated statements;
- conjunction-heavy factual sentences such as `RAG retrieves evidence and generates grounded answers`;
- answers containing evidence citations such as `[Evidence-1]`;
- malformed list fragments.

Atomic splitting is controlled by `detection.atomic_claim_splitting_enabled`.

## Support scoring

Each claim is scored against retrieved evidence using a hybrid of:

- exact support matching;
- lexical overlap;
- embedding cosine similarity;
- combined top evidence context;
- factual consistency checks.

The configurable thresholds are:

- `detection.support_threshold`
- `detection.weak_support_threshold`
- backward-compatible aliases `similarity_support_threshold` and `similarity_warning_threshold`

A claim above the support threshold is labeled `supported`. A claim above the weak threshold but below the support threshold is labeled `weak_support`. Otherwise it is `unsupported`.

## Rule categories

Specific legacy rule flags are preserved for compatibility, but results also expose generic paper-friendly categories:

- `numeric_mismatch`
- `temporal_mismatch`
- `entity_mismatch`
- `definition_mismatch`
- `unsupported_method_claim`
- `unsupported_task_claim`
- `unsupported_application_claim`
- `unsupported_performance_claim`
- `unsupported_training_claim`

Examples:

- `claim_year_not_supported_by_evidence` maps to `temporal_mismatch`.
- `fine_tuning_not_in_evidence` maps to `unsupported_method_claim`.
- `unsupported_task_example_not_in_evidence` maps to `unsupported_task_claim`.

These categories are easier to report in tables and qualitative analysis.

## NLI verification

NLI is optional. When enabled, evidence is used as the premise and each claim is used as the hypothesis. The verifier supports sentence-transformers CrossEncoder first, then a Hugging Face sequence-classification fallback. If neither backend is available, it returns `label='unavailable'` instead of crashing.

Configurable NLI thresholds:

- `detection.nli_entailment_threshold`
- `detection.nli_contradiction_threshold`
- `verification.nli_contradiction_relevance_threshold`

NLI is disabled by default in demo mode because it is heavier than the rest of the pipeline.

## Correction and post-correction verification

The correction step rewrites unsafe answers using retrieved evidence only. It is conservative:

- If raw claims are already supported and the correction model is unavailable, the raw answer is preserved.
- If unsupported critical claims remain, the repair step removes them or returns an insufficient-evidence response.
- Malformed corrections are replaced with `Insufficient evidence available in the knowledge base.`
- The corrected answer is verified again using the same detector.

The before/after detector outputs are then passed to correction metrics.

## Metrics

Retrieval metrics:

- recall@k
- MRR
- evidence keyword hit rate
- answerability retrieval success

Detection metrics:

- claim-level accuracy
- precision, recall, F1
- macro F1
- confusion matrix
- unsupported recall
- false positive rate
- false negative rate

Correction metrics:

- correction success rate
- hallucination reduction
- support improvement
- unsafe correction rate
- corrected unsupported claim count
- corrected critical unsupported claim count
- no-change rate
- malformed correction rate

## Limitations

The system is designed for local final-year research, not as a complete fact-checking oracle. Main limitations are:

- lightweight demo embeddings are weaker than sentence-transformer embeddings;
- rule categories are broad and cannot cover all factual errors;
- NLI can be unavailable or slow on low-memory systems;
- answerability checks are conservative and may refuse borderline answerable cases;
- correction quality depends on retrieved evidence quality;
- PDF extraction can introduce noisy chunks;
- the 100-example dataset is useful for reproducible project evaluation but should be expanded for stronger external validity.
