# Experiments

## Datasets

The main reproducible benchmark is `data/evaluation/final_eval_set.jsonl`. It contains 100 manually specified records. Each record includes:

- query;
- fixed manual answer;
- evidence snippets;
- expected evidence keywords;
- answerability label;
- gold claim labels.

The categories are:

1. supported factual claims;
2. unsupported fabricated claims;
3. wrong entity claims;
4. wrong year/date claims;
5. unsupported numeric claims;
6. unsupported method claims;
7. unsupported benchmark claims;
8. unsupported application/task claims;
9. definition mismatch claims;
10. insufficient-evidence queries.

Manual-answer evaluation avoids LLM sampling variance and makes final tables reproducible.

## Baselines

Retrieval baselines:

- BM25-only retrieval;
- dense-only retrieval;
- hybrid retrieval.

Detection baselines:

- similarity-only detection;
- similarity + factual rules;
- similarity + NLI;
- similarity + factual rules + NLI.

Full-system variant:

- hybrid retrieval + answerability + similarity + factual rules + optional NLI + correction + post-correction verification.

## Ablations

Run:

```bash
python scripts/run_ablation_study.py
```

Generated outputs:

```text
paper_artifacts/tables/ablation_results.csv
paper_artifacts/tables/ablation_summary.json
```

Use real NLI with:

```bash
export HALLUCINATION_RAG_PROFILE=research
python scripts/run_ablation_study.py --enable-real-nli
```

The default ablation command keeps NLI disabled or unavailable-safe so it can run on low-resource laptops.

## Evaluation protocol

1. Run document ingestion.
2. Build the retrieval index.
3. Run final evaluation on the 100-example manual-answer set.
4. Run the ablation study.
5. Inspect quantitative tables and qualitative examples.

Recommended commands:

```bash
pytest -q
python scripts/check_setup.py
python scripts/ingest_documents.py
python scripts/build_vector_index.py --reset
python scripts/run_final_evaluation.py --retrieval hybrid --nli off --correction on
python scripts/run_ablation_study.py
```

## Result tables

Final evaluation writes:

```text
paper_artifacts/tables/final_detection_results.csv
paper_artifacts/metrics/final_detection_summary.json
paper_artifacts/tables/final_correction_results.csv
paper_artifacts/metrics/final_correction_summary.json
paper_artifacts/tables/retrieval_results.csv
paper_artifacts/metrics/retrieval_summary.json
paper_artifacts/figures/confusion_matrix.png
```

Detection summary contains:

- accuracy;
- precision;
- recall;
- F1;
- macro F1;
- unsupported recall;
- false positive rate;
- false negative rate;
- confusion matrix.

Retrieval summary contains:

- recall@k;
- MRR;
- evidence keyword hit rate;
- answerability retrieval success.

Correction summary contains:

- correction success rate;
- hallucination reduction;
- support improvement;
- unsafe correction rate;
- corrected unsupported claim count;
- corrected critical unsupported claim count;
- no-change rate;
- malformed correction rate.

## Qualitative examples

Final evaluation writes:

```text
paper_artifacts/qualitative_examples/qualitative_success_cases.md
paper_artifacts/qualitative_examples/qualitative_failure_cases.md
```

Use success cases to show unsupported claims being removed or refused. Use failure cases to discuss conservative refusals, weak evidence, remaining unsupported claims, or cases where evidence retrieval was insufficient.

## Notes for reporting

When writing the paper, report demo-mode and research-mode settings separately. Demo-mode results are useful to show laptop feasibility. Research-mode results with NLI enabled are more appropriate for the final comparison if the model can be loaded reliably.
