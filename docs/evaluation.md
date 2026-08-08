# RAG evaluation

This project evaluates retrieval and answer quality separately. The evaluation fixture repository lives under `backend/tests/fixtures/evaluation_repository`, and the curated dataset has 30 realistic questions in `docs/evaluation/retrieval-evaluation-dataset.json`. Each case identifies expected file paths and, where practical, expected symbols.

## Repeatable retrieval run

With Qdrant running and embedding credentials configured, run from `backend/`:

```powershell
python -m app.evaluation_runner `
  --dataset ..\docs\evaluation\retrieval-evaluation-dataset.json `
  --repository-id evaluation-fixture `
  --repository-path tests\fixtures\evaluation_repository `
  --output ..\data\evaluation-results
```

The runner first reindexes the fixture repository, then evaluates Vector, Hybrid, and Hybrid + reranker with the same question set. It writes `retrieval-results.json` and `retrieval-report.md`. The reusable generation evaluator likewise writes `generation-results.json` and `generation-report.md` when invoked by an answer-evaluation workflow. These files contain only executed measurements; they are ignored by Git, so no fabricated baseline is checked in.

## Retrieval metrics

- **Recall@1/3/5/10**: the fraction of questions with at least one expected file/symbol in the first K results.
- **MRR**: mean reciprocal rank of each question’s first relevant result.
- **Precision@1/3/5/10**: fraction of returned positions that are relevant, using K as the denominator even when fewer results are returned.

The report compares all three methods without changing their retrieval settings during a run. The metrics template in `docs/evaluation/retrieval-metrics-template.json` records the results of repeated experiments.

## Generation evaluation

Citation precision and recall are deterministic: returned citations are compared with the case’s expected file paths and symbols. `evaluate_generation_case` also accepts an optional `AnswerJudge` for 1–5 relevance and groundedness scores.

LLM-as-judge scores are useful for repeated qualitative review but are not objective truth: results can vary by judge model, prompt, temperature, and bias toward fluent answers. Keep judge prompts/versioning fixed, retain the raw answer and sources for audit, and report deterministic citation metrics beside any judge scores.

## Actual results

No results are recorded in the repository until the runner completes against configured embedding, Qdrant, and reranker services. This prevents placeholder or invented measurements from being mistaken for engineering evidence.
