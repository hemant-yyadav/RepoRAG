# Neural reranking

Reranking follows hybrid retrieval and selects the most relevant context before it reaches generation:

```text
query → hybrid candidates (N) → cross-encoder reranker → final context (K)
```

`RerankerProvider` keeps the pipeline vendor-neutral. The supplied local `CrossEncoderRerankerProvider` lazily loads the model configured by `RERANKER_MODEL` and performs pair scoring in `RERANKER_BATCH_SIZE` batches. The default model is a multilingual BGE cross-encoder that works for mixed code and text retrieval; it can be replaced through configuration.

The service fetches at most `RERANKER_CANDIDATE_COUNT` hybrid candidates and emits at most `RERANKER_FINAL_COUNT` results. Equal relevance scores are ordered deterministically by prior hybrid rank and chunk ID. When `RERANKER_FAIL_OPEN=true`, a model failure safely returns the hybrid order rather than blocking development; setting it to false surfaces the provider error.

The evaluation question set now has a [metrics template](evaluation/retrieval-metrics-template.json) for comparing vector-only, hybrid, and hybrid-plus-reranking hit rate and MRR. No reranking evaluation is automated yet.
