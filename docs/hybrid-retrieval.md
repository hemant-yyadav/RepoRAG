# Hybrid retrieval

Hybrid retrieval combines semantic vector matches from Qdrant with lexical BM25 matches from a repository-scoped in-memory index. This helps source code queries that contain exact identifiers such as `authenticateUser`, `getUserById`, and `JWT_SECRET`.

## Lexical indexing

`BM25Index` indexes each chunk's file path, symbol name, and code content. Its tokenizer preserves whole identifiers while also adding snake_case and camelCase components, so `authenticateUser` can match both the exact identifier and terms such as `authenticate` and `user`.

The indexing service updates this process-local index after successful Qdrant replacement and removes the repository's lexical entries on deletion. As an in-memory development index, it is rebuilt when repositories are re-indexed after an application restart.

## Fusion

Semantic and lexical candidate lists are merged with weighted Reciprocal Rank Fusion:

```text
fused_score = vector_weight / (rrf_k + vector_rank)
            + bm25_weight / (rrf_k + bm25_rank)
```

The weights, `rrf_k`, candidate pool size, and default result count are environment-configurable. Each result exposes final rank, fused score, vector rank, BM25 rank, and chunk metadata for evaluation. Every vector and lexical lookup is scoped to the requested repository ID.

No reranking, hybrid generation changes, frontend work, or conversation memory is part of this phase.
