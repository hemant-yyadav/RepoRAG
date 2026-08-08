# Basic semantic retrieval

The development retrieval endpoint verifies semantic search without generating an answer. It performs this sequence:

```text
question → embedding provider → Qdrant cosine similarity search → ranked chunks
```

`POST /retrieval/search` accepts a repository ID, query, optional `top_k`, and optional score threshold. It returns rank, similarity score, file path, source line range, symbol, chunk type, and chunk text for inspection.

Every Qdrant query includes a mandatory `repository_id` payload filter. Results therefore cannot cross repository boundaries, even if unrelated repositories share one collection.

`RETRIEVAL_TOP_K` controls the default result limit. `RETRIEVAL_SCORE_THRESHOLD` is optional; when set, Qdrant excludes lower-scoring matches. This phase intentionally has no answer generation, hybrid search, reranking, or conversation history.
