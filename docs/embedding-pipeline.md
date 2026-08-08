# Embedding pipeline

An embedding is a numeric vector representing the semantic meaning of text or code. Later retrieval can compare these vectors to find chunks related to a question, even when the wording differs.

## Provider boundary

`EmbeddingProvider` is a small interface that accepts a batch of strings and returns vectors in the same order. `EmbeddingService` owns batching, retries, logging, and the mapping back to `CodeChunk`; it has no vendor-specific behavior. The supplied `OpenAICompatibleEmbeddingProvider` talks to an OpenAI-compatible `/embeddings` endpoint, but another provider only needs to implement this interface.

The embedding model is intentionally separate from Gemini. Generation models and embedding models solve different tasks, and choosing a strong retrieval embedding model should not constrain the model used for answer generation.

## Model and dimensions

`EMBEDDING_MODEL` selects the provider model (the example uses `text-embedding-3-small`). Vector dimensions are determined by the provider/model. `EMBEDDING_DIMENSIONS` is optional for providers that support requesting a reduced dimension; leave it unset to use the model default. All vectors written later to a collection must use a consistent dimension.

## Batching and resilience

The service sends up to `EMBEDDING_BATCH_SIZE` chunks per provider request, maintaining input order across batches. A failed batch is retried up to `EMBEDDING_MAX_RETRIES` times with exponential delay beginning at `EMBEDDING_INITIAL_BACKOFF_SECONDS`. A permanently failed batch raises a clear provider error and no vector database action is attempted.
