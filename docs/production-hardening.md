# Production hardening

The hardening work keeps the existing service boundaries while adding bounded operational safeguards.

## Ingestion and indexing

- Each clone uses a `TemporaryDirectory`, so checkouts are cleaned even when ingestion fails.
- Individual file size, repository file-count, and repository total-size limits are configurable.
- Dependency, build, generated, minified, sourcemap, binary, environment, and common vendor paths are excluded before content processing.
- GitHub clone failures use finite exponential backoff. A failed clone removes only its temporary checkout before retrying.
- Repository IDs are deterministic. Re-submitting an unchanged repository computes a content signature and skips chunking, embeddings, and Qdrant replacement.
- Lifecycle status now includes safe stage/progress fields (`queued`, `ingesting`, `indexing`, `completed`, `failed`). Failures expose a generic status message while details remain in server logs.

## External services and timing

- Embeddings already batch and retry transient provider failures.
- Gemini retries only transport, rate-limit (429), and server-side (5xx) failures with configured finite backoff; malformed/empty responses fail safely.
- Qdrant calls use finite retry/backoff around external operations.
- Structured timing logs cover ingestion, chunking, embedding, hybrid retrieval, reranking, and generation. They record operation names/counts/durations—not query text, source content, API keys, or secrets.

The local cross-encoder model cache and repository-scoped BM25 index are intentionally retained as useful expensive-operation caches. File and answer outputs are not broadly cached, avoiding stale repository data and privacy surprises.

## Security limits

Repository content is treated as untrusted text only; no repository code is executed. Public GitHub URL validation prevents arbitrary clone targets. File inspection accepts only normalized repository-relative POSIX paths and rejects absolute paths, backslashes, and traversal segments. API errors remain safe for clients, while detailed exceptions are logged server-side.

Gemini streaming is not enabled in this phase: the existing citation validation requires complete output before returning verified sources. This keeps source integrity deterministic; a future streaming endpoint must buffer/validate citations before finalizing sources.
