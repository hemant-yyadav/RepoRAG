# Production-style FastAPI API

FastAPI exposes OpenAPI documentation at `/docs` and `/openapi.json`. Route handlers delegate to services; they do not clone, embed, search, or generate directly.

## Main endpoints

- `POST /repositories/index` accepts a public GitHub URL and returns `202 Accepted` with a deterministic repository ID and `indexing` status. The indexing job runs through FastAPI background tasks, so a later worker queue can replace this execution mechanism without changing the API contract.
- `GET /repositories`, `GET /repositories/{id}`, and `GET /repositories/{id}/status` expose lifecycle metadata.
- `DELETE /repositories/{id}` removes indexed data and its lifecycle record.
- `GET /repositories/{id}/files` lists indexed paths; the existing file/symbol inspection endpoints provide their contents.
- `POST /repositories/search` returns reranked chunks without generation.
- `POST /chat` returns grounded answers, verified sources, conversation ID, and retrieval diagnostics.
- `GET /health` reports API availability.

Requests use Pydantic validation for URLs, identifiers, messages, queries, and result limits. Expected client errors return consistent 4xx responses, while detailed indexing failures are logged server-side and exposed only as safe status messages. Repository lifecycle records are process-local in this phase; the service boundary is intentionally ready for persistence or a background queue later.
