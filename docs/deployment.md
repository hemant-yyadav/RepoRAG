# Docker and deployment

## Local container stack

Copy `.env.example` to `.env`, populate the provider keys required for the features you intend to use, then run:

```powershell
docker compose up --build
```

The local services are available at:

- Frontend: `http://localhost:3000`
- Backend OpenAPI: `http://localhost:8000/docs`
- Backend process health: `http://localhost:8000/health`
- Backend Qdrant connectivity: `http://localhost:8000/health/qdrant`
- Qdrant dashboard: `http://localhost:6333/dashboard`

Compose injects `QDRANT_URL=http://qdrant:6333` into the backend, while browser code uses `NEXT_PUBLIC_API_URL=http://localhost:8000`. The backend and Qdrant include Docker health checks; inspect `docker compose ps` and service logs when startup fails.

## Environment variables

`.env.example` is the complete local configuration reference. Required when the related feature is used:

- `GEMINI_API_KEY`, `GEMINI_MODEL`: grounded answers and conversational query rewriting.
- `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL`: repository embeddings.
- `QDRANT_URL`, optionally `QDRANT_API_KEY`, and `QDRANT_COLLECTION_NAME`: vector persistence.

All remaining values set operational limits and defaults: CORS, GitHub/repository limits, chunking, batching/retries, retrieval/reranking, conversation history, and file inspection. Do not commit a populated `.env` file.

## Practical deployment topology

```text
Next.js frontend (Vercel or equivalent)
        ↓ HTTPS API requests
FastAPI backend (Render, Railway, container platform, or equivalent)
        ↓                 ↘
Qdrant Cloud/self-hosted   Gemini API and embedding provider
```

Deploy the frontend with `NEXT_PUBLIC_API_URL` set to the public backend URL. Deploy the backend from `backend/Dockerfile`, set `CORS_ORIGINS` to the frontend origin, and configure secrets through the platform’s secret manager. Use managed Qdrant or a persistent self-hosted volume; do not use ephemeral vector storage for a production index.

The application does not assume a particular deployment provider. Background indexing is currently executed through FastAPI background tasks; for multi-instance production workloads, replace that execution mechanism with a durable worker queue while preserving the lifecycle service contract.

## Build verification

The images are intentionally not claimed as verified until `docker build` or `docker compose up --build` completes in an environment with Docker available. Frontend production builds can be verified independently with `npm run build`.
