# Phase 0 architecture

This small monorepo separates a FastAPI backend from a Next.js presentation layer.

- `backend/app/api`: HTTP routing.
- `backend/app/core`: configuration, logging, and shared application concerns.
- `backend/app/models`: typed API models.
- `backend/app/services`: future integration boundaries; no RAG service exists yet.
- `frontend`: presentation layer only; it has no RAG client implementation.

External-service settings are environment-driven. Qdrant and Gemini settings are placeholders for later work.
