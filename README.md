# Codebase RAG Assistant

The project currently provides application scaffolding, public GitHub repository ingestion, code-aware chunking, a provider-independent embedding pipeline, Qdrant vector storage, hybrid repository-isolated retrieval, neural reranking, and grounded Gemini answer generation with validated source tracking. Generation receives only the reranked final context. It does **not** implement conversation memory or frontend chat.

## Repository layout

```text
backend/       FastAPI application and tests
frontend/      Minimal Next.js application
docs/          Architecture and project notes
data/          Reserved for local, untracked application data
```

## Prerequisites

- Python 3.11 or newer
- Node.js 20 or newer

## Configuration

Copy the example environment file before starting services:

```powershell
Copy-Item .env.example .env
```

The backend reads this root `.env` file when run from `backend/`. The Gemini and Qdrant variables are reserved for later phases and are not used yet.

## Run the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000/health` to confirm the API is running.

### Ingest a repository

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/repositories/ingest `
  -ContentType "application/json" `
  -Body '{"repository_url":"https://github.com/user/repository"}'
```

Only public `https://github.com/owner/repository` URLs are accepted. The response contains file counts, total size, and detected languages; source-file contents remain internal. Files over `MAX_FILE_SIZE_BYTES` (1 MiB by default), binary files, environment files, dependencies, build output, and common minified files are skipped.

### Backend tests

```powershell
cd backend
pytest
```

## Run the frontend

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The page displays a Phase 0 status panel and a backend-health placeholder.

## Current scope

The next phases may add repository ingestion, document chunking, embeddings, Qdrant storage, Gemini generation, and a chat experience. None of those features are part of this phase.
