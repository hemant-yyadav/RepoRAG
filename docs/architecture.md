# Architecture

```mermaid
flowchart TB
  subgraph Client
    WEB["Next.js frontend"]
  end
  subgraph API["FastAPI backend"]
    LIFE["Repository lifecycle"]
    ING["GitHub ingestion"]
    CHUNK["Code-aware chunking"]
    EMB["Embedding service"]
    RET["Hybrid retrieval + reranking"]
    GEN["Grounded Gemini generation"]
  end
  subgraph Storage
    QD[("Qdrant vectors + metadata")]
    BM["In-memory repository BM25"]
  end
  WEB --> LIFE
  LIFE --> ING --> CHUNK --> EMB --> QD
  CHUNK --> BM
  WEB --> RET
  RET --> QD
  RET --> BM
  RET --> GEN --> WEB
```

The monorepo keeps HTTP concerns in `backend/app/api`, configuration/logging/resilience in `backend/app/core`, typed models in `backend/app/models`, and domain integrations in `backend/app/services`. The frontend uses one typed API client rather than scattering network calls through components.

Repository content is treated as untrusted data throughout the flow; it is read and indexed but never executed.
