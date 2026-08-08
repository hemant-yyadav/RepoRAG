from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.models.chunk import CodeChunk


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A ranked stored chunk returned by semantic similarity search."""

    chunk: CodeChunk
    score: float
    metadata: dict[str, str | int | bool]


class RetrievalRequest(BaseModel):
    repository_id: str = Field(min_length=1, max_length=256)
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)


class RetrievalResponseItem(BaseModel):
    rank: int
    score: float
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None
    chunk_type: str
    content: str
    fused_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    relevance_score: float | None = None
    hybrid_score: float | None = None


class RetrievalResponse(BaseModel):
    repository_id: str
    query: str
    results: list[RetrievalResponseItem]
