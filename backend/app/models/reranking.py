from dataclasses import dataclass

from app.models.chunk import CodeChunk


@dataclass(frozen=True, slots=True)
class RerankedRetrievalResult:
    """A final context candidate after neural relevance scoring."""

    rank: int
    chunk: CodeChunk
    relevance_score: float
    hybrid_score: float
    vector_rank: int | None
    bm25_rank: int | None
    metadata: dict[str, str | int | bool]
