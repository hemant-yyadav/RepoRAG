from dataclasses import dataclass

from app.models.chunk import CodeChunk


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    """A fused candidate with enough rank detail to evaluate retrieval quality."""

    rank: int
    chunk: CodeChunk
    fused_score: float
    vector_rank: int | None
    bm25_rank: int | None
    metadata: dict[str, str | int | bool]
