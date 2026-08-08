from dataclasses import dataclass

from app.models.chunk import CodeChunk


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A chunk and its vector, ready for a later vector-storage phase."""

    chunk_id: str
    vector: list[float]
    chunk: CodeChunk
