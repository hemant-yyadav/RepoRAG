from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """Backend-owned source metadata for one retrieved chunk."""

    citation_id: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None
    chunk_id: str


class AnswerRequest(BaseModel):
    repository_id: str = Field(min_length=1, max_length=256)
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)


class CitationResponse(BaseModel):
    citation_id: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None
    chunk_id: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[CitationResponse]
