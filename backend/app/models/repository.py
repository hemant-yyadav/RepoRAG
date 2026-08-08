from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class RepositoryFile:
    """A supported, normalized text file from an ingested repository."""

    path: str
    language: str
    content: str
    size_bytes: int
    line_count: int


class IngestRepositoryRequest(BaseModel):
    repository_url: str = Field(min_length=1, max_length=2_048)


class IngestRepositoryResponse(BaseModel):
    repository_url: str
    repository_name: str
    file_count: int
    total_size_bytes: int
    languages: list[str]
