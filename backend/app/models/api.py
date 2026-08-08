from pydantic import BaseModel, Field, HttpUrl


class RepositoryIndexRequest(BaseModel):
    repository_url: HttpUrl


class RepositorySummaryResponse(BaseModel):
    repository_id: str
    repository_url: str
    repository_name: str
    status: str
    file_count: int = 0
    total_size_bytes: int = 0
    languages: list[str] = []


class RepositoryStatusResponse(RepositorySummaryResponse):
    indexed_chunk_count: int = 0
    error: str | None = None


class RepositoryFileListItem(BaseModel):
    file_path: str
    language: str
    chunk_count: int


class RepositoryFileListResponse(BaseModel):
    repository_id: str
    files: list[RepositoryFileListItem]


class SearchRequest(BaseModel):
    repository_id: str = Field(min_length=1, max_length=256)
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    file_path: str | None = Field(default=None, max_length=1_024)
    symbol_name: str | None = Field(default=None, max_length=512)
    language: str | None = Field(default=None, max_length=128)
