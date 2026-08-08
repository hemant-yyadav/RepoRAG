from pydantic import BaseModel, Field

from app.models.citation import CitationResponse


class CodeChunkResponse(BaseModel):
    chunk_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol_name: str | None
    chunk_type: str
    content: str


class FileExplainRequest(BaseModel):
    file_path: str = Field(min_length=1, max_length=1_024)


class FileExplainResponse(BaseModel):
    file_path: str
    answer: str
    sources: list[CitationResponse]


class CodeSearchResponseItem(CodeChunkResponse):
    rank: int
    relevance_score: float


class CodeSearchResponse(BaseModel):
    results: list[CodeSearchResponseItem]
