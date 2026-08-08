from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.models.citation import CitationResponse


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    role: str
    content: str


class ChatRequest(BaseModel):
    repository_id: str = Field(min_length=1, max_length=256)
    conversation_id: str | None = Field(default=None, max_length=256)
    message: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=100)


class ChatRetrievalDebug(BaseModel):
    standalone_query: str
    rank: int
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[CitationResponse]
    conversation_id: str
    retrieval: list[ChatRetrievalDebug]
