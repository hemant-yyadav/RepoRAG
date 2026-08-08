from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.models.citation import CitationResponse
from app.models.conversation import ChatRequest, ChatResponse, ChatRetrievalDebug
from app.services.conversation import get_conversation_service
from app.services.generation import GenerationProviderError

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    """Answer a multi-turn repository question using a standalone rewritten retrieval query."""
    try:
        result = get_conversation_service(get_settings()).chat(
            repository_id=payload.repository_id,
            conversation_id=payload.conversation_id,
            message=payload.message,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except GenerationProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return ChatResponse(
        answer=result.answer.answer,
        sources=[
            CitationResponse(
                citation_id=source.citation_id,
                file_path=source.file_path,
                start_line=source.start_line,
                end_line=source.end_line,
                symbol_name=source.symbol_name,
                chunk_id=source.chunk_id,
            )
            for source in result.answer.sources
        ],
        conversation_id=result.conversation_id,
        retrieval=[
            ChatRetrievalDebug(
                standalone_query=result.standalone_query,
                rank=item.rank,
                file_path=item.chunk.file_path,
                start_line=item.chunk.start_line,
                end_line=item.chunk.end_line,
                symbol_name=item.chunk.symbol_name,
                relevance_score=item.relevance_score,
            )
            for item in result.retrieved
        ],
    )
