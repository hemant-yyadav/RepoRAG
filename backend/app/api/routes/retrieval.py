from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.models.retrieval import RetrievalRequest, RetrievalResponse, RetrievalResponseItem
from app.services.retrieval import create_retrieval_service

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalResponse)
def search(payload: RetrievalRequest) -> RetrievalResponse:
    """Return ranked semantic matches, without generating an answer."""
    try:
        results = create_retrieval_service(get_settings()).retrieve(
            repository_id=payload.repository_id,
            query=payload.query,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return RetrievalResponse(
        repository_id=payload.repository_id,
        query=payload.query,
        results=[
            RetrievalResponseItem(
                rank=rank,
                score=result.score,
                file_path=result.chunk.file_path,
                start_line=result.chunk.start_line,
                end_line=result.chunk.end_line,
                symbol_name=result.chunk.symbol_name,
                chunk_type=result.chunk.chunk_type,
                content=result.chunk.content,
            )
            for rank, result in enumerate(results, start=1)
        ],
    )
