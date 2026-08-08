from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.models.citation import AnswerRequest, AnswerResponse, CitationResponse
from app.services.generation import GenerationProviderError, create_generation_service
from app.services.reranked_retrieval import create_reranked_retrieval_service

router = APIRouter(prefix="/answers", tags=["answers"])


@router.post("/generate", response_model=AnswerResponse)
def generate_answer(payload: AnswerRequest) -> AnswerResponse:
    """Generate a grounded answer and return only citations verified by the backend."""
    try:
        settings = get_settings()
        retrieved = create_reranked_retrieval_service(settings).retrieve(
            repository_id=payload.repository_id,
            query=payload.query,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
        )
        generated = create_generation_service(settings).generate(payload.query, retrieved)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except GenerationProviderError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return AnswerResponse(
        answer=generated.answer,
        sources=[
            CitationResponse(
                citation_id=citation.citation_id,
                file_path=citation.file_path,
                start_line=citation.start_line,
                end_line=citation.end_line,
                symbol_name=citation.symbol_name,
                chunk_id=citation.chunk_id,
            )
            for citation in generated.sources
        ],
    )
