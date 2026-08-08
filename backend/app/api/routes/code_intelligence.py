from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import get_settings
from app.models.citation import CitationResponse
from app.models.code_intelligence import (
    CodeChunkResponse,
    CodeSearchResponse,
    CodeSearchResponseItem,
    FileExplainRequest,
    FileExplainResponse,
)
from app.services.code_intelligence import create_code_intelligence_service
from app.services.generation import GenerationProviderError

router = APIRouter(prefix="/repositories/{repository_id}", tags=["code intelligence"])


def _chunk_response(chunk) -> CodeChunkResponse:
    return CodeChunkResponse(
        chunk_id=chunk.chunk_id,
        file_path=chunk.file_path,
        language=chunk.language,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        symbol_name=chunk.symbol_name,
        chunk_type=chunk.chunk_type,
        content=chunk.content,
    )


@router.post("/files/explain", response_model=FileExplainResponse)
def explain_file(repository_id: str, payload: FileExplainRequest) -> FileExplainResponse:
    try:
        generated = create_code_intelligence_service(get_settings()).explain_file(repository_id, payload.file_path)
    except (ValueError, GenerationProviderError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return FileExplainResponse(
        file_path=payload.file_path,
        answer=generated.answer,
        sources=[CitationResponse(
            citation_id=source.citation_id, file_path=source.file_path, start_line=source.start_line,
            end_line=source.end_line, symbol_name=source.symbol_name, chunk_id=source.chunk_id,
        ) for source in generated.sources],
    )


@router.get("/files/{file_path:path}", response_model=list[CodeChunkResponse])
def get_file(repository_id: str, file_path: str) -> list[CodeChunkResponse]:
    chunks = create_code_intelligence_service(get_settings()).get_file(repository_id, file_path)
    return [_chunk_response(item.chunk) for item in chunks]


@router.get("/symbols/{symbol_name}", response_model=list[CodeChunkResponse])
def get_symbol(repository_id: str, symbol_name: str) -> list[CodeChunkResponse]:
    chunks = create_code_intelligence_service(get_settings()).get_symbol(repository_id, symbol_name)
    return [_chunk_response(item.chunk) for item in chunks]


@router.get("/search", response_model=CodeSearchResponse)
def search_code(
    repository_id: str,
    query: str = Query(min_length=1, max_length=10_000),
    top_k: int | None = Query(default=None, ge=1, le=100),
    file_path: str | None = None,
    symbol_name: str | None = None,
    language: str | None = None,
) -> CodeSearchResponse:
    try:
        results = create_code_intelligence_service(get_settings()).search(
            repository_id, query, top_k, file_path, symbol_name, language
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return CodeSearchResponse(results=[
        CodeSearchResponseItem(**_chunk_response(result.chunk).model_dump(), rank=result.rank, relevance_score=result.relevance_score)
        for result in results
    ])
