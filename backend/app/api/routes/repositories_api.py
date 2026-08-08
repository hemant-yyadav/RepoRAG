from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.core.config import get_settings
from app.models.api import (
    RepositoryFileListItem,
    RepositoryFileListResponse,
    RepositoryIndexRequest,
    RepositoryStatusResponse,
    RepositorySummaryResponse,
    SearchRequest,
)
from app.models.code_intelligence import CodeSearchResponse, CodeSearchResponseItem
from app.services.code_intelligence import create_code_intelligence_service
from app.services.repository_ingestion import InvalidRepositoryUrlError
from app.services.repository_lifecycle import (
    RepositoryNotFoundError,
    RepositoryRecord,
    create_repository_lifecycle_service,
)

router = APIRouter(prefix="/repositories", tags=["repositories"])


def _summary(record: RepositoryRecord) -> RepositorySummaryResponse:
    return RepositorySummaryResponse(
        repository_id=record.repository_id,
        repository_url=record.repository_url,
        repository_name=record.repository_name,
        status=record.status,
        file_count=record.file_count,
        total_size_bytes=record.total_size_bytes,
        languages=record.languages,
    )


def _not_found(exc: RepositoryNotFoundError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post("/index", response_model=RepositorySummaryResponse, status_code=status.HTTP_202_ACCEPTED)
def index_repository(payload: RepositoryIndexRequest, background_tasks: BackgroundTasks) -> RepositorySummaryResponse:
    service = create_repository_lifecycle_service(get_settings())
    try:
        record = service.submit_indexing(str(payload.repository_url))
    except InvalidRepositoryUrlError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    background_tasks.add_task(service.run_indexing, record.repository_id)
    return _summary(record)


@router.get("", response_model=list[RepositorySummaryResponse])
def list_repositories() -> list[RepositorySummaryResponse]:
    return [_summary(record) for record in create_repository_lifecycle_service(get_settings()).list()]


@router.get("/{repository_id}", response_model=RepositorySummaryResponse)
def get_repository(repository_id: str) -> RepositorySummaryResponse:
    try:
        return _summary(create_repository_lifecycle_service(get_settings()).get(repository_id))
    except RepositoryNotFoundError as exc:
        raise _not_found(exc) from exc


@router.delete("/{repository_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_repository(repository_id: str) -> None:
    try:
        create_repository_lifecycle_service(get_settings()).delete(repository_id)
    except RepositoryNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/{repository_id}/status", response_model=RepositoryStatusResponse)
def repository_status(repository_id: str) -> RepositoryStatusResponse:
    try:
        record = create_repository_lifecycle_service(get_settings()).get(repository_id)
    except RepositoryNotFoundError as exc:
        raise _not_found(exc) from exc
    return RepositoryStatusResponse(**_summary(record).model_dump(), indexed_chunk_count=record.indexed_chunk_count, error=record.error)


@router.get("/{repository_id}/files", response_model=RepositoryFileListResponse)
def list_files(repository_id: str) -> RepositoryFileListResponse:
    chunks = create_code_intelligence_service(get_settings()).list_files(repository_id)
    grouped: dict[tuple[str, str], int] = {}
    for item in chunks:
        key = (item.chunk.file_path, item.chunk.language)
        grouped[key] = grouped.get(key, 0) + 1
    return RepositoryFileListResponse(
        repository_id=repository_id,
        files=[RepositoryFileListItem(file_path=path, language=language, chunk_count=count) for (path, language), count in sorted(grouped.items())],
    )


@router.post("/search", response_model=CodeSearchResponse, tags=["search"])
def search(payload: SearchRequest) -> CodeSearchResponse:
    try:
        results = create_code_intelligence_service(get_settings()).search(
            payload.repository_id, payload.query, payload.top_k, payload.file_path, payload.symbol_name, payload.language
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return CodeSearchResponse(results=[
        CodeSearchResponseItem(
            chunk_id=result.chunk.chunk_id, file_path=result.chunk.file_path, language=result.chunk.language,
            start_line=result.chunk.start_line, end_line=result.chunk.end_line, symbol_name=result.chunk.symbol_name,
            chunk_type=result.chunk.chunk_type, content=result.chunk.content,
            rank=result.rank, relevance_score=result.relevance_score,
        )
        for result in results
    ])
