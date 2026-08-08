from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.models.repository import IngestRepositoryRequest, IngestRepositoryResponse
from app.services.repository_ingestion import (
    InvalidRepositoryUrlError,
    RepositoryCloneError,
    RepositoryIngestionService,
)

router = APIRouter(prefix="/repositories", tags=["repositories"])


@router.post("/ingest", response_model=IngestRepositoryResponse, status_code=status.HTTP_200_OK)
def ingest_repository(payload: IngestRepositoryRequest) -> IngestRepositoryResponse:
    """Ingest a public GitHub repository without creating embeddings or an index."""
    settings = get_settings()
    service = RepositoryIngestionService(
        max_file_size_bytes=settings.max_file_size_bytes,
        clone_timeout_seconds=settings.git_clone_timeout_seconds,
    )
    try:
        result = service.ingest(payload.repository_url)
    except InvalidRepositoryUrlError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RepositoryCloneError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return IngestRepositoryResponse(
        repository_url=result.repository_url,
        repository_name=result.repository_name,
        file_count=len(result.files),
        total_size_bytes=result.total_size_bytes,
        languages=result.languages,
    )
