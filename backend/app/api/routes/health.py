import logging

from fastapi import APIRouter, HTTPException, status
from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.models.health import HealthResponse

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return the availability of the API process."""
    return HealthResponse(status="ok")


@router.get("/health/qdrant", response_model=HealthResponse)
async def qdrant_health_check() -> HealthResponse:
    """Check Qdrant connectivity for deployment diagnostics without leaking details."""
    settings = get_settings()
    if not settings.qdrant_url:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Qdrant is not configured")
    try:
        QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key).get_collections()
    except Exception:
        logger.warning("qdrant health check failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Qdrant is unavailable") from None
    return HealthResponse(status="ok")
