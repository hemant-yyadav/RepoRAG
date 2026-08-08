"""Repository lifecycle orchestration for production-style HTTP endpoints."""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.resilience import retry_operation
from app.services.indexing import RepositoryIndexingService
from app.services.repository_ingestion import (
    RepositoryIngestionService,
    parse_github_repository_url,
)

logger = logging.getLogger(__name__)


class RepositoryNotFoundError(Exception):
    """The requested repository does not exist in this application process."""


@dataclass(slots=True)
class RepositoryRecord:
    repository_id: str
    repository_url: str
    repository_name: str
    status: str
    file_count: int = 0
    total_size_bytes: int = 0
    languages: list[str] = field(default_factory=list)
    indexed_chunk_count: int = 0
    error: str | None = None
    stage: str = "queued"
    progress_percent: int = 0
    content_signature: str | None = None


class RepositoryRegistry:
    """Process-local status registry; storage can later be swapped without route changes."""

    def __init__(self) -> None:
        self._records: dict[str, RepositoryRecord] = {}

    def create_or_replace(self, repository_url: str, repository_name: str) -> RepositoryRecord:
        repository_id = repository_id_from_url(repository_url)
        record = self._records.get(repository_id)
        if record:
            record.status = "indexing"
            record.stage = "queued"
            record.progress_percent = 0
            record.error = None
            return record
        record = RepositoryRecord(repository_id, repository_url, repository_name, status="indexing")
        self._records[repository_id] = record
        return record

    def get(self, repository_id: str) -> RepositoryRecord:
        try:
            return self._records[repository_id]
        except KeyError as exc:
            raise RepositoryNotFoundError("Repository was not found") from exc

    def list(self) -> list[RepositoryRecord]:
        return sorted(self._records.values(), key=lambda record: record.repository_name.lower())

    def delete(self, repository_id: str) -> None:
        self.get(repository_id)
        del self._records[repository_id]


class RepositoryLifecycleService:
    def __init__(
        self,
        registry: RepositoryRegistry,
        ingestion_factory: Callable[[], RepositoryIngestionService],
        indexing_factory: Callable[[], RepositoryIndexingService],
        max_indexing_retries: int = 1,
        indexing_initial_backoff_seconds: float = 1.0,
    ) -> None:
        self._registry = registry
        self._ingestion_factory = ingestion_factory
        self._indexing_factory = indexing_factory
        self._max_indexing_retries = max_indexing_retries
        self._indexing_initial_backoff_seconds = indexing_initial_backoff_seconds

    def submit_indexing(self, repository_url: str) -> RepositoryRecord:
        parsed = parse_github_repository_url(repository_url)
        return self._registry.create_or_replace(parsed.canonical_url.removesuffix(".git"), parsed.name)

    def run_indexing(self, repository_id: str) -> None:
        record = self._registry.get(repository_id)
        try:
            record.stage = "ingesting"
            record.progress_percent = 15
            ingested = self._ingestion_factory().ingest(record.repository_url)
            signature = repository_content_signature(ingested.files)
            if signature == record.content_signature and record.status == "indexing":
                record.status = "ready"
                record.stage = "completed"
                record.progress_percent = 100
                logger.info("repository %s unchanged; skipping re-index", repository_id)
                return
            record.stage = "indexing"
            record.progress_percent = 45
            indexed = retry_operation(
                "repository_indexing",
                lambda: self._indexing_factory().index_repository(repository_id, ingested.files),
                (Exception,),
                self._max_indexing_retries,
                self._indexing_initial_backoff_seconds,
                logger,
            )
            record.status = "ready"
            record.file_count = len(ingested.files)
            record.total_size_bytes = ingested.total_size_bytes
            record.languages = ingested.languages
            record.indexed_chunk_count = indexed.chunk_count
            record.error = None
            record.stage = "completed"
            record.progress_percent = 100
            record.content_signature = signature
            logger.info("repository %s indexed successfully", repository_id)
        except Exception:  # Errors remain server-side; the status API exposes only a safe message.
            logger.exception("repository indexing failed for %s", repository_id)
            record.status = "failed"
            record.stage = "failed"
            record.error = "Indexing failed. Check server logs for details."

    def get(self, repository_id: str) -> RepositoryRecord:
        return self._registry.get(repository_id)

    def list(self) -> list[RepositoryRecord]:
        return self._registry.list()

    def delete(self, repository_id: str) -> None:
        self._registry.get(repository_id)
        self._indexing_factory().delete_repository(repository_id)
        self._registry.delete(repository_id)


def repository_id_from_url(repository_url: str) -> str:
    return "repo_" + hashlib.sha256(repository_url.encode("utf-8")).hexdigest()[:16]


def repository_content_signature(files) -> str:
    """Stable signature makes an unchanged re-index a no-op within the lifecycle registry."""
    digest = hashlib.sha256()
    for repository_file in sorted(files, key=lambda item: item.path):
        digest.update(repository_file.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(repository_file.content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


_shared_registry = RepositoryRegistry()


def create_repository_lifecycle_service(settings: Settings) -> RepositoryLifecycleService:
    from app.services.chunking import CodeChunkingService
    from app.services.embedding import create_embedding_service
    from app.services.qdrant_store import QdrantStore

    return RepositoryLifecycleService(
        registry=_shared_registry,
        ingestion_factory=lambda: RepositoryIngestionService(
            settings.max_file_size_bytes,
            settings.git_clone_timeout_seconds,
            settings.max_repository_files,
            settings.max_repository_size_bytes,
            settings.github_max_retries,
            settings.github_initial_backoff_seconds,
        ),
        indexing_factory=lambda: RepositoryIndexingService(
            CodeChunkingService(),
            create_embedding_service(settings),
            QdrantStore.from_settings(
                settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection_name,
                settings.qdrant_max_retries, settings.qdrant_initial_backoff_seconds,
            ),
        ),
        max_indexing_retries=settings.indexing_max_retries,
        indexing_initial_backoff_seconds=settings.indexing_initial_backoff_seconds,
    )
