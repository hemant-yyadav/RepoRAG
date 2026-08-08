"""Repository lifecycle orchestration for production-style HTTP endpoints."""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.config import Settings
from app.services.indexing import RepositoryIndexingService
from app.services.repository_ingestion import (
    InvalidRepositoryUrlError,
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


class RepositoryRegistry:
    """Process-local status registry; storage can later be swapped without route changes."""

    def __init__(self) -> None:
        self._records: dict[str, RepositoryRecord] = {}

    def create_or_replace(self, repository_url: str, repository_name: str) -> RepositoryRecord:
        repository_id = repository_id_from_url(repository_url)
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
    ) -> None:
        self._registry = registry
        self._ingestion_factory = ingestion_factory
        self._indexing_factory = indexing_factory

    def submit_indexing(self, repository_url: str) -> RepositoryRecord:
        parsed = parse_github_repository_url(repository_url)
        return self._registry.create_or_replace(parsed.canonical_url.removesuffix(".git"), parsed.name)

    def run_indexing(self, repository_id: str) -> None:
        record = self._registry.get(repository_id)
        try:
            ingested = self._ingestion_factory().ingest(record.repository_url)
            indexed = self._indexing_factory().index_repository(repository_id, ingested.files)
            record.status = "ready"
            record.file_count = len(ingested.files)
            record.total_size_bytes = ingested.total_size_bytes
            record.languages = ingested.languages
            record.indexed_chunk_count = indexed.chunk_count
            record.error = None
            logger.info("repository %s indexed successfully", repository_id)
        except Exception:  # Errors remain server-side; the status API exposes only a safe message.
            logger.exception("repository indexing failed for %s", repository_id)
            record.status = "failed"
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


_shared_registry = RepositoryRegistry()


def create_repository_lifecycle_service(settings: Settings) -> RepositoryLifecycleService:
    from app.services.chunking import CodeChunkingService
    from app.services.embedding import create_embedding_service
    from app.services.qdrant_store import QdrantStore

    return RepositoryLifecycleService(
        registry=_shared_registry,
        ingestion_factory=lambda: RepositoryIngestionService(
            settings.max_file_size_bytes, settings.git_clone_timeout_seconds
        ),
        indexing_factory=lambda: RepositoryIndexingService(
            CodeChunkingService(),
            create_embedding_service(settings),
            QdrantStore.from_settings(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection_name),
        ),
    )
