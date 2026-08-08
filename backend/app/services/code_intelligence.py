"""Practical repository/file/symbol inspection built on indexed chunk metadata."""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from app.core.config import Settings
from app.models.retrieval import RetrievalResult
from app.models.reranking import RerankedRetrievalResult
from app.services.generation import create_generation_service
from app.services.qdrant_store import QdrantStore, StoredChunk
from app.services.reranked_retrieval import create_reranked_retrieval_service


class ChunkInspector(Protocol):
    def list_chunks(
        self, repository_id: str, filters: Mapping[str, str | int | bool] | None = None, limit: int = 200
    ) -> list[StoredChunk]:
        """Return repository-isolated chunks matching exact metadata."""


class RepositoryCodeIntelligenceService:
    def __init__(
        self,
        inspector: ChunkInspector,
        retriever: object,
        generator_factory: Callable[[], object] | None = None,
        max_file_chunks: int = 200,
    ) -> None:
        self._inspector = inspector
        self._retriever = retriever
        self._generator_factory = generator_factory
        self._max_file_chunks = max_file_chunks

    def get_file(self, repository_id: str, file_path: str) -> list[StoredChunk]:
        return self._inspector.list_chunks(repository_id, {"file_path": file_path}, self._max_file_chunks)

    def list_files(self, repository_id: str) -> list[StoredChunk]:
        return self._inspector.list_chunks(repository_id, limit=self._max_file_chunks)

    def get_symbol(self, repository_id: str, symbol_name: str) -> list[StoredChunk]:
        return self._inspector.list_chunks(repository_id, {"symbol_name": symbol_name}, self._max_file_chunks)

    def search(
        self,
        repository_id: str,
        query: str,
        top_k: int | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
        language: str | None = None,
    ) -> list[RerankedRetrievalResult]:
        filters = {
            key: value
            for key, value in {
                "file_path": file_path,
                "symbol_name": symbol_name,
                "language": language,
            }.items()
            if value
        }
        return self._retriever.retrieve(
            repository_id, query, top_k=top_k, metadata_filters=filters or None
        )

    def explain_file(self, repository_id: str, file_path: str):
        chunks = self.get_file(repository_id, file_path)
        context = [RetrievalResult(chunk=item.chunk, score=1.0, metadata=item.chunk.metadata) for item in chunks]
        if self._generator_factory is None:
            raise ValueError("generation is not configured")
        return self._generator_factory().generate(f"Explain the file {file_path}.", context)


def create_code_intelligence_service(settings: Settings) -> RepositoryCodeIntelligenceService:
    return RepositoryCodeIntelligenceService(
        inspector=QdrantStore.from_settings(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection_name),
        retriever=create_reranked_retrieval_service(settings),
        generator_factory=lambda: create_generation_service(settings),
        max_file_chunks=settings.file_inspection_max_chunks,
    )
