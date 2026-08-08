"""Coordinates parsing, embedding, and storage without exposing retrieval behavior."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.models.repository import RepositoryFile
from app.services.chunking import CodeChunkingService
from app.services.embedding import EmbeddingService
from app.services.lexical import BM25Index, get_lexical_index
from app.services.qdrant_store import QdrantStore, RepositoryIndexStatus


@dataclass(frozen=True, slots=True)
class IndexingResult:
    repository_id: str
    chunk_count: int
    vector_dimension: int


class RepositoryIndexingService:
    """Index one repository through the existing, independently testable stages."""

    def __init__(
        self,
        chunking_service: CodeChunkingService,
        embedding_service: EmbeddingService,
        vector_store: QdrantStore,
        lexical_index: BM25Index | None = None,
    ) -> None:
        self._chunking_service = chunking_service
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._lexical_index = lexical_index or get_lexical_index()

    def index_repository(self, repository_id: str, files: Sequence[RepositoryFile]) -> IndexingResult:
        chunks = self._chunking_service.chunk_files(repository_id, files)
        embedded_chunks = self._embedding_service.embed_chunks(chunks)
        self._vector_store.replace_repository(repository_id, embedded_chunks)
        self._lexical_index.index_chunks(repository_id, chunks)
        dimension = len(embedded_chunks[0].vector) if embedded_chunks else 0
        return IndexingResult(repository_id=repository_id, chunk_count=len(chunks), vector_dimension=dimension)

    def delete_repository(self, repository_id: str) -> None:
        self._vector_store.delete_repository(repository_id)
        self._lexical_index.delete_repository(repository_id)

    def repository_status(self, repository_id: str) -> RepositoryIndexStatus:
        return self._vector_store.repository_status(repository_id)
