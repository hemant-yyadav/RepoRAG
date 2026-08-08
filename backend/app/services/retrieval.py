"""Basic repository-isolated semantic retrieval, without generation or reranking."""

from collections.abc import Sequence
from collections.abc import Mapping
from typing import Protocol

from app.core.config import Settings
from app.models.chunk import CodeChunk
from app.models.retrieval import RetrievalResult
from app.services.embedding import create_embedding_service
from app.services.qdrant_store import QdrantStore, StoredSearchResult


class QueryEmbedder(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return vectors in the same order as the supplied texts."""


class VectorSearcher(Protocol):
    def similarity_search(
        self,
        repository_id: str,
        query_vector: Sequence[float],
        limit: int,
        score_threshold: float | None = None,
        metadata_filters: Mapping[str, str | int | bool] | None = None,
    ) -> list[StoredSearchResult]:
        """Return only points belonging to the requested repository."""


class RetrievalService:
    def __init__(
        self,
        embedder: QueryEmbedder,
        vector_store: VectorSearcher,
        default_top_k: int = 5,
        default_score_threshold: float | None = None,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default top_k must be positive")
        self._embedder = embedder
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._default_score_threshold = default_score_threshold

    def retrieve(
        self,
        repository_id: str,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filters: Mapping[str, str | int | bool] | None = None,
    ) -> list[RetrievalResult]:
        if not repository_id.strip():
            raise ValueError("repository_id must not be blank")
        if not query.strip():
            raise ValueError("query must not be blank")
        limit = top_k if top_k is not None else self._default_top_k
        threshold = score_threshold if score_threshold is not None else self._default_score_threshold
        if limit < 1:
            raise ValueError("top_k must be positive")

        vectors = self._embedder.embed_texts([query])
        if len(vectors) != 1:
            raise ValueError("query embedding provider returned an invalid vector count")
        stored_results = self._vector_store.similarity_search(repository_id, vectors[0], limit, threshold, metadata_filters)
        return [self._to_result(stored) for stored in stored_results]

    @staticmethod
    def _to_result(stored: StoredSearchResult) -> RetrievalResult:
        payload = stored.payload
        metadata_keys = {"repository_id", "chunk_id", "file_path", "language", "start_line", "end_line", "symbol_name", "chunk_type", "content", "content_hash"}
        chunk = CodeChunk(
            chunk_id=str(payload["chunk_id"]),
            repository_id=str(payload["repository_id"]),
            file_path=str(payload["file_path"]),
            language=str(payload["language"]),
            content=str(payload["content"]),
            start_line=int(payload["start_line"]),
            end_line=int(payload["end_line"]),
            symbol_name=str(payload["symbol_name"]) if payload.get("symbol_name") is not None else None,
            chunk_type=str(payload["chunk_type"]),
            metadata={key: value for key, value in payload.items() if key not in metadata_keys and isinstance(value, (str, int, bool))},
        )
        return RetrievalResult(chunk=chunk, score=stored.score, metadata=chunk.metadata)


def create_retrieval_service(settings: Settings) -> RetrievalService:
    return RetrievalService(
        embedder=create_embedding_service(settings),
        vector_store=QdrantStore.from_settings(
            settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection_name
        ),
        default_top_k=settings.retrieval_top_k,
        default_score_threshold=settings.retrieval_score_threshold,
    )
