"""Qdrant persistence for embedded chunks; retrieval remains intentionally absent."""

import hashlib
import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.models.embedding import EmbeddedChunk
from app.models.chunk import CodeChunk

logger = logging.getLogger(__name__)
POINT_NAMESPACE = uuid.UUID("0de7d8c3-5910-48bb-bd81-cc78732cc1ad")


@dataclass(frozen=True, slots=True)
class RepositoryIndexStatus:
    repository_id: str
    indexed_chunk_count: int


@dataclass(frozen=True, slots=True)
class StoredSearchResult:
    score: float
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class StoredChunk:
    chunk: CodeChunk


class QdrantStore:
    """Owns collection lifecycle and isolated repository-level persistence."""

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        if not collection_name.strip():
            raise ValueError("Qdrant collection name must not be blank")
        self._client = client
        self.collection_name = collection_name

    @classmethod
    def from_settings(cls, url: str | None, api_key: str | None, collection_name: str) -> "QdrantStore":
        if not url:
            raise ValueError("QDRANT_URL is required")
        return cls(QdrantClient(url=url, api_key=api_key), collection_name)

    def ensure_collection(self, vector_size: int) -> None:
        if vector_size < 1:
            raise ValueError("vector size must be positive")
        if self._client.collection_exists(self.collection_name):
            return
        logger.info("creating Qdrant collection %s with vector size %d", self.collection_name, vector_size)
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def upsert(self, embedded_chunks: Sequence[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        vector_size = len(embedded_chunks[0].vector)
        if any(len(chunk.vector) != vector_size for chunk in embedded_chunks):
            raise ValueError("all vectors in an upsert must have the same dimension")
        self.ensure_collection(vector_size)
        points = [
            models.PointStruct(
                id=deterministic_point_id(chunk.chunk_id),
                vector=chunk.vector,
                payload=chunk_payload(chunk),
            )
            for chunk in embedded_chunks
        ]
        self._client.upsert(collection_name=self.collection_name, points=points, wait=True)
        logger.info("upserted %d chunks into %s", len(points), self.collection_name)

    def replace_repository(self, repository_id: str, embedded_chunks: Sequence[EmbeddedChunk]) -> None:
        """Replace a repository atomically at the logical level, avoiding stale points."""
        self.delete_repository(repository_id)
        self.upsert(embedded_chunks)

    def delete_repository(self, repository_id: str) -> None:
        if not self._client.collection_exists(self.collection_name):
            return
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=self.repository_filter(repository_id)),
            wait=True,
        )
        logger.info("deleted indexed chunks for repository %s", repository_id)

    def repository_status(self, repository_id: str) -> RepositoryIndexStatus:
        if not self._client.collection_exists(self.collection_name):
            return RepositoryIndexStatus(repository_id=repository_id, indexed_chunk_count=0)
        result = self._client.count(
            collection_name=self.collection_name,
            count_filter=self.repository_filter(repository_id),
            exact=True,
        )
        return RepositoryIndexStatus(repository_id=repository_id, indexed_chunk_count=result.count)

    def similarity_search(
        self,
        repository_id: str,
        query_vector: Sequence[float],
        limit: int,
        score_threshold: float | None = None,
        metadata_filters: Mapping[str, str | int | bool] | None = None,
    ) -> list[StoredSearchResult]:
        """Run a repository-scoped vector search; answer generation is out of scope."""
        if limit < 1:
            raise ValueError("search limit must be positive")
        if not self._client.collection_exists(self.collection_name):
            return []
        criteria: dict[str, str | int | bool] = {"repository_id": repository_id}
        if metadata_filters:
            criteria.update(metadata_filters)
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            query_filter=metadata_filter(criteria),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return [
            StoredSearchResult(score=point.score, payload=dict(point.payload or {}))
            for point in response.points
        ]

    def list_chunks(
        self, repository_id: str, filters: Mapping[str, str | int | bool] | None = None, limit: int = 200
    ) -> list[StoredChunk]:
        """List exact repository chunks by metadata for file/symbol inspection."""
        if limit < 1 or not self._client.collection_exists(self.collection_name):
            return []
        criteria: dict[str, str | int | bool] = {"repository_id": repository_id}
        if filters:
            criteria.update(filters)
        points, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=metadata_filter(criteria),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        chunks = [StoredChunk(chunk=chunk_from_payload(dict(point.payload or {}))) for point in points]
        return sorted(chunks, key=lambda item: (item.chunk.file_path, item.chunk.start_line, item.chunk.chunk_id))

    @staticmethod
    def repository_filter(repository_id: str) -> models.Filter:
        return metadata_filter({"repository_id": repository_id})


def deterministic_point_id(chunk_id: str) -> str:
    """Derive a repeatable Qdrant UUID from the deterministic chunk identifier."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def chunk_payload(embedded_chunk: EmbeddedChunk) -> dict[str, str | int | bool | None]:
    chunk = embedded_chunk.chunk
    return {
        "repository_id": chunk.repository_id,
        "chunk_id": chunk.chunk_id,
        "file_path": chunk.file_path,
        "language": chunk.language,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "symbol_name": chunk.symbol_name,
        "chunk_type": chunk.chunk_type,
        "content": chunk.content,
        "content_hash": hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
        **chunk.metadata,
    }


def metadata_filter(criteria: Mapping[str, str | int | bool]) -> models.Filter:
    """Build a Qdrant payload filter for supported metadata equality conditions."""
    return models.Filter(
        must=[
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in criteria.items()
        ]
    )


def chunk_from_payload(payload: Mapping[str, object]) -> CodeChunk:
    """Reconstruct a chunk from Qdrant payload metadata without trusting external callers."""
    core_keys = {
        "repository_id", "chunk_id", "file_path", "language", "start_line", "end_line",
        "symbol_name", "chunk_type", "content", "content_hash",
    }
    return CodeChunk(
        chunk_id=str(payload["chunk_id"]),
        repository_id=str(payload["repository_id"]),
        file_path=str(payload["file_path"]),
        language=str(payload["language"]),
        content=str(payload["content"]),
        start_line=int(payload["start_line"]),
        end_line=int(payload["end_line"]),
        symbol_name=str(payload["symbol_name"]) if payload.get("symbol_name") is not None else None,
        chunk_type=str(payload["chunk_type"]),
        metadata={key: value for key, value in payload.items() if key not in core_keys and isinstance(value, (str, int, bool))},
    )
