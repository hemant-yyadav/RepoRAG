from dataclasses import dataclass
from types import SimpleNamespace

from qdrant_client import models

from app.models.chunk import CodeChunk
from app.models.embedding import EmbeddedChunk
from app.services.qdrant_store import QdrantStore, chunk_payload, deterministic_point_id, metadata_filter


@dataclass
class CountResult:
    count: int


class FakeQdrantClient:
    def __init__(self) -> None:
        self.exists = False
        self.create_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.delete_calls: list[dict[str, object]] = []
        self.count_calls: list[dict[str, object]] = []

    def collection_exists(self, _: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs: object) -> None:
        self.exists = True
        self.create_calls.append(kwargs)

    def upsert(self, **kwargs: object) -> None:
        self.upsert_calls.append(kwargs)

    def delete(self, **kwargs: object) -> None:
        self.delete_calls.append(kwargs)

    def count(self, **kwargs: object) -> CountResult:
        self.count_calls.append(kwargs)
        return CountResult(count=7)

    def query_points(self, **kwargs: object) -> SimpleNamespace:
        self.query_call = kwargs
        return SimpleNamespace(points=[])


def make_embedded(repository_id: str = "repo-a", chunk_id: str = "chunk-1") -> EmbeddedChunk:
    chunk = CodeChunk(
        chunk_id=chunk_id,
        repository_id=repository_id,
        file_path="src/auth/service.py",
        language="Python",
        content="def authenticate():\n    return True\n",
        start_line=42,
        end_line=43,
        symbol_name="authenticate",
        chunk_type="function",
        metadata={"part": 1, "parts": 1},
    )
    return EmbeddedChunk(chunk_id=chunk_id, vector=[0.1, 0.2, 0.3], chunk=chunk)


def test_upsert_creates_cosine_collection_and_points() -> None:
    client = FakeQdrantClient()
    store = QdrantStore(client, "chunks")  # type: ignore[arg-type]

    store.upsert([make_embedded()])

    assert len(client.create_calls) == 1
    vector_config = client.create_calls[0]["vectors_config"]
    assert vector_config.size == 3
    assert vector_config.distance == models.Distance.COSINE
    assert len(client.upsert_calls) == 1
    assert client.upsert_calls[0]["wait"] is True
    assert client.upsert_calls[0]["points"][0].id == deterministic_point_id("chunk-1")


def test_point_ids_are_deterministic_and_payload_is_complete() -> None:
    embedded = make_embedded()
    payload = chunk_payload(embedded)

    assert deterministic_point_id("chunk-1") == deterministic_point_id("chunk-1")
    assert deterministic_point_id("chunk-1") != deterministic_point_id("chunk-2")
    assert payload["repository_id"] == "repo-a"
    assert payload["file_path"] == "src/auth/service.py"
    assert payload["start_line"] == 42
    assert payload["end_line"] == 43
    assert payload["symbol_name"] == "authenticate"
    assert payload["chunk_type"] == "function"
    assert payload["content"] == embedded.chunk.content
    assert len(payload["content_hash"]) == 64


def test_delete_and_status_are_isolated_by_repository() -> None:
    client = FakeQdrantClient()
    client.exists = True
    store = QdrantStore(client, "chunks")  # type: ignore[arg-type]

    store.delete_repository("repo-a")
    status = store.repository_status("repo-b")

    delete_filter = client.delete_calls[0]["points_selector"].filter
    status_filter = client.count_calls[0]["count_filter"]
    assert delete_filter.must[0].match.value == "repo-a"
    assert status_filter.must[0].match.value == "repo-b"
    assert status.indexed_chunk_count == 7


def test_metadata_filter_supports_multiple_fields() -> None:
    filter_value = metadata_filter({"repository_id": "repo-a", "language": "Python"})

    assert [(condition.key, condition.match.value) for condition in filter_value.must] == [
        ("repository_id", "repo-a"),
        ("language", "Python"),
    ]


def test_replacing_repository_deletes_before_upserting() -> None:
    client = FakeQdrantClient()
    client.exists = True
    store = QdrantStore(client, "chunks")  # type: ignore[arg-type]

    store.replace_repository("repo-a", [make_embedded("repo-a")])

    assert len(client.delete_calls) == 1
    assert len(client.upsert_calls) == 1


def test_similarity_search_applies_repository_filter() -> None:
    client = FakeQdrantClient()
    client.exists = True
    store = QdrantStore(client, "chunks")  # type: ignore[arg-type]

    assert store.similarity_search("repo-a", [0.1, 0.2], limit=3, score_threshold=0.4) == []

    assert client.query_call["query"] == [0.1, 0.2]
    assert client.query_call["limit"] == 3
    assert client.query_call["score_threshold"] == 0.4
    assert client.query_call["query_filter"].must[0].match.value == "repo-a"
