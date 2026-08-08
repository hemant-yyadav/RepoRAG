from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from app.services.qdrant_store import StoredSearchResult
from app.services.retrieval import RetrievalService


def stored(score: float, chunk_id: str) -> StoredSearchResult:
    return StoredSearchResult(
        score=score,
        payload={
            "repository_id": "repo-a",
            "chunk_id": chunk_id,
            "file_path": f"src/{chunk_id}.py",
            "language": "Python",
            "start_line": 10,
            "end_line": 12,
            "symbol_name": "authenticate",
            "chunk_type": "function",
            "content": "def authenticate(): pass",
            "content_hash": "hash",
        },
    )


@dataclass
class FakeEmbedder:
    queries: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.queries.append(texts)
        return [[0.2, 0.8]]


@dataclass
class FakeVectorSearcher:
    results: list[StoredSearchResult] = field(default_factory=list)
    calls: list[tuple[str, list[float], int, float | None]] = field(default_factory=list)

    def similarity_search(
        self, repository_id: str, query_vector: list[float], limit: int, score_threshold: float | None = None
    ) -> list[StoredSearchResult]:
        self.calls.append((repository_id, query_vector, limit, score_threshold))
        return self.results


def test_retrieval_preserves_ranked_qdrant_order_and_metadata() -> None:
    searcher = FakeVectorSearcher(results=[stored(0.95, "first"), stored(0.71, "second")])
    service = RetrievalService(FakeEmbedder(), searcher, default_top_k=5)  # type: ignore[arg-type]

    results = service.retrieve("repo-a", "Where is JWT authentication implemented?")

    assert [result.chunk.chunk_id for result in results] == ["first", "second"]
    assert [result.score for result in results] == [0.95, 0.71]
    assert results[0].chunk.file_path == "src/first.py"
    assert searcher.calls == [("repo-a", [0.2, 0.8], 5, None)]


def test_retrieval_forwards_repository_top_k_and_score_threshold() -> None:
    searcher = FakeVectorSearcher()
    service = RetrievalService(FakeEmbedder(), searcher)  # type: ignore[arg-type]

    assert service.retrieve("isolated-repository", "find auth", top_k=3, score_threshold=0.6) == []

    assert searcher.calls == [("isolated-repository", [0.2, 0.8], 3, 0.6)]


def test_retrieval_returns_empty_results() -> None:
    service = RetrievalService(FakeEmbedder(), FakeVectorSearcher())  # type: ignore[arg-type]

    assert service.retrieve("repo-a", "missing implementation") == []


@pytest.mark.parametrize("repository_id, query", [("", "auth"), ("repo-a", "  ")])
def test_retrieval_rejects_malformed_queries(repository_id: str, query: str) -> None:
    service = RetrievalService(FakeEmbedder(), FakeVectorSearcher())  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        service.retrieve(repository_id, query)


def test_retrieval_api_rejects_blank_query_before_provider_access() -> None:
    from app.main import app

    response = TestClient(app).post(
        "/retrieval/search", json={"repository_id": "repo-a", "query": ""}
    )

    assert response.status_code == 422
