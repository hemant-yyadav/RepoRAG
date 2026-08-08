from dataclasses import dataclass, field

import pytest

from app.models.chunk import CodeChunk
from app.models.hybrid_retrieval import HybridRetrievalResult
from app.services.reranking import RerankerProviderError, RerankingConfig, RerankingService


def candidate(chunk_id: str, rank: int, score: float) -> HybridRetrievalResult:
    chunk = CodeChunk(
        chunk_id=chunk_id,
        repository_id="repo-a",
        file_path=f"src/{chunk_id}.py",
        language="Python",
        content=f"def {chunk_id}(): pass",
        start_line=10,
        end_line=10,
        symbol_name=chunk_id,
        chunk_type="function",
        metadata={"part": 1},
    )
    return HybridRetrievalResult(rank, chunk, score, rank, None, chunk.metadata)


@dataclass
class FakeReranker:
    scores: list[float] = field(default_factory=list)
    error: Exception | None = None
    calls: list[tuple[str, list[str], str, int]] = field(default_factory=list)

    def rerank(self, query: str, documents: list[str], model: str, batch_size: int) -> list[float]:
        self.calls.append((query, documents, model, batch_size))
        if self.error:
            raise self.error
        return self.scores


def test_reranking_orders_by_relevance_and_preserves_metadata() -> None:
    provider = FakeReranker(scores=[0.2, 0.9, 0.5])
    service = RerankingService(provider, RerankingConfig(model="test", final_count=2, batch_size=7))

    results = service.rerank("find authentication", [candidate("a", 1, 0.8), candidate("b", 2, 0.7), candidate("c", 3, 0.6)])

    assert [result.chunk.chunk_id for result in results] == ["b", "c"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].metadata == {"part": 1}
    assert provider.calls[0][3] == 7


def test_reranking_respects_candidate_and_final_counts() -> None:
    provider = FakeReranker(scores=[0.1, 0.9])
    service = RerankingService(provider, RerankingConfig(model="test", candidate_count=2, final_count=1))

    results = service.rerank("query", [candidate("a", 1, 0.8), candidate("b", 2, 0.7), candidate("c", 3, 0.6)])

    assert [result.chunk.chunk_id for result in results] == ["b"]
    assert len(provider.calls[0][1]) == 2


def test_empty_candidates_skip_reranker() -> None:
    provider = FakeReranker()
    service = RerankingService(provider, RerankingConfig(model="test"))

    assert service.rerank("query", []) == []
    assert provider.calls == []


def test_score_ties_are_deterministic_by_prior_rank() -> None:
    service = RerankingService(FakeReranker(scores=[0.5, 0.5]), RerankingConfig(model="test"))

    results = service.rerank("query", [candidate("later", 2, 0.7), candidate("first", 1, 0.6)])

    assert [result.chunk.chunk_id for result in results] == ["first", "later"]


def test_reranker_failure_falls_back_to_hybrid_ranking() -> None:
    provider = FakeReranker(error=RerankerProviderError("offline"))
    service = RerankingService(provider, RerankingConfig(model="test", fail_open=True))

    results = service.rerank("query", [candidate("a", 1, 0.9), candidate("b", 2, 0.7)])

    assert [result.chunk.chunk_id for result in results] == ["a", "b"]
    assert [result.relevance_score for result in results] == [0.9, 0.7]


def test_reranker_failure_can_fail_closed() -> None:
    service = RerankingService(
        FakeReranker(error=RerankerProviderError("offline")),
        RerankingConfig(model="test", fail_open=False),
    )

    with pytest.raises(RerankerProviderError):
        service.rerank("query", [candidate("a", 1, 0.9)])
