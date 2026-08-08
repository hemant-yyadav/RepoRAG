"""Thin composition of hybrid retrieval and neural reranking."""

from collections.abc import Mapping

from app.core.config import Settings
from app.models.reranking import RerankedRetrievalResult
from app.services.hybrid_retrieval import HybridRetrievalService, create_hybrid_retrieval_service
from app.services.reranking import RerankingService, create_reranking_service


class RerankedRetrievalService:
    def __init__(self, hybrid_service: HybridRetrievalService, reranking_service: RerankingService) -> None:
        self._hybrid_service = hybrid_service
        self._reranking_service = reranking_service

    def retrieve(
        self,
        repository_id: str,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filters: Mapping[str, str | int | bool] | None = None,
    ) -> list[RerankedRetrievalResult]:
        candidates = self._hybrid_service.retrieve(
            repository_id=repository_id,
            query=query,
            top_k=self._reranking_service.candidate_count,
            score_threshold=score_threshold,
            metadata_filters=metadata_filters,
        )
        return self._reranking_service.rerank(query, candidates, final_count=top_k)


def create_reranked_retrieval_service(settings: Settings) -> RerankedRetrievalService:
    return RerankedRetrievalService(
        hybrid_service=create_hybrid_retrieval_service(settings),
        reranking_service=create_reranking_service(settings),
    )
