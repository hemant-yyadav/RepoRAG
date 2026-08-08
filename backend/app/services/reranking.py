"""Configurable neural reranking after hybrid retrieval."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.core.config import Settings
from app.core.observability import log_timing
from app.models.hybrid_retrieval import HybridRetrievalResult
from app.models.reranking import RerankedRetrievalResult

logger = logging.getLogger(__name__)


class RerankerProviderError(Exception):
    """A reranker could not score the supplied candidate documents."""


class RerankerProvider(Protocol):
    """Provider contract: return one relevance score per candidate, in input order."""

    def rerank(self, query: str, documents: Sequence[str], model: str, batch_size: int) -> list[float]:
        """Score query-document pairs in batches where the provider supports it."""


@dataclass(frozen=True, slots=True)
class RerankingConfig:
    model: str
    candidate_count: int = 20
    final_count: int = 5
    batch_size: int = 16
    fail_open: bool = True

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("reranker model must not be blank")
        if self.candidate_count < 1 or self.final_count < 1 or self.batch_size < 1:
            raise ValueError("reranker counts and batch size must be positive")


class CrossEncoderRerankerProvider:
    """Lazy local cross-encoder adapter suitable for code and natural-language matching."""

    def __init__(self) -> None:
        self._models: dict[str, object] = {}

    def rerank(self, query: str, documents: Sequence[str], model: str, batch_size: int) -> list[float]:
        if not documents:
            return []
        try:
            encoder = self._models.get(model)
            if encoder is None:
                from sentence_transformers import CrossEncoder

                encoder = CrossEncoder(model)
                self._models[model] = encoder
            scores = encoder.predict([(query, document) for document in documents], batch_size=batch_size)
            return [float(score) for score in scores]
        except Exception as exc:  # Provider and model-loading errors become a controlled failure.
            raise RerankerProviderError("Cross-encoder reranking failed") from exc


class RerankingService:
    """Scores hybrid candidates and deterministically selects bounded final context."""

    def __init__(self, provider: RerankerProvider, config: RerankingConfig) -> None:
        self._provider = provider
        self._config = config

    @property
    def candidate_count(self) -> int:
        return self._config.candidate_count

    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridRetrievalResult],
        final_count: int | None = None,
    ) -> list[RerankedRetrievalResult]:
        if not candidates:
            return []
        limit = final_count if final_count is not None else self._config.final_count
        if limit < 1:
            raise ValueError("final count must be positive")
        selected = list(candidates[: self._config.candidate_count])
        with log_timing(logger, "reranking", candidate_count=len(selected)):
            try:
                scores = self._provider.rerank(
                    query,
                    [candidate.chunk.content for candidate in selected],
                    self._config.model,
                    self._config.batch_size,
                )
                if len(scores) != len(selected):
                    raise RerankerProviderError("Reranker returned a mismatched score count")
            except RerankerProviderError:
                if not self._config.fail_open:
                    raise
                logger.warning("reranker failed; using hybrid ranking as fallback")
                scores = [candidate.fused_score for candidate in selected]

        ranked = sorted(
            zip(selected, scores, strict=True),
            key=lambda item: (-item[1], item[0].rank, item[0].chunk.chunk_id),
        )[:limit]
        return [
            RerankedRetrievalResult(
                rank=rank,
                chunk=candidate.chunk,
                relevance_score=score,
                hybrid_score=candidate.fused_score,
                vector_rank=candidate.vector_rank,
                bm25_rank=candidate.bm25_rank,
                metadata=candidate.metadata,
            )
            for rank, (candidate, score) in enumerate(ranked, start=1)
        ]


def create_reranking_service(settings: Settings) -> RerankingService:
    return RerankingService(
        provider=CrossEncoderRerankerProvider(),
        config=RerankingConfig(
            model=settings.reranker_model,
            candidate_count=settings.reranker_candidate_count,
            final_count=settings.reranker_final_count,
            batch_size=settings.reranker_batch_size,
            fail_open=settings.reranker_fail_open,
        ),
    )
