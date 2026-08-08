"""Fusion of semantic and lexical repository-scoped retrieval results."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.core.config import Settings
from app.core.observability import log_timing
import logging
from app.models.hybrid_retrieval import HybridRetrievalResult
from app.services.lexical import BM25Index
from app.services.retrieval import QueryEmbedder, VectorSearcher, RetrievalService
from app.services.embedding import create_embedding_service
from app.services.qdrant_store import QdrantStore
from app.services.lexical import get_lexical_index

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HybridRetrievalConfig:
    top_k: int = 5
    candidate_pool_size: int = 20
    rrf_k: int = 60
    vector_weight: float = 1.0
    bm25_weight: float = 1.0
    score_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.candidate_pool_size < 1:
            raise ValueError("top_k and candidate pool size must be positive")
        if self.rrf_k < 1:
            raise ValueError("RRF k must be positive")
        if self.vector_weight < 0 or self.bm25_weight < 0:
            raise ValueError("retrieval weights cannot be negative")
        if self.vector_weight == 0 and self.bm25_weight == 0:
            raise ValueError("at least one retrieval weight must be positive")


class HybridRetrievalService:
    """Queries both indexes and merges them with weighted Reciprocal Rank Fusion."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        vector_store: VectorSearcher,
        lexical_index: BM25Index,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._config = config or HybridRetrievalConfig()

    def retrieve(
        self,
        repository_id: str,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filters: Mapping[str, str | int | bool] | None = None,
    ) -> list[HybridRetrievalResult]:
        if not repository_id.strip() or not query.strip():
            raise ValueError("repository_id and query must not be blank")
        requested_top_k = top_k or self._config.top_k
        if requested_top_k < 1:
            raise ValueError("top_k must be positive")
        with log_timing(logger, "hybrid_retrieval", repository_id=repository_id):
            vectors = self._embedder.embed_texts([query])
            if len(vectors) != 1:
                raise ValueError("query embedding provider returned an invalid vector count")
            semantic = self._vector_store.similarity_search(
                repository_id, vectors[0], self._config.candidate_pool_size,
                score_threshold if score_threshold is not None else self._config.score_threshold, metadata_filters,
            )
            lexical = self._lexical_index.search(repository_id, query, self._config.candidate_pool_size, metadata_filters)
        return fuse_ranked_results(semantic, lexical, self._config, requested_top_k)


def fuse_ranked_results(
    semantic_results: Sequence[object], lexical_results: Sequence[object], config: HybridRetrievalConfig, top_k: int
) -> list[HybridRetrievalResult]:
    """Apply weighted RRF while retaining rank provenance from both result lists."""
    candidates: dict[str, dict[str, object]] = {}
    for rank, stored in enumerate(semantic_results, start=1):
        retrieval = RetrievalService._to_result(stored)  # type: ignore[arg-type]
        candidate = candidates.setdefault(
            retrieval.chunk.chunk_id,
            {"chunk": retrieval.chunk, "vector_rank": None, "bm25_rank": None, "score": 0.0},
        )
        if candidate["vector_rank"] is None:
            candidate["vector_rank"] = rank
            candidate["score"] = float(candidate["score"]) + config.vector_weight / (config.rrf_k + rank)
    for rank, lexical in enumerate(lexical_results, start=1):
        candidate = candidates.setdefault(
            lexical.chunk.chunk_id,  # type: ignore[attr-defined]
            {"chunk": lexical.chunk, "vector_rank": None, "bm25_rank": None, "score": 0.0},  # type: ignore[attr-defined]
        )
        if candidate["bm25_rank"] is None:
            candidate["bm25_rank"] = rank
            candidate["score"] = float(candidate["score"]) + config.bm25_weight / (config.rrf_k + rank)
    ordered = sorted(
        candidates.values(), key=lambda candidate: (-float(candidate["score"]), candidate["chunk"].chunk_id)  # type: ignore[union-attr]
    )[:top_k]
    return [
        HybridRetrievalResult(
            rank=rank,
            chunk=candidate["chunk"],  # type: ignore[arg-type]
            fused_score=float(candidate["score"]),
            vector_rank=candidate["vector_rank"],  # type: ignore[arg-type]
            bm25_rank=candidate["bm25_rank"],  # type: ignore[arg-type]
            metadata=candidate["chunk"].metadata,  # type: ignore[union-attr]
        )
        for rank, candidate in enumerate(ordered, start=1)
    ]


def create_hybrid_retrieval_service(settings: Settings) -> HybridRetrievalService:
    return HybridRetrievalService(
        embedder=create_embedding_service(settings),
        vector_store=QdrantStore.from_settings(
            settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection_name,
            settings.qdrant_max_retries, settings.qdrant_initial_backoff_seconds,
        ),
        lexical_index=get_lexical_index(),
        config=HybridRetrievalConfig(
            top_k=settings.retrieval_top_k,
            candidate_pool_size=settings.retrieval_candidate_pool_size,
            rrf_k=settings.retrieval_rrf_k,
            vector_weight=settings.retrieval_vector_weight,
            bm25_weight=settings.retrieval_bm25_weight,
            score_threshold=settings.retrieval_score_threshold,
        ),
    )
