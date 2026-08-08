"""Provider-neutral batched embedding pipeline with retry handling."""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.core.config import Settings
from app.models.chunk import CodeChunk
from app.models.embedding import EmbeddedChunk

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base error raised while creating embeddings."""


class EmbeddingProviderError(EmbeddingError):
    """An embedding provider rejected or could not complete a request."""


class EmbeddingProvider(Protocol):
    """Minimal provider contract; implementations must preserve input ordering."""

    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        """Embed a batch of texts and return one vector per input, in input order."""


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    model: str
    batch_size: int = 32
    max_retries: int = 3
    initial_backoff_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("embedding model must not be blank")
        if self.batch_size < 1:
            raise ValueError("embedding batch size must be positive")
        if self.max_retries < 0:
            raise ValueError("max retries cannot be negative")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial backoff seconds cannot be negative")


class OpenAICompatibleEmbeddingProvider:
    """HTTP implementation for OpenAI-compatible `/embeddings` APIs.

    The provider is isolated behind ``EmbeddingProvider`` so model vendors can be
    changed without affecting batching or downstream code.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        dimensions: int | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("embedding API key is required")
        if dimensions is not None and dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        self._base_url = base_url.rstrip("/")
        self._dimensions = dimensions
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, object] = {"model": model, "input": list(texts)}
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions
        try:
            response = self._client.post(
                f"{self._base_url}/embeddings", headers=self._headers, json=payload
            )
            response.raise_for_status()
            data = response.json()["data"]
            ordered = sorted(data, key=lambda item: item["index"])
            vectors = [item["embedding"] for item in ordered]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise EmbeddingProviderError("Embedding provider request failed") from exc
        if len(vectors) != len(texts) or not all(isinstance(vector, list) for vector in vectors):
            raise EmbeddingProviderError("Embedding provider returned an invalid response")
        return vectors


class EmbeddingService:
    """Batches chunks, retries provider requests, and preserves source ordering."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        config: EmbeddingConfig,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._provider = provider
        self._config = config
        self._sleep = sleep

    def embed_chunks(self, chunks: Sequence[CodeChunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []
        embedded: list[EmbeddedChunk] = []
        for batch_number, batch in enumerate(_batches(chunks, self._config.batch_size), start=1):
            logger.info("embedding batch %d containing %d chunks", batch_number, len(batch))
            vectors = self._embed_with_retry([chunk.content for chunk in batch])
            if len(vectors) != len(batch):
                raise EmbeddingProviderError("Embedding provider returned a mismatched vector count")
            embedded.extend(
                EmbeddedChunk(chunk_id=chunk.chunk_id, vector=vector, chunk=chunk)
                for chunk, vector in zip(batch, vectors, strict=True)
            )
        return embedded

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed arbitrary query text while using the same provider and retry policy."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for batch in _text_batches(texts, self._config.batch_size):
            vectors.extend(self._embed_with_retry(batch))
        return vectors

    def _embed_with_retry(self, texts: Sequence[str]) -> list[list[float]]:
        for attempt in range(self._config.max_retries + 1):
            try:
                return self._provider.embed(texts, self._config.model)
            except EmbeddingProviderError:
                if attempt == self._config.max_retries:
                    raise
                delay = self._config.initial_backoff_seconds * (2**attempt)
                logger.warning("embedding batch failed; retrying in %.2f seconds", delay)
                self._sleep(delay)
        raise AssertionError("unreachable")


def create_embedding_service(settings: Settings) -> EmbeddingService:
    """Build the configured service without exposing provider details to callers."""
    if settings.embedding_provider != "openai_compatible":
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
    if not settings.embedding_api_key:
        raise ValueError("EMBEDDING_API_KEY is required for the configured embedding provider")
    provider = OpenAICompatibleEmbeddingProvider(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        dimensions=settings.embedding_dimensions,
    )
    return EmbeddingService(
        provider=provider,
        config=EmbeddingConfig(
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            max_retries=settings.embedding_max_retries,
            initial_backoff_seconds=settings.embedding_initial_backoff_seconds,
        ),
    )


def _batches(items: Sequence[CodeChunk], batch_size: int) -> list[Sequence[CodeChunk]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _text_batches(items: Sequence[str], batch_size: int) -> list[Sequence[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]
