from dataclasses import dataclass, field

import pytest

from app.models.chunk import CodeChunk
from app.services.embedding import (
    EmbeddingConfig,
    EmbeddingProviderError,
    EmbeddingService,
)


def make_chunk(index: int) -> CodeChunk:
    return CodeChunk(
        chunk_id=f"chunk-{index}",
        repository_id="repo-1",
        file_path="src/example.py",
        language="Python",
        content=f"content-{index}",
        start_line=index,
        end_line=index,
        symbol_name=None,
        chunk_type="fallback",
    )


@dataclass
class FakeEmbeddingProvider:
    failures_before_success: int = 0
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.failures_before_success:
            self.failures_before_success -= 1
            raise EmbeddingProviderError("temporary provider failure")
        return [[float(int(text.rsplit("-", 1)[1]))] for text in texts]


def test_embedding_preserves_chunk_mapping_and_order() -> None:
    chunks = [make_chunk(3), make_chunk(1), make_chunk(2)]
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider, EmbeddingConfig(model="test-model"))

    embedded = service.embed_chunks(chunks)

    assert [item.chunk_id for item in embedded] == ["chunk-3", "chunk-1", "chunk-2"]
    assert [item.vector for item in embedded] == [[3.0], [1.0], [2.0]]
    assert [item.chunk for item in embedded] == chunks


def test_embedding_uses_configured_batches() -> None:
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider, EmbeddingConfig(model="test-model", batch_size=2))

    embedded = service.embed_chunks([make_chunk(index) for index in range(5)])

    assert len(embedded) == 5
    assert provider.calls == [
        ["content-0", "content-1"],
        ["content-2", "content-3"],
        ["content-4"],
    ]


def test_empty_input_does_not_call_provider() -> None:
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider, EmbeddingConfig(model="test-model"))

    assert service.embed_chunks([]) == []
    assert provider.calls == []


def test_provider_failure_retries_with_exponential_backoff() -> None:
    delays: list[float] = []
    provider = FakeEmbeddingProvider(failures_before_success=2)
    service = EmbeddingService(
        provider,
        EmbeddingConfig(model="test-model", max_retries=2, initial_backoff_seconds=0.25),
        sleep=delays.append,
    )

    embedded = service.embed_chunks([make_chunk(1)])

    assert embedded[0].vector == [1.0]
    assert len(provider.calls) == 3
    assert delays == [0.25, 0.5]


def test_provider_failure_after_retries_is_raised() -> None:
    provider = FakeEmbeddingProvider(failures_before_success=3)
    service = EmbeddingService(provider, EmbeddingConfig(model="test-model", max_retries=2), sleep=lambda _: None)

    with pytest.raises(EmbeddingProviderError):
        service.embed_chunks([make_chunk(1)])


@pytest.mark.parametrize(
    "config",
    [
        {"model": ""},
        {"model": "test", "batch_size": 0},
        {"model": "test", "max_retries": -1},
        {"model": "test", "initial_backoff_seconds": -0.1},
    ],
)
def test_embedding_configuration_validation(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EmbeddingConfig(**config)  # type: ignore[arg-type]
