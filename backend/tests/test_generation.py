from dataclasses import dataclass, field

import pytest

from app.models.chunk import CodeChunk
from app.models.retrieval import RetrievalResult
from app.services.generation import (
    GenerationConfig,
    GenerationProviderError,
    GenerationService,
    GeminiProvider,
    build_context,
    build_prompt,
    format_retrieval_context,
)


def retrieved(content: str = "def authenticate():\n    return token\n", path: str = "src/auth.py") -> RetrievalResult:
    chunk = CodeChunk(
        chunk_id="chunk-1",
        repository_id="repo-1",
        file_path=path,
        language="Python",
        content=content,
        start_line=42,
        end_line=43,
        symbol_name="authenticate",
        chunk_type="function",
    )
    return RetrievalResult(chunk=chunk, score=0.92, metadata={})


@dataclass
class FakeGemini:
    answer: str = "JWT authentication is handled by authenticate."
    error: Exception | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def generate(self, prompt: str, model: str) -> str:
        self.calls.append((prompt, model))
        if self.error:
            raise self.error
        return self.answer


def test_context_formatting_includes_only_grounding_fields() -> None:
    formatted = format_retrieval_context(retrieved())

    assert "[1]" in formatted
    assert "File: src/auth.py" in formatted
    assert "Lines: 42-43" in formatted
    assert "Symbol: authenticate" in formatted
    assert "Language: Python" in formatted
    assert "def authenticate" in formatted
    assert "score" not in formatted


def test_prompt_construction_combines_question_and_context() -> None:
    prompt = build_prompt("Where is JWT auth?", "file_path: src/auth.py")

    assert "Where is JWT auth?" in prompt
    assert "file_path: src/auth.py" in prompt
    assert "using only the retrieved context" in prompt


def test_empty_retrieval_returns_explicit_insufficiency_without_gemini_call() -> None:
    provider = FakeGemini()
    service = GenerationService(provider, GenerationConfig(model="configured-model"))

    answer = service.generate("Where is JWT auth?", [])

    assert "insufficient" in answer.answer
    assert answer.used_chunk_count == 0
    assert provider.calls == []


def test_successful_generation_uses_configured_model() -> None:
    provider = FakeGemini()
    service = GenerationService(provider, GenerationConfig(model="configured-model"))

    answer = service.generate("Where is JWT auth?", [retrieved()])

    assert answer.answer == "JWT authentication is handled by authenticate."
    assert answer.used_chunk_count == 1
    assert provider.calls[0][1] == "configured-model"


def test_gemini_failure_is_propagated() -> None:
    provider = FakeGemini(error=GenerationProviderError("service unavailable"))
    service = GenerationService(provider, GenerationConfig(model="configured-model"))

    with pytest.raises(GenerationProviderError):
        service.generate("Where is JWT auth?", [retrieved()])


def test_context_limit_skips_excessively_large_chunks() -> None:
    small = retrieved(content="small\n", path="small.py")
    oversized = retrieved(content="x" * 1_000, path="large.py")

    context, count = build_context([small, oversized], max_context_chars=200, max_context_chunks=12)

    assert count == 1
    assert "small.py" in context
    assert "large.py" not in context


@pytest.mark.parametrize(
    "config",
    [
        {"model": ""},
        {"model": "test", "max_context_chars": 0},
        {"model": "test", "max_context_chunks": 0},
    ],
)
def test_generation_configuration_validation(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GenerationConfig(**config)  # type: ignore[arg-type]


def test_gemini_provider_requires_api_key() -> None:
    with pytest.raises(ValueError):
        GeminiProvider(api_key="")
