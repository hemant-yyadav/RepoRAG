"""Grounded answer generation from retrieved chunks, isolated from retrieval itself."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.core.config import Settings
from app.models.generation import GeneratedAnswer
from app.models.citation import SourceCitation
from app.models.retrieval import RetrievalResult

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """You answer questions about a code repository using only the retrieved context.
Never invent files, functions, implementation details, or behavior not present in that context.
Clearly label inferences as inferences. If the context is insufficient, say so plainly.
Prefer precise code-level explanations. Do not claim to have inspected code that was not provided.
For every factual repository claim, cite the supplied source identifier in square brackets, such as [1].
Only use source identifiers that appear in the retrieved context."""


class GenerationProviderError(Exception):
    """A generation provider request could not be completed."""


class LLMProvider(Protocol):
    """Small provider contract that keeps generation vendor-neutral."""

    def generate(self, prompt: str, model: str) -> str:
        """Return generated plain text for a fully constructed prompt."""


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    model: str
    max_context_chars: int = 20_000
    max_context_chunks: int = 12

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Gemini model must not be blank")
        if self.max_context_chars < 1:
            raise ValueError("maximum context characters must be positive")
        if self.max_context_chunks < 1:
            raise ValueError("maximum context chunks must be positive")


@dataclass(frozen=True, slots=True)
class CitedContext:
    content: str
    citations: list[SourceCitation]


class GeminiProvider:
    """Gemini REST API adapter, replaceable through the LLMProvider protocol."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def generate(self, prompt: str, model: str) -> str:
        if not model.strip():
            raise ValueError("Gemini model must not be blank")
        model_name = model.removeprefix("models/")
        try:
            response = self._client.post(
                f"{self._base_url}/models/{model_name}:generateContent",
                params={"key": self._api_key},
                json={
                    "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTIONS}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                },
            )
            response.raise_for_status()
            candidates = response.json()["candidates"]
            parts = candidates[0]["content"]["parts"]
            answer = "".join(part.get("text", "") for part in parts).strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise GenerationProviderError("Gemini generation request failed") from exc
        if not answer:
            raise GenerationProviderError("Gemini returned an empty answer")
        return answer


class GenerationService:
    """Builds bounded, structured context and delegates answer generation to an LLM."""

    def __init__(self, provider: LLMProvider, config: GenerationConfig) -> None:
        self._provider = provider
        self._config = config

    def generate(self, question: str, retrieved_chunks: Sequence[RetrievalResult]) -> GeneratedAnswer:
        if not question.strip():
            raise ValueError("question must not be blank")
        cited_context = build_cited_context(
            retrieved_chunks,
            max_context_chars=self._config.max_context_chars,
            max_context_chunks=self._config.max_context_chunks,
        )
        if not cited_context.content:
            return GeneratedAnswer(
                answer="The retrieved repository context is insufficient to answer this question.",
                used_chunk_count=0,
                context_char_count=0,
            )
        prompt = build_prompt(question, cited_context.content)
        logger.info("generating answer from %d retrieved chunks", len(cited_context.citations))
        answer = self._provider.generate(prompt, self._config.model)
        answer, sources = validate_answer_citations(answer, cited_context.citations)
        return GeneratedAnswer(
            answer=answer,
            used_chunk_count=len(cited_context.citations),
            context_char_count=len(cited_context.content),
            sources=sources,
        )


def format_retrieval_context(result: RetrievalResult, citation_id: str = "1") -> str:
    """Format only source-location fields and code content needed for grounding."""
    chunk = result.chunk
    symbol = chunk.symbol_name or "<module>"
    return (
        f"[{citation_id}]\n"
        f"File: {chunk.file_path}\n"
        f"Lines: {chunk.start_line}-{chunk.end_line}\n"
        f"Symbol: {symbol}\n"
        f"Language: {chunk.language}\n"
        "content:\n"
        f"{chunk.content}\n"
    )


def build_context(
    retrieved_chunks: Sequence[RetrievalResult], max_context_chars: int, max_context_chunks: int
) -> tuple[str, int]:
    """Compatibility helper returning only the formatted context and its chunk count."""
    cited_context = build_cited_context(retrieved_chunks, max_context_chars, max_context_chunks)
    return cited_context.content, len(cited_context.citations)


def build_cited_context(
    retrieved_chunks: Sequence[RetrievalResult], max_context_chars: int, max_context_chunks: int
) -> CitedContext:
    """Assign stable IDs in retrieval order and retain only complete bounded chunks."""
    context_blocks: list[str] = []
    citations: list[SourceCitation] = []
    context_size = 0
    for result in retrieved_chunks[:max_context_chunks]:
        citation_id = str(len(citations) + 1)
        block = format_retrieval_context(result, citation_id)
        if context_size + len(block) > max_context_chars:
            continue
        context_blocks.append(block)
        chunk = result.chunk
        citations.append(SourceCitation(
            citation_id=citation_id,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            symbol_name=chunk.symbol_name,
            chunk_id=chunk.chunk_id,
        ))
        context_size += len(block)
    return CitedContext(content="\n".join(context_blocks), citations=citations)


def build_prompt(question: str, context: str) -> str:
    return f"""Repository question:
{question}

Retrieved repository context:
{context}

Answer the repository question using only the retrieved context above."""


def validate_answer_citations(
    answer: str, available_citations: Sequence[SourceCitation]
) -> tuple[str, list[SourceCitation]]:
    """Remove unknown citation markers and return valid source metadata in answer order."""
    citations_by_id = {citation.citation_id: citation for citation in available_citations}
    used_ids: list[str] = []

    def replace(match: re.Match[str]) -> str:
        citation_id = match.group(1)
        if citation_id not in citations_by_id:
            return ""
        if citation_id not in used_ids:
            used_ids.append(citation_id)
        return match.group(0)

    safe_answer = re.sub(r"\[(\d+)\]", replace, answer)
    return safe_answer, [citations_by_id[citation_id] for citation_id in used_ids]


def create_generation_service(settings: Settings) -> GenerationService:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required")
    if not settings.gemini_model:
        raise ValueError("GEMINI_MODEL is required")
    return GenerationService(
        provider=GeminiProvider(api_key=settings.gemini_api_key),
        config=GenerationConfig(
            model=settings.gemini_model,
            max_context_chars=settings.generation_max_context_chars,
            max_context_chunks=settings.generation_max_context_chunks,
        ),
    )
