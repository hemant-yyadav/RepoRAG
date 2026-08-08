"""Repository-scoped BM25 index with source-code-aware tokenization."""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from app.models.chunk import CodeChunk

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
CAMEL_CASE_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclass(frozen=True, slots=True)
class LexicalSearchResult:
    chunk: CodeChunk
    score: float


@dataclass(frozen=True, slots=True)
class _IndexedDocument:
    chunk: CodeChunk
    term_frequencies: Counter[str]
    length: int


class BM25Index:
    """Small BM25 implementation kept in memory and isolated by repository ID."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")
        self._k1 = k1
        self._b = b
        self._documents: dict[str, list[_IndexedDocument]] = {}
        self._document_frequencies: dict[str, Counter[str]] = {}
        self._average_lengths: dict[str, float] = {}

    def index_chunks(self, repository_id: str, chunks: Iterable[CodeChunk]) -> None:
        """Replace all lexical documents for one repository, making re-indexing idempotent."""
        documents: list[_IndexedDocument] = []
        document_frequencies: Counter[str] = Counter()
        for chunk in chunks:
            tokens = tokenize_code(_searchable_text(chunk))
            frequencies = Counter(tokens)
            documents.append(_IndexedDocument(chunk=chunk, term_frequencies=frequencies, length=len(tokens)))
            document_frequencies.update(frequencies.keys())
        self._documents[repository_id] = documents
        self._document_frequencies[repository_id] = document_frequencies
        self._average_lengths[repository_id] = (
            sum(document.length for document in documents) / len(documents) if documents else 0.0
        )

    def delete_repository(self, repository_id: str) -> None:
        self._documents.pop(repository_id, None)
        self._document_frequencies.pop(repository_id, None)
        self._average_lengths.pop(repository_id, None)

    def search(self, repository_id: str, query: str, limit: int) -> list[LexicalSearchResult]:
        if limit < 1:
            raise ValueError("lexical search limit must be positive")
        documents = self._documents.get(repository_id, [])
        if not documents:
            return []
        query_tokens = tokenize_code(query)
        if not query_tokens:
            return []
        document_frequencies = self._document_frequencies[repository_id]
        average_length = self._average_lengths[repository_id] or 1.0
        document_count = len(documents)
        results: list[LexicalSearchResult] = []
        for document in documents:
            score = 0.0
            for token in set(query_tokens):
                frequency = document.term_frequencies.get(token, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log(1 + (document_count - document_frequencies[token] + 0.5) / (document_frequencies[token] + 0.5))
                denominator = frequency + self._k1 * (1 - self._b + self._b * document.length / average_length)
                score += inverse_frequency * frequency * (self._k1 + 1) / denominator
            if score > 0:
                results.append(LexicalSearchResult(chunk=document.chunk, score=score))
        return sorted(results, key=lambda result: (-result.score, result.chunk.chunk_id))[:limit]


def tokenize_code(text: str) -> list[str]:
    """Keep complete identifiers as well as snake_case and camelCase components."""
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        lowered = token.lower()
        tokens.append(lowered)
        for component in CAMEL_CASE_PATTERN.sub(" ", token).replace("_", " ").split():
            component = component.lower()
            if component != lowered:
                tokens.append(component)
    return tokens


def _searchable_text(chunk: CodeChunk) -> str:
    return "\n".join(filter(None, [chunk.file_path, chunk.symbol_name or "", chunk.content]))


_shared_lexical_index = BM25Index()


def get_lexical_index() -> BM25Index:
    """Return the process-local lexical index shared by indexing and retrieval services."""
    return _shared_lexical_index
