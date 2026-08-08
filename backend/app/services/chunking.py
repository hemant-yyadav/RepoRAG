"""Semantic, deterministic source and document chunking without embedding concerns."""

import ast
import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass

from app.models.chunk import CodeChunk
from app.models.repository import RepositoryFile
from app.core.observability import log_timing

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Controls the line-aware fallback used when a structure exceeds the size limit."""

    max_chunk_size: int = 4_000
    overlap_lines: int = 2
    min_chunk_size: int = 200

    def __post_init__(self) -> None:
        if self.max_chunk_size < 1:
            raise ValueError("max_chunk_size must be positive")
        if self.overlap_lines < 0:
            raise ValueError("overlap_lines cannot be negative")
        if self.min_chunk_size < 0:
            raise ValueError("min_chunk_size cannot be negative")


class CodeChunkingService:
    """Produces semantic chunks from normalized repository files."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_files(self, repository_id: str, files: Iterable[RepositoryFile]) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        with log_timing(logger, "chunking", repository_id=repository_id):
            for repository_file in files:
                chunks.extend(self.chunk_file(repository_id, repository_file))
        return chunks

    def chunk_file(self, repository_id: str, repository_file: RepositoryFile) -> list[CodeChunk]:
        if repository_file.language == "Python":
            return self._chunk_python(repository_id, repository_file)
        if repository_file.language == "Markdown":
            return self._chunk_markdown(repository_id, repository_file)
        return self._chunk_text(repository_id, repository_file, "fallback")

    def _chunk_python(self, repository_id: str, repository_file: RepositoryFile) -> list[CodeChunk]:
        try:
            module = ast.parse(repository_file.content)
        except SyntaxError:
            return self._chunk_text(repository_id, repository_file, "fallback")

        lines = repository_file.content.splitlines(keepends=True)
        chunks: list[CodeChunk] = []
        covered_lines: set[int] = set()
        for node in module.body:
            if not hasattr(node, "end_lineno"):
                continue
            start_line, end_line = _node_start_line(node), node.end_lineno
            covered_lines.update(range(start_line, end_line + 1))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                chunks.extend(self._make_structure_chunks(
                    repository_id, repository_file, lines, start_line, end_line, None, "import"
                ))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.extend(self._make_structure_chunks(
                    repository_id, repository_file, lines, start_line, end_line, node.name, "function"
                ))
            elif isinstance(node, ast.ClassDef):
                chunks.extend(self._chunk_class(
                    repository_id, repository_file, lines, node, start_line, end_line
                ))

        # Keep unstructured top-level statements (constants, decorators, etc.) rather than losing them.
        uncovered = [line for line in range(1, len(lines) + 1) if line not in covered_lines]
        for start_line, end_line in _contiguous_ranges(uncovered):
            content = "".join(lines[start_line - 1 : end_line])
            if content.strip():
                chunks.extend(self._split_and_make(
                    repository_id, repository_file, content, start_line, None, "module"
                ))
        return sorted(chunks, key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.chunk_id))

    def _chunk_class(
        self,
        repository_id: str,
        repository_file: RepositoryFile,
        lines: list[str],
        node: ast.ClassDef,
        start_line: int,
        end_line: int,
    ) -> list[CodeChunk]:
        content = "".join(lines[start_line - 1 : end_line])
        if len(content) <= self.config.max_chunk_size:
            return [self._make_chunk(
                repository_id, repository_file, content, start_line, end_line, node.name, "class"
            )]

        chunks: list[CodeChunk] = []
        cursor = start_line
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_start = _node_start_line(child)
                if cursor < method_start:
                    context = "".join(lines[cursor - 1 : method_start - 1])
                    if context.strip():
                        chunks.extend(self._split_and_make(
                            repository_id, repository_file, context, cursor, node.name, "class"
                        ))
                chunks.extend(self._make_structure_chunks(
                    repository_id,
                    repository_file,
                    lines,
                    method_start,
                    child.end_lineno,
                    f"{node.name}.{child.name}",
                    "method",
                ))
                cursor = child.end_lineno + 1
        if cursor <= end_line:
            context = "".join(lines[cursor - 1 : end_line])
            if context.strip():
                chunks.extend(self._split_and_make(
                    repository_id, repository_file, context, cursor, node.name, "class"
                ))
        if not chunks:
            chunks.extend(self._split_and_make(
                repository_id, repository_file, content, start_line, node.name, "class"
            ))
        return chunks

    def _chunk_markdown(self, repository_id: str, repository_file: RepositoryFile) -> list[CodeChunk]:
        lines = repository_file.content.splitlines(keepends=True)
        heading_lines = [index + 1 for index, line in enumerate(lines) if line.lstrip().startswith("#")]
        if not heading_lines:
            return self._chunk_text(repository_id, repository_file, "fallback")

        chunks: list[CodeChunk] = []
        if heading_lines[0] > 1:
            prefix = "".join(lines[: heading_lines[0] - 1])
            if prefix.strip():
                chunks.extend(self._split_and_make(
                    repository_id, repository_file, prefix, 1, None, "fallback"
                ))
        for index, start_line in enumerate(heading_lines):
            end_line = heading_lines[index + 1] - 1 if index + 1 < len(heading_lines) else len(lines)
            content = "".join(lines[start_line - 1 : end_line])
            title = lines[start_line - 1].lstrip().lstrip("#").strip()
            chunks.extend(self._split_and_make(
                repository_id, repository_file, content, start_line, title or None, "markdown_section"
            ))
        return chunks

    def _chunk_text(
        self, repository_id: str, repository_file: RepositoryFile, chunk_type: str
    ) -> list[CodeChunk]:
        return self._split_and_make(
            repository_id, repository_file, repository_file.content, 1, None, chunk_type
        )

    def _make_structure_chunks(
        self,
        repository_id: str,
        repository_file: RepositoryFile,
        lines: list[str],
        start_line: int,
        end_line: int,
        symbol_name: str | None,
        chunk_type: str,
    ) -> list[CodeChunk]:
        content = "".join(lines[start_line - 1 : end_line])
        return self._split_and_make(
            repository_id, repository_file, content, start_line, symbol_name, chunk_type
        )

    def _split_and_make(
        self,
        repository_id: str,
        repository_file: RepositoryFile,
        content: str,
        start_line: int,
        symbol_name: str | None,
        chunk_type: str,
    ) -> list[CodeChunk]:
        pieces = _split_by_lines(content, start_line, self.config)
        return [self._make_chunk(
            repository_id, repository_file, piece_content, piece_start, piece_end,
            symbol_name, chunk_type, part=index + 1, parts=len(pieces)
        ) for index, (piece_content, piece_start, piece_end) in enumerate(pieces)]

    def _make_chunk(
        self,
        repository_id: str,
        repository_file: RepositoryFile,
        content: str,
        start_line: int,
        end_line: int,
        symbol_name: str | None,
        chunk_type: str,
        part: int = 1,
        parts: int = 1,
    ) -> CodeChunk:
        identity = "\x1f".join([
            repository_id, repository_file.path, str(start_line), str(end_line),
            symbol_name or "", chunk_type, content,
        ])
        return CodeChunk(
            chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            repository_id=repository_id,
            file_path=repository_file.path,
            language=repository_file.language,
            content=content,
            start_line=start_line,
            end_line=end_line,
            symbol_name=symbol_name,
            chunk_type=chunk_type,
            metadata={"part": part, "parts": parts},
        )


def _split_by_lines(content: str, start_line: int, config: ChunkingConfig) -> list[tuple[str, int, int]]:
    """Split only at lines, retaining a configurable number of prior lines as overlap."""
    lines = content.splitlines(keepends=True)
    if not lines:
        return []
    pieces: list[tuple[str, int, int]] = []
    index = 0
    while index < len(lines):
        end = index
        size = 0
        while end < len(lines) and (size + len(lines[end]) <= config.max_chunk_size or end == index):
            size += len(lines[end])
            end += 1
        piece = "".join(lines[index:end])
        # A small trailing fragment is merged when it fits, avoiding meaningless chunks.
        if end < len(lines):
            remaining = "".join(lines[end:])
            if len(remaining) < config.min_chunk_size and len(piece) + len(remaining) <= config.max_chunk_size:
                piece += remaining
                end = len(lines)
        pieces.append((piece, start_line + index, start_line + end - 1))
        if end == len(lines):
            break
        next_index = max(index + 1, end - config.overlap_lines)
        index = next_index
    return pieces


def _contiguous_ranges(lines: list[int]) -> Iterable[tuple[int, int]]:
    if not lines:
        return
    start = previous = lines[0]
    for line in lines[1:]:
        if line != previous + 1:
            yield start, previous
            start = line
        previous = line
    yield start, previous


def _node_start_line(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(decorator.lineno for decorator in decorators)])
