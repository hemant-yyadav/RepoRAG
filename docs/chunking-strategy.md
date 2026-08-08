# Code-aware chunking strategy

Chunking converts normalized `RepositoryFile` values into deterministic `CodeChunk` values. It has no embedding, vector database, or LLM dependency.

## Structure first

Python is parsed with the standard-library AST. Top-level imports, functions, and classes become chunks with their exact source-line ranges. Nested functions remain inside their parent function, preserving local context. A class that exceeds the configured maximum is represented by its direct methods, each carrying a `ClassName.method_name` symbol; its header, attributes, and other non-method code are retained as class-context chunks.

Markdown is divided on headings. Each heading and its body becomes a `markdown_section` with the heading as its symbol name.

## Fallback behavior

Other languages—and syntactically invalid Python—use a line-aware fallback. It never cuts through a line, observes the maximum character size when possible, and can repeat a configurable number of preceding lines into the next chunk. Small trailing content is merged when it fits the maximum size.

## Configuration

- `MAX_CHUNK_SIZE` (default `4000`): maximum characters per chunk.
- `CHUNK_OVERLAP_LINES` (default `2`): repeated trailing lines in a fallback continuation.
- `MIN_CHUNK_SIZE` (default `200`): trailing fragments below this size are merged if safe.

Each chunk carries its repository identifier, file path, language, source lines, optional symbol, type, and deterministic SHA-256 chunk identifier. This metadata supports later user interfaces such as `src/auth/service.py, lines 42–81`.
