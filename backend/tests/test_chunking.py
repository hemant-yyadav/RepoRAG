from pathlib import Path

from app.models.repository import RepositoryFile
from app.services.chunking import ChunkingConfig, CodeChunkingService


def make_file(path: str, language: str, content: str) -> RepositoryFile:
    return RepositoryFile(
        path=path,
        language=language,
        content=content,
        size_bytes=len(content.encode("utf-8")),
        line_count=len(content.splitlines()),
    )


def test_python_fixture_preserves_import_class_and_function() -> None:
    content = (Path(__file__).parent / "fixtures/sample_repository/python_module.py").read_text(
        encoding="utf-8"
    )
    chunks = CodeChunkingService().chunk_file("repo-1", make_file("python_module.py", "Python", content))

    assert [(chunk.chunk_type, chunk.symbol_name) for chunk in chunks] == [
        ("import", None),
        ("class", "Greeter"),
        ("function", "add"),
    ]
    assert chunks[1].start_line == 4
    assert chunks[1].end_line == 6
    assert chunks[2].start_line == 9
    assert chunks[2].end_line == 10


def test_nested_python_function_remains_with_its_parent() -> None:
    content = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
    chunks = CodeChunkingService().chunk_file("repo-1", make_file("nested.py", "Python", content))

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "function"
    assert chunks[0].symbol_name == "outer"
    assert "def inner" in chunks[0].content
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 4)


def test_long_python_function_splits_at_lines_with_symbol_and_overlap() -> None:
    content = "def long_function():\n" + "".join(f"    value_{index} = {index}\n" for index in range(12))
    service = CodeChunkingService(ChunkingConfig(max_chunk_size=55, overlap_lines=1, min_chunk_size=0))
    chunks = service.chunk_file("repo-1", make_file("long.py", "Python", content))

    assert len(chunks) > 1
    assert all(chunk.chunk_type == "function" for chunk in chunks)
    assert all(chunk.symbol_name == "long_function" for chunk in chunks)
    assert all(len(chunk.content) <= 55 for chunk in chunks)
    assert chunks[1].start_line <= chunks[0].end_line


def test_markdown_sections_have_titles_and_accurate_lines() -> None:
    content = (Path(__file__).parent / "fixtures/sample_repository/README.md").read_text(encoding="utf-8")
    chunks = CodeChunkingService().chunk_file("repo-1", make_file("README.md", "Markdown", content))

    assert [chunk.symbol_name for chunk in chunks] == ["Sample repository", "Installation", "Usage"]
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 4), (5, 8), (9, 11)]
    assert all(chunk.chunk_type == "markdown_section" for chunk in chunks)


def test_unsupported_language_uses_line_aware_fallback() -> None:
    content = "a\nb\nc\nd\n"
    service = CodeChunkingService(ChunkingConfig(max_chunk_size=5, overlap_lines=0, min_chunk_size=0))
    chunks = service.chunk_file("repo-1", make_file("notes.txt", "Plain text", content))

    assert [chunk.chunk_type for chunk in chunks] == ["fallback", "fallback"]
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 2), (3, 4)]


def test_chunk_ids_are_deterministic_and_include_repository_identity() -> None:
    file = make_file("module.js", "JavaScript", "const answer = 42;\n")
    service = CodeChunkingService()

    first = service.chunk_file("repo-1", file)[0]
    second = service.chunk_file("repo-1", file)[0]
    other_repository = service.chunk_file("repo-2", file)[0]

    assert first.chunk_id == second.chunk_id
    assert first.chunk_id != other_repository.chunk_id
    assert first.file_path == "module.js"
