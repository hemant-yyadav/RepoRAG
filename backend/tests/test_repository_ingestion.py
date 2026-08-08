from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.models.repository import RepositoryFile
from app.services.repository_ingestion import (
    IngestionResult,
    InvalidRepositoryUrlError,
    detect_language,
    iter_repository_files,
    parse_github_repository_url,
    should_ignore_path,
)


def test_valid_github_url_is_normalized() -> None:
    parsed = parse_github_repository_url("https://github.com/openai/example.git")

    assert parsed.canonical_url == "https://github.com/openai/example.git"
    assert parsed.name == "example"


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/openai/example",
        "https://gitlab.com/openai/example",
        "https://github.com/openai/example/tree/main",
        "https://github.com/openai/example?tab=readme",
    ],
)
def test_invalid_github_urls_are_rejected(url: str) -> None:
    with pytest.raises(InvalidRepositoryUrlError):
        parse_github_repository_url(url)


@pytest.mark.parametrize(
    ("filename", "language"),
    [("main.py", "Python"), ("client.tsx", "TypeScript"), ("README.md", "Markdown")],
)
def test_language_detection_is_extension_based(filename: str, language: str) -> None:
    assert detect_language(Path(filename)) == language


def test_unsupported_file_types_are_not_ingested(tmp_path: Path) -> None:
    image = tmp_path / "logo.png"
    image.write_bytes(b"not-a-source-file")

    assert detect_language(image) is None
    assert list(iter_repository_files(tmp_path, max_file_size_bytes=1_000)) == []


def test_ignored_directories_and_files_are_excluded(tmp_path: Path) -> None:
    ignored_directory = tmp_path / "node_modules" / "package.js"
    ignored_directory.parent.mkdir()
    ignored_directory.write_text("export default {};", encoding="utf-8")
    environment_file = tmp_path / ".env.local"
    environment_file.write_text("SECRET=value", encoding="utf-8")
    minified_file = tmp_path / "bundle.min.js"
    minified_file.write_text("const x=1", encoding="utf-8")

    assert should_ignore_path(ignored_directory, tmp_path)
    assert should_ignore_path(environment_file, tmp_path)
    assert should_ignore_path(minified_file, tmp_path)


def test_file_size_and_binary_protection(tmp_path: Path) -> None:
    (tmp_path / "large.py").write_text("x" * 21, encoding="utf-8")
    (tmp_path / "binary.py").write_bytes(b"text\x00binary")
    (tmp_path / "small.py").write_text("print('ok')\n", encoding="utf-8")

    files = list(iter_repository_files(tmp_path, max_file_size_bytes=20))

    assert [file.path for file in files] == ["small.py"]


def test_normalized_repository_file_fields(tmp_path: Path) -> None:
    source = tmp_path / "src" / "main.py"
    source.parent.mkdir()
    source.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")

    file = next(iter_repository_files(tmp_path, max_file_size_bytes=1_000))

    assert isinstance(file, RepositoryFile)
    assert file.path == "src/main.py"
    assert file.language == "Python"
    assert file.content == "def greet():\n    return 'hello'\n"
    assert file.size_bytes == len(file.content.encode("utf-8"))
    assert file.line_count == 2


def test_ingestion_endpoint_returns_metadata_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import app
    from app.services.repository_ingestion import RepositoryIngestionService

    result = IngestionResult(
        repository_url="https://github.com/openai/example",
        repository_name="example",
        files=[RepositoryFile("main.py", "Python", "print('private')", 16, 1)],
    )
    monkeypatch.setattr(RepositoryIngestionService, "ingest", lambda self, _: result)

    response = TestClient(app).post(
        "/repositories/ingest", json={"repository_url": "https://github.com/openai/example"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "repository_url": "https://github.com/openai/example",
        "repository_name": "example",
        "file_count": 1,
        "total_size_bytes": 16,
        "languages": ["Python"],
    }
