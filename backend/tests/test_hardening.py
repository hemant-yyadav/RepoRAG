from pathlib import Path

import pytest

from app.core.resilience import retry_operation
from app.models.repository import RepositoryFile
from app.services.code_intelligence import validate_indexed_path
from app.services.repository_ingestion import RepositorySizeLimitError, iter_repository_files, should_ignore_path
from app.services.repository_lifecycle import RepositoryLifecycleService, RepositoryRegistry


def test_repository_size_limit_stops_large_collections(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("a" * 12, encoding="utf-8")
    (tmp_path / "two.py").write_text("b" * 12, encoding="utf-8")

    with pytest.raises(RepositorySizeLimitError):
        list(iter_repository_files(tmp_path, max_file_size_bytes=100, max_repository_size_bytes=20))


def test_generated_files_are_filtered(tmp_path: Path) -> None:
    generated = tmp_path / "client.generated.ts"
    generated.write_text("export const generated = true", encoding="utf-8")

    assert should_ignore_path(generated, tmp_path)
    assert list(iter_repository_files(tmp_path, 1_000)) == []


@pytest.mark.parametrize("value", ["../secret.py", "/etc/passwd", "src\\auth.py", "./src/auth.py"])
def test_indexed_path_rejects_traversal_and_host_paths(value: str) -> None:
    with pytest.raises(ValueError):
        validate_indexed_path(value)


def test_retry_operation_is_bounded() -> None:
    attempts: list[int] = []
    delays: list[float] = []

    def fail() -> None:
        attempts.append(1)
        raise ConnectionError("temporary")

    with pytest.raises(ConnectionError):
        retry_operation("test", fail, (ConnectionError,), 2, 0.1, __import__("logging").getLogger(__name__), delays.append)

    assert len(attempts) == 3
    assert delays == [0.1, 0.2]


def test_unchanged_reindex_skips_expensive_indexing() -> None:
    class Ingestion:
        def ingest(self, url):
            return type("Result", (), {
                "files": [RepositoryFile("src/a.py", "Python", "pass", 4, 1)],
                "total_size_bytes": 4,
                "languages": ["Python"],
            })()

    class Indexer:
        calls = 0
        def index_repository(self, repository_id, files):
            self.calls += 1
            return type("Indexed", (), {"chunk_count": 1})()
        def delete_repository(self, repository_id):
            return None

    indexer = Indexer()
    lifecycle = RepositoryLifecycleService(RepositoryRegistry(), lambda: Ingestion(), lambda: indexer)
    record = lifecycle.submit_indexing("https://github.com/acme/demo")
    lifecycle.run_indexing(record.repository_id)
    lifecycle.submit_indexing("https://github.com/acme/demo")
    lifecycle.run_indexing(record.repository_id)

    assert indexer.calls == 1
    assert lifecycle.get(record.repository_id).status == "ready"
