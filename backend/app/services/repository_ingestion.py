"""Fetch and normalize public GitHub repositories for later processing phases."""

import logging
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.models.repository import RepositoryFile

logger = logging.getLogger(__name__)

LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "Python",
    ".c": "C",
    ".h": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".rs": "Rust",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".sql": "SQL",
    ".md": "Markdown",
    ".mdx": "Markdown",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
}

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    "target",
    "coverage",
}
IGNORED_FILENAMES = {".env"}


class RepositoryIngestionError(Exception):
    """Base error for an ingestion operation that cannot complete."""


class InvalidRepositoryUrlError(RepositoryIngestionError):
    """Raised when a URL is not a public GitHub repository URL."""


class RepositoryCloneError(RepositoryIngestionError):
    """Raised when Git cannot obtain a repository."""


@dataclass(frozen=True, slots=True)
class ParsedRepositoryUrl:
    canonical_url: str
    name: str


@dataclass(frozen=True, slots=True)
class IngestionResult:
    repository_url: str
    repository_name: str
    files: list[RepositoryFile]

    @property
    def total_size_bytes(self) -> int:
        return sum(file.size_bytes for file in self.files)

    @property
    def languages(self) -> list[str]:
        return sorted({file.language for file in self.files})


def parse_github_repository_url(repository_url: str) -> ParsedRepositoryUrl:
    """Validate a canonical HTTPS GitHub repository URL and derive its identity."""
    parsed = urlparse(repository_url.strip())
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise InvalidRepositoryUrlError("Repository URL must use https://github.com/owner/repository")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parsed.params or parsed.query or parsed.fragment:
        raise InvalidRepositoryUrlError("Repository URL must contain only an owner and repository name")

    owner, repository = parts
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository:
        raise InvalidRepositoryUrlError("Repository URL must include an owner and repository name")

    return ParsedRepositoryUrl(
        canonical_url=f"https://github.com/{owner}/{repository}.git",
        name=repository,
    )


def detect_language(path: Path) -> str | None:
    """Return the language mapped from a file extension, if supported."""
    return LANGUAGE_BY_EXTENSION.get(path.suffix.lower())


def should_ignore_path(path: Path, root: Path) -> bool:
    """Decide whether a path is excluded before its contents are opened."""
    relative_parts = path.relative_to(root).parts
    if any(part in IGNORED_DIRECTORIES for part in relative_parts[:-1]):
        return True
    filename = path.name.lower()
    return filename in IGNORED_FILENAMES or filename.startswith(".env.") or ".min." in filename


def iter_repository_files(root: Path, max_file_size_bytes: int) -> Iterable[RepositoryFile]:
    """Walk a checkout and yield only supported, bounded, non-binary source files."""
    for path in root.rglob("*"):
        if not path.is_file() or should_ignore_path(path, root):
            continue
        language = detect_language(path)
        if language is None:
            continue
        size_bytes = path.stat().st_size
        if size_bytes > max_file_size_bytes:
            logger.info("skipping oversized file: %s", path)
            continue
        raw_content = path.read_bytes()
        if b"\x00" in raw_content:
            logger.info("skipping binary file: %s", path)
            continue
        content = raw_content.decode("utf-8", errors="replace")
        yield RepositoryFile(
            path=path.relative_to(root).as_posix(),
            language=language,
            content=content,
            size_bytes=size_bytes,
            line_count=len(content.splitlines()),
        )


class RepositoryIngestionService:
    """Clones a public repository to a temporary location and normalizes its files."""

    def __init__(self, max_file_size_bytes: int, clone_timeout_seconds: int) -> None:
        self.max_file_size_bytes = max_file_size_bytes
        self.clone_timeout_seconds = clone_timeout_seconds

    def ingest(self, repository_url: str) -> IngestionResult:
        repository = parse_github_repository_url(repository_url)
        with tempfile.TemporaryDirectory(prefix="codebase-rag-") as temporary_directory:
            checkout_path = Path(temporary_directory) / "repository"
            self._clone(repository.canonical_url, checkout_path)
            files = list(iter_repository_files(checkout_path, self.max_file_size_bytes))

        logger.info("ingested repository %s with %d files", repository.name, len(files))
        return IngestionResult(
            repository_url=repository.canonical_url.removesuffix(".git"),
            repository_name=repository.name,
            files=files,
        )

    def _clone(self, repository_url: str, checkout_path: Path) -> None:
        if shutil.which("git") is None:
            raise RepositoryCloneError("Git is required to ingest repositories")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--no-tags", repository_url, str(checkout_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.clone_timeout_seconds,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.warning("repository clone failed for %s", repository_url)
            raise RepositoryCloneError("Unable to retrieve the public GitHub repository") from exc
