from dataclasses import dataclass, field

from app.models.chunk import CodeChunk
from app.models.generation import GeneratedAnswer
from app.models.reranking import RerankedRetrievalResult
from app.services.code_intelligence import RepositoryCodeIntelligenceService
from app.services.qdrant_store import StoredChunk


def chunk(chunk_id: str, repository_id: str, path: str, symbol: str | None = None) -> CodeChunk:
    return CodeChunk(chunk_id, repository_id, path, "Python", "def example(): pass", 10, 10, symbol, "function")


@dataclass
class FakeInspector:
    chunks: list[StoredChunk]
    calls: list[tuple[str, dict[str, str], int]] = field(default_factory=list)

    def list_chunks(self, repository_id: str, filters=None, limit=200):
        self.calls.append((repository_id, filters, limit))
        return [item for item in self.chunks if item.chunk.repository_id == repository_id and all(
            getattr(item.chunk, key) == value for key, value in (filters or {}).items()
        )]


@dataclass
class FakeRetriever:
    calls: list[tuple[str, str, dict[str, str] | None]] = field(default_factory=list)

    def retrieve(self, repository_id, query, top_k=None, score_threshold=None, metadata_filters=None):
        self.calls.append((repository_id, query, metadata_filters))
        item = chunk("search", repository_id, "src/auth.py", "authenticate")
        return [RerankedRetrievalResult(1, item, 0.9, 0.2, 1, 1, item.metadata)]


@dataclass
class FakeGenerator:
    contexts: list[list[object]] = field(default_factory=list)

    def generate(self, question, context):
        self.contexts.append(context)
        return GeneratedAnswer("File explanation", len(context), 10)


def make_service():
    inspector = FakeInspector([
        StoredChunk(chunk("one", "repo-a", "src/auth.py", "authenticate")),
        StoredChunk(chunk("two", "repo-a", "src/users.py", "get_user")),
        StoredChunk(chunk("foreign", "repo-b", "src/auth.py", "authenticate")),
    ])
    retriever = FakeRetriever()
    generator = FakeGenerator()
    return RepositoryCodeIntelligenceService(inspector, retriever, lambda: generator), inspector, retriever, generator


def test_file_lookup_is_repository_isolated() -> None:
    service, inspector, _, _ = make_service()

    results = service.get_file("repo-a", "src/auth.py")

    assert [item.chunk.chunk_id for item in results] == ["one"]
    assert inspector.calls[0][0] == "repo-a"
    assert inspector.calls[0][1] == {"file_path": "src/auth.py"}


def test_symbol_lookup_uses_metadata_filter() -> None:
    service, inspector, _, _ = make_service()

    results = service.get_symbol("repo-a", "get_user")

    assert [item.chunk.file_path for item in results] == ["src/users.py"]
    assert inspector.calls[0][1] == {"symbol_name": "get_user"}


def test_natural_language_search_forwards_optional_context_filters() -> None:
    service, _, retriever, _ = make_service()

    results = service.search("repo-a", "Where is authentication?", file_path="src/auth.py", language="Python")

    assert results[0].chunk.symbol_name == "authenticate"
    assert retriever.calls == [("repo-a", "Where is authentication?", {"file_path": "src/auth.py", "language": "Python"})]


def test_explain_file_uses_only_selected_file_chunks() -> None:
    service, _, _, generator = make_service()

    answer = service.explain_file("repo-a", "src/auth.py")

    assert answer.answer == "File explanation"
    assert len(generator.contexts[0]) == 1
    assert generator.contexts[0][0].chunk.file_path == "src/auth.py"
