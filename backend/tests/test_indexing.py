from dataclasses import dataclass, field

from app.models.repository import RepositoryFile
from app.services.chunking import ChunkingConfig, CodeChunkingService
from app.services.embedding import EmbeddingConfig, EmbeddingService
from app.services.indexing import RepositoryIndexingService


@dataclass
class FakeEmbeddingProvider:
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


@dataclass
class FakeVectorStore:
    replacements: list[tuple[str, list[object]]] = field(default_factory=list)

    def replace_repository(self, repository_id: str, chunks: list[object]) -> None:
        self.replacements.append((repository_id, chunks))


def test_indexing_composes_file_chunk_embedding_and_storage() -> None:
    repository_file = RepositoryFile(
        path="src/example.py",
        language="Python",
        content="def answer():\n    return 42\n",
        size_bytes=28,
        line_count=2,
    )
    vector_store = FakeVectorStore()
    service = RepositoryIndexingService(
        CodeChunkingService(ChunkingConfig(min_chunk_size=0)),
        EmbeddingService(FakeEmbeddingProvider(), EmbeddingConfig(model="test")),  # type: ignore[arg-type]
        vector_store,  # type: ignore[arg-type]
    )

    result = service.index_repository("repo-1", [repository_file])

    assert result.repository_id == "repo-1"
    assert result.chunk_count == 1
    assert result.vector_dimension == 2
    stored_id, stored_chunks = vector_store.replacements[0]
    assert stored_id == "repo-1"
    assert stored_chunks[0].chunk.repository_id == "repo-1"
