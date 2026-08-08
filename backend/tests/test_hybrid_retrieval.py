from dataclasses import dataclass

from app.models.chunk import CodeChunk
from app.services.hybrid_retrieval import HybridRetrievalConfig, HybridRetrievalService, fuse_ranked_results
from app.services.lexical import BM25Index, LexicalSearchResult, tokenize_code
from app.services.qdrant_store import StoredSearchResult


def chunk(chunk_id: str, content: str, path: str = "src/auth.py") -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        repository_id="repo-a",
        file_path=path,
        language="Python",
        content=content,
        start_line=1,
        end_line=2,
        symbol_name=None,
        chunk_type="function",
    )


def stored(code_chunk: CodeChunk, score: float) -> StoredSearchResult:
    return StoredSearchResult(
        score=score,
        payload={
            "repository_id": code_chunk.repository_id,
            "chunk_id": code_chunk.chunk_id,
            "file_path": code_chunk.file_path,
            "language": code_chunk.language,
            "start_line": code_chunk.start_line,
            "end_line": code_chunk.end_line,
            "symbol_name": code_chunk.symbol_name,
            "chunk_type": code_chunk.chunk_type,
            "content": code_chunk.content,
            "content_hash": "hash",
        },
    )


@dataclass
class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2]]


@dataclass
class FakeVectorStore:
    results: list[StoredSearchResult]
    repository_ids: list[str]

    def similarity_search(self, repository_id: str, query_vector: list[float], limit: int, score_threshold=None):
        self.repository_ids.append(repository_id)
        return self.results[:limit]


def test_tokenization_keeps_complete_and_component_identifiers() -> None:
    tokens = tokenize_code("authenticateUser JWT_SECRET src/auth/service.py")

    assert "authenticateuser" in tokens
    assert "authenticate" in tokens
    assert "user" in tokens
    assert "jwt_secret" in tokens
    assert "jwt" in tokens
    assert "secret" in tokens


def test_bm25_finds_exact_identifier_and_isolates_repositories() -> None:
    index = BM25Index()
    exact = chunk("exact", "def authenticateUser(): pass", "src/auth.py")
    other = chunk("other", "def getUserById(): pass", "src/users.py")
    index.index_chunks("repo-a", [exact, other])
    index.index_chunks("repo-b", [chunk("foreign", "JWT_SECRET = 'x'", "src/secret.py")])

    assert [result.chunk.chunk_id for result in index.search("repo-a", "authenticateUser", 5)] == ["exact"]
    assert index.search("repo-a", "JWT_SECRET", 5) == []


def test_hybrid_fusion_promotes_candidates_found_by_both_systems() -> None:
    semantic_only = chunk("semantic", "semantic context")
    shared = chunk("shared", "def authenticateUser(): pass")
    fused = fuse_ranked_results(
        [stored(semantic_only, 0.9), stored(shared, 0.8)],
        [LexicalSearchResult(shared, 4.0)],
        HybridRetrievalConfig(rrf_k=10),
        top_k=5,
    )

    assert fused[0].chunk.chunk_id == "shared"
    assert fused[0].vector_rank == 2
    assert fused[0].bm25_rank == 1
    assert fused[1].chunk.chunk_id == "semantic"


def test_duplicate_cross_index_results_are_returned_once() -> None:
    shared = chunk("shared", "JWT_SECRET = 'x'")
    fused = fuse_ranked_results(
        [stored(shared, 0.8)],
        [LexicalSearchResult(shared, 3.0)],
        HybridRetrievalConfig(),
        top_k=5,
    )

    assert len(fused) == 1
    assert fused[0].vector_rank == 1
    assert fused[0].bm25_rank == 1


def test_hybrid_service_returns_empty_results_and_forwards_repository_id() -> None:
    index = BM25Index()
    vector_store = FakeVectorStore(results=[], repository_ids=[])
    service = HybridRetrievalService(FakeEmbedder(), vector_store, index)  # type: ignore[arg-type]

    assert service.retrieve("repo-a", "Where is authentication handled?") == []
    assert vector_store.repository_ids == ["repo-a"]
