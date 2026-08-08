from dataclasses import dataclass

from app.models.chunk import CodeChunk
from app.models.retrieval import RetrievalResult
from app.services.generation import GenerationConfig, GenerationService


def result(chunk_id: str, path: str, start_line: int) -> RetrievalResult:
    chunk = CodeChunk(
        chunk_id=chunk_id,
        repository_id="repo-1",
        file_path=path,
        language="Python",
        content="def target():\n    pass\n",
        start_line=start_line,
        end_line=start_line + 1,
        symbol_name="target",
        chunk_type="function",
    )
    return RetrievalResult(chunk=chunk, score=0.9, metadata={})


@dataclass
class FakeLLM:
    answer: str

    def generate(self, prompt: str, model: str) -> str:
        return self.answer


def generate(answer: str, chunks: list[RetrievalResult]):
    return GenerationService(FakeLLM(answer), GenerationConfig(model="test")).generate("Where?", chunks)


def test_valid_citation_preserves_backend_source_metadata() -> None:
    generated = generate("Authentication is handled here [1].", [result("chunk-a", "src/auth.py", 31)])

    assert len(generated.sources) == 1
    source = generated.sources[0]
    assert source.citation_id == "1"
    assert source.file_path == "src/auth.py"
    assert source.start_line == 31
    assert source.end_line == 32
    assert source.symbol_name == "target"
    assert source.chunk_id == "chunk-a"


def test_invalid_citation_is_removed_and_not_returned() -> None:
    generated = generate("Authentication is handled here [99].", [result("chunk-a", "src/auth.py", 31)])

    assert "[99]" not in generated.answer
    assert generated.sources == []


def test_missing_citations_return_no_sources() -> None:
    generated = generate("Authentication is handled here.", [result("chunk-a", "src/auth.py", 31)])

    assert generated.sources == []


def test_duplicate_citations_are_deduplicated() -> None:
    generated = generate("Authentication is handled here [1] and again [1].", [result("chunk-a", "src/auth.py", 31)])

    assert [source.citation_id for source in generated.sources] == ["1"]


def test_sources_follow_first_citation_order() -> None:
    chunks = [result("chunk-a", "src/a.py", 10), result("chunk-b", "src/b.py", 20)]
    generated = generate("Second source [2], then first source [1].", chunks)

    assert [source.chunk_id for source in generated.sources] == ["chunk-b", "chunk-a"]
