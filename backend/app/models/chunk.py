from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CodeChunk:
    """A deterministic, line-addressable unit ready for a future embedding phase."""

    chunk_id: str
    repository_id: str
    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int
    symbol_name: str | None
    chunk_type: str
    metadata: dict[str, str | int | bool] = field(default_factory=dict)
