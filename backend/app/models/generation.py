from dataclasses import dataclass, field

from app.models.citation import SourceCitation


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Grounded generation output plus context-use diagnostics."""

    answer: str
    used_chunk_count: int
    context_char_count: int
    sources: list[SourceCitation] = field(default_factory=list)
