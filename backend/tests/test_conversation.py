from dataclasses import dataclass, field

from app.models.generation import GeneratedAnswer
from app.services.conversation import (
    ConversationStore,
    ConversationalRetrievalService,
    QueryRewriter,
    build_rewrite_prompt,
)
from app.services.generation import GenerationProviderError


@dataclass
class FakeLLM:
    response: str = "How does the authentication implementation validate JWT tokens?"
    error: Exception | None = None
    calls: list[str] = field(default_factory=list)

    def generate(self, prompt: str, model: str) -> str:
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return self.response


@dataclass
class FakeRetriever:
    queries: list[tuple[str, str, int | None]] = field(default_factory=list)

    def retrieve(self, repository_id: str, query: str, top_k=None):
        self.queries.append((repository_id, query, top_k))
        return []


@dataclass
class FakeGenerator:
    queries: list[str] = field(default_factory=list)

    def generate(self, query: str, retrieved):
        self.queries.append(query)
        return GeneratedAnswer(answer="Grounded answer", used_chunk_count=0, context_char_count=0)


def make_service(llm: FakeLLM, history_limit: int = 6):
    retriever = FakeRetriever()
    generator = FakeGenerator()
    service = ConversationalRetrievalService(
        ConversationStore(history_limit), QueryRewriter(llm, "test-model", history_limit), retriever, generator
    )
    return service, retriever, generator


def test_first_turn_uses_original_question_without_rewrite() -> None:
    llm = FakeLLM()
    service, retriever, _ = make_service(llm)

    result = service.chat("repo-a", "Where is authentication implemented?", "conversation-a")

    assert result.standalone_query == "Where is authentication implemented?"
    assert llm.calls == []
    assert retriever.queries[0][1] == "Where is authentication implemented?"


def test_follow_up_is_rewritten_before_retrieval() -> None:
    llm = FakeLLM()
    service, retriever, generator = make_service(llm)
    service.chat("repo-a", "Where is authentication implemented?", "conversation-a")

    result = service.chat("repo-a", "How does it validate the token?", "conversation-a")

    assert result.standalone_query == "How does the authentication implementation validate JWT tokens?"
    assert "How does it validate the token?" in llm.calls[0]
    assert retriever.queries[-1][1] == result.standalone_query
    assert generator.queries[-1] == result.standalone_query


def test_query_rewrite_failure_falls_back_to_original_message() -> None:
    llm = FakeLLM(error=GenerationProviderError("offline"))
    service, retriever, _ = make_service(llm)
    service.chat("repo-a", "Explain authentication", "conversation-a")

    result = service.chat("repo-a", "How does it work?", "conversation-a")

    assert result.standalone_query == "How does it work?"
    assert retriever.queries[-1][1] == "How does it work?"


def test_history_is_bounded_and_repository_isolated() -> None:
    llm = FakeLLM()
    service, _, _ = make_service(llm, history_limit=2)
    service.chat("repo-a", "first", "shared")
    service.chat("repo-a", "second", "shared")
    service.chat("repo-a", "third", "shared")

    assert len(service._store.get("repo-a", "shared")) == 2
    other = service.chat("repo-b", "How does it work?", "shared")
    assert other.standalone_query == "How does it work?"


def test_rewrite_prompt_contains_history_for_ambiguous_references() -> None:
    prompt = build_rewrite_prompt(
        "How does it validate the token?",
        [
            type("Message", (), {"role": "user", "content": "Where is authentication implemented?"})(),
            type("Message", (), {"role": "assistant", "content": "It is in src/auth/jwt.py."})(),
        ],
    )

    assert "Where is authentication implemented?" in prompt
    assert "How does it validate the token?" in prompt
