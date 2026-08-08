"""Bounded multi-turn query rewriting while keeping retrieval history-independent."""

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import Settings
from app.models.conversation import ConversationMessage
from app.models.generation import GeneratedAnswer
from app.models.reranking import RerankedRetrievalResult
from app.services.generation import GenerationProviderError, GeminiProvider, LLMProvider, create_generation_service
from app.services.reranked_retrieval import RerankedRetrievalService, create_reranked_retrieval_service

logger = logging.getLogger(__name__)


class ConversationStore:
    """Process-local message store partitioned by repository and conversation ID."""

    def __init__(self, max_history_messages: int) -> None:
        if max_history_messages < 1:
            raise ValueError("maximum history messages must be positive")
        self._max_history_messages = max_history_messages
        self._histories: dict[tuple[str, str], list[ConversationMessage]] = {}

    def get(self, repository_id: str, conversation_id: str) -> list[ConversationMessage]:
        return list(self._histories.get((repository_id, conversation_id), []))

    def append(self, repository_id: str, conversation_id: str, message: ConversationMessage) -> None:
        key = (repository_id, conversation_id)
        history = self._histories.setdefault(key, [])
        history.append(message)
        self._histories[key] = history[-self._max_history_messages :]


class QueryRewriter:
    """Uses an LLM to turn a follow-up into a history-independent retrieval query."""

    def __init__(self, provider: LLMProvider, model: str, history_length: int) -> None:
        if not model.strip():
            raise ValueError("Gemini model must not be blank")
        if history_length < 1:
            raise ValueError("query rewrite history length must be positive")
        self._provider = provider
        self._model = model
        self._history_length = history_length

    def rewrite(self, message: str, history: Sequence[ConversationMessage]) -> str:
        if not history:
            return message
        prompt = build_rewrite_prompt(message, history[-self._history_length :])
        try:
            rewritten = self._provider.generate(prompt, self._model).strip()
            return rewritten or message
        except (GenerationProviderError, ValueError):
            logger.warning("query rewriting failed; using original message")
            return message


@dataclass(frozen=True, slots=True)
class ChatResult:
    conversation_id: str
    standalone_query: str
    answer: GeneratedAnswer
    retrieved: list[RerankedRetrievalResult]


class ConversationalRetrievalService:
    """Coordinates rewrite → retrieval → bounded context generation for one chat turn."""

    def __init__(
        self,
        store: ConversationStore,
        rewriter: QueryRewriter,
        retriever: RerankedRetrievalService,
        generator: object,
    ) -> None:
        self._store = store
        self._rewriter = rewriter
        self._retriever = retriever
        self._generator = generator

    def chat(
        self, repository_id: str, message: str, conversation_id: str | None = None, top_k: int | None = None
    ) -> ChatResult:
        if not repository_id.strip() or not message.strip():
            raise ValueError("repository_id and message must not be blank")
        conversation_id = conversation_id or str(uuid.uuid4())
        history = self._store.get(repository_id, conversation_id)
        standalone_query = self._rewriter.rewrite(message, history)
        retrieved = self._retriever.retrieve(repository_id, standalone_query, top_k=top_k)
        answer = self._generator.generate(standalone_query, retrieved)
        self._store.append(repository_id, conversation_id, ConversationMessage("user", message))
        self._store.append(repository_id, conversation_id, ConversationMessage("assistant", answer.answer))
        return ChatResult(conversation_id, standalone_query, answer, retrieved)


def build_rewrite_prompt(message: str, history: Sequence[ConversationMessage]) -> str:
    transcript = "\n".join(f"{item.role}: {item.content}" for item in history)
    return f"""Rewrite the latest user message into one standalone repository-search query.
Resolve pronouns and ambiguous references using the conversation history.
Return only the rewritten query, with no explanation or citation.

Conversation history:
{transcript}

Latest user message:
{message}
"""


_shared_conversation_service: ConversationalRetrievalService | None = None


def get_conversation_service(settings: Settings) -> ConversationalRetrievalService:
    """Create one process-local service so consecutive HTTP calls share bounded history."""
    global _shared_conversation_service
    if _shared_conversation_service is None:
        if not settings.gemini_api_key or not settings.gemini_model:
            raise ValueError("GEMINI_API_KEY and GEMINI_MODEL are required for conversational querying")
        _shared_conversation_service = ConversationalRetrievalService(
            store=ConversationStore(settings.conversation_max_history_messages),
            rewriter=QueryRewriter(
                GeminiProvider(
                    api_key=settings.gemini_api_key,
                    max_retries=settings.gemini_max_retries,
                    initial_backoff_seconds=settings.gemini_initial_backoff_seconds,
                ),
                settings.gemini_model,
                settings.conversation_rewrite_history_length,
            ),
            retriever=create_reranked_retrieval_service(settings),
            generator=create_generation_service(settings),
        )
    return _shared_conversation_service
