from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.models.conversation import ConversationMessage
from app.models.generation import GeneratedAnswer
from app.services.conversation import ChatResult
from app.services.repository_lifecycle import RepositoryRecord, RepositoryNotFoundError


class FakeLifecycle:
    def __init__(self) -> None:
        self.record = RepositoryRecord("repo_123", "https://github.com/acme/demo", "demo", "indexing")
        self.deleted: list[str] = []

    def submit_indexing(self, _: str):
        return self.record

    def run_indexing(self, repository_id: str) -> None:
        if repository_id == self.record.repository_id:
            self.record.status = "ready"

    def list(self):
        return [self.record]

    def get(self, repository_id: str):
        if repository_id != self.record.repository_id:
            raise RepositoryNotFoundError("Repository was not found")
        return self.record

    def delete(self, repository_id: str) -> None:
        self.get(repository_id)
        self.deleted.append(repository_id)


@dataclass
class FakeChunk:
    chunk_id: str = "chunk-1"
    file_path: str = "src/auth.py"
    language: str = "Python"
    start_line: int = 1
    end_line: int = 3
    symbol_name: str | None = "authenticate"
    chunk_type: str = "function"
    content: str = "def authenticate(): pass"


class FakeCodeIntelligence:
    def list_files(self, repository_id: str):
        return [type("Stored", (), {"chunk": FakeChunk()})()]

    def search(self, *args):
        chunk = FakeChunk()
        return [type("Result", (), {"chunk": chunk, "rank": 1, "relevance_score": 0.9})()]


class FakeChatService:
    def chat(self, repository_id, message, conversation_id=None, top_k=None):
        return ChatResult(
            conversation_id=conversation_id or "new-conversation",
            standalone_query=message,
            answer=GeneratedAnswer("Grounded answer", 0, 0),
            retrieved=[],
        )


def test_production_api_endpoints_with_mocked_services(monkeypatch):
    from app.api.routes import chat as chat_route
    from app.api.routes import repositories_api as repository_route
    from app.main import app

    lifecycle = FakeLifecycle()
    monkeypatch.setattr(repository_route, "create_repository_lifecycle_service", lambda _: lifecycle)
    monkeypatch.setattr(repository_route, "create_code_intelligence_service", lambda _: FakeCodeIntelligence())
    monkeypatch.setattr(chat_route, "get_conversation_service", lambda _: FakeChatService())
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    index = client.post("/repositories/index", json={"repository_url": "https://github.com/acme/demo"})
    assert index.status_code == 202
    assert index.json()["repository_id"] == "repo_123"
    assert client.get("/repositories").json()[0]["repository_name"] == "demo"
    assert client.get("/repositories/repo_123/status").json()["status"] == "ready"
    assert client.get("/repositories/repo_123/files").json()["files"][0]["file_path"] == "src/auth.py"
    assert client.post("/repositories/search", json={"repository_id": "repo_123", "query": "authentication"}).json()["results"][0]["rank"] == 1
    assert client.post("/chat", json={"repository_id": "repo_123", "message": "Where is auth?"}).json()["answer"] == "Grounded answer"
    assert client.delete("/repositories/repo_123").status_code == 204


def test_production_api_validates_requests_and_hides_not_found(monkeypatch):
    from app.api.routes import repositories_api as repository_route
    from app.main import app

    lifecycle = FakeLifecycle()
    monkeypatch.setattr(repository_route, "create_repository_lifecycle_service", lambda _: lifecycle)
    client = TestClient(app)

    assert client.post("/repositories/index", json={"repository_url": "not-a-url"}).status_code == 422
    assert client.post("/repositories/search", json={"repository_id": "repo_123", "query": ""}).status_code == 422
    missing = client.get("/repositories/missing")
    assert missing.status_code == 404
    assert "traceback" not in missing.text.lower()
