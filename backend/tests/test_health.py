from fastapi.testclient import TestClient

from app.main import app


def test_health_check_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_qdrant_health_hides_connection_details(monkeypatch) -> None:
    from app.api.routes import health
    from app.core.config import Settings

    monkeypatch.setattr(health, "get_settings", lambda: Settings(qdrant_url="http://qdrant.test"))
    monkeypatch.setattr(health.QdrantClient, "get_collections", lambda self: (_ for _ in ()).throw(RuntimeError("secret host")))
    response = TestClient(app).get("/health/qdrant")

    assert response.status_code == 503
    assert response.json() == {"detail": "Qdrant is unavailable"}
