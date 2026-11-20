from fastapi.testclient import TestClient

from app.main import app


def test_health_is_public_and_reports_mock_mode(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("MOCK_MODE", "true")

    # with the lifespan: the models are loaded at startup, not left over from another test
    with TestClient(app) as client:
        response = client.get("/ai/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "UP",
        "models": {"detector": False, "animal": False, "behavior": False, "audio": False},
        "mockMode": True,
    }
