from fastapi.testclient import TestClient

from app.main import app


def test_health_is_public_and_reports_mock_mode(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("MOCK_MODE", "true")

    response = TestClient(app).get("/ai/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "UP",
        "models": {"detector": False, "behavior": False, "audio": False},
        "mockMode": True,
    }
