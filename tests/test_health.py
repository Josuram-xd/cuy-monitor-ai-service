from fastapi.testclient import TestClient

import app.main as main_module
from app.behavior.engine import TrainedBehaviorEngine
from app.config import get_settings
from app.main import app


def test_health_is_public_and_reports_mock_mode(monkeypatch):
    monkeypatch.setattr(main_module, "load_animal_detector", lambda *_: None)
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
        "behaviorModel": None,
    }


def test_the_behavior_model_is_rules_when_only_a_detector_is_there(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BEHAVIOR_MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(main_module, "load_detector", lambda *_: object())
    monkeypatch.setattr(main_module, "load_animal_detector", lambda *_: None)
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            assert isinstance(app.state.behavior_engine, TrainedBehaviorEngine)
            body = client.get("/ai/health").json()
    finally:
        get_settings.cache_clear()

    assert body["behaviorModel"] == "rules"
    assert body["mockMode"] is False
