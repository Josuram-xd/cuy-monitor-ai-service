import hashlib

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.vision.model_store import load_detector
from scripts.fetch_models import fetch


def test_a_missing_or_broken_detector_is_never_fatal(tmp_path):
    assert load_detector(tmp_path / "detector.onnx", 0.35) is None
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"not an onnx model")
    assert load_detector(broken, 0.35) is None


def test_health_reports_a_loaded_detector_even_in_mock_mode(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setattr(main, "load_detector", lambda path, confidence: object())

    with TestClient(main.app) as client:
        models = client.get("/ai/health").json()["models"]

    assert models == {"detector": True, "behavior": False, "audio": False}


def test_the_model_download_is_checked_against_its_sha256(tmp_path):
    source = tmp_path / "release.onnx"
    source.write_bytes(b"model bytes")
    digest = hashlib.sha256(b"model bytes").hexdigest()

    target = fetch(source.as_uri(), digest, tmp_path / "models" / "detector.onnx")

    assert target.read_bytes() == b"model bytes"
    with pytest.raises(ValueError, match="sha256"):
        fetch(source.as_uri(), "0" * 64, tmp_path / "models" / "other.onnx")
    assert not (tmp_path / "models" / "other.onnx").exists()
