import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture
def client(monkeypatch, request):
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("MOCK_MODE", getattr(request, "param", "true"))
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("path", "filename", "content_type"),
    [
        ("/ai/frames", "frame.jpg", "image/jpeg"),
        ("/ai/audio", "clip.wav", "audio/wav"),
    ],
)
def test_mock_ingestion_endpoints_accept_authenticated_media(client, path, filename, content_type):
    response = client.post(
        path,
        headers={"X-API-Key": "test-secret"},
        data={"capturedAt": "2026-10-05T14:32:00Z", "cageId": "cage-test"},
        files={"file": (filename, b"mock media", content_type)},
    )

    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "mockMode": True}


def test_frame_endpoint_requires_api_key(client):
    response = client.post(
        "/ai/frames",
        data={"capturedAt": "2026-10-05T14:32:00Z", "cageId": "cage-test"},
        files={"file": ("frame.jpg", b"mock media", "image/jpeg")},
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    ("path", "filename", "content_type"),
    [
        ("/ai/frames", "frame.png", "image/png"),
        ("/ai/audio", "clip.mp3", "audio/mpeg"),
    ],
)
def test_ingestion_endpoints_reject_unsupported_media(client, path, filename, content_type):
    response = client.post(
        path,
        headers={"X-API-Key": "test-secret"},
        data={"capturedAt": "2026-10-05T14:32:00Z", "cageId": "cage-test"},
        files={"file": (filename, b"mock media", content_type)},
    )

    assert response.status_code == 415


@pytest.mark.parametrize("client", ["false"], indirect=True)
def test_ingestion_endpoints_return_unavailable_when_mock_mode_is_disabled(client):
    response = client.post(
        "/ai/audio",
        headers={"X-API-Key": "test-secret"},
        data={"capturedAt": "2026-10-05T14:32:00Z", "cageId": "cage-test"},
        files={"file": ("clip.wav", b"mock media", "audio/wav")},
    )

    assert response.status_code == 503
