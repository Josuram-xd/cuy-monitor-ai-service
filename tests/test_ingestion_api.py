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
        ("/ai/v1/frames", "frame.jpg", "image/jpeg"),
        ("/ai/v1/audio", "clip.wav", "audio/wav"),
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


@pytest.mark.parametrize("client", ["false"], indirect=True)
def test_the_latest_frame_of_a_cage_is_served_back(client):
    frame = b"\xff\xd8 jpeg bytes"
    client.post(
        "/ai/v1/frames",
        headers={"X-API-Key": "test-secret"},
        data={"capturedAt": "2026-10-05T14:32:00Z", "cageId": "cage-1"},
        files={"file": ("frame.jpg", frame, "image/jpeg")},
    )

    response = client.get(
        "/ai/v1/frames/latest", headers={"X-API-Key": "test-secret"}, params={"cageId": "cage-1"}
    )

    assert response.status_code == 200
    assert response.content == frame
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-captured-at"].startswith("2026-10-05T14:32:00")


@pytest.mark.parametrize("client", ["false"], indirect=True)
def test_a_cage_without_a_frame_gets_404(client):
    response = client.get(
        "/ai/v1/frames/latest", headers={"X-API-Key": "test-secret"}, params={"cageId": "cage-9"}
    )

    assert response.status_code == 404


def test_the_latest_frame_needs_the_api_key(client):
    assert client.get("/ai/v1/frames/latest", params={"cageId": "cage-1"}).status_code == 401


def test_frame_endpoint_requires_api_key(client):
    response = client.post(
        "/ai/v1/frames",
        data={"capturedAt": "2026-10-05T14:32:00Z", "cageId": "cage-test"},
        files={"file": ("frame.jpg", b"mock media", "image/jpeg")},
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    ("path", "filename", "content_type"),
    [
        ("/ai/v1/frames", "frame.png", "image/png"),
        ("/ai/v1/audio", "clip.mp3", "audio/mpeg"),
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
