import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

HEADERS = {"X-API-Key": "test-secret"}
FORM = {"capturedAt": "2026-10-05T14:32:00Z", "cageId": "cage-1"}


class _Client:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


def _wav(frequency: float, amplitude: float) -> bytes:
    rate = 16_000
    t = np.arange(rate) / rate
    samples = (amplitude * 32767 * np.sin(2 * np.pi * frequency * t)).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(samples.tobytes())
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("CAGE_ID", "cage-1")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        # no real backend in a test: capture what would be sent
        app.state.backend_event_client = _Client()
        yield test_client
    get_settings.cache_clear()


def test_health_reports_the_real_engines(client):
    body = client.get("/ai/health").json()

    assert body["mockMode"] is False
    assert body["models"] == {"detector": True, "behavior": True, "audio": True}


def test_a_frame_is_kept_for_the_next_analysis(client):
    response = client.post(
        "/ai/frames", headers=HEADERS, data=FORM, files={"file": ("f.jpg", b"jpeg", "image/jpeg")}
    )

    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "mockMode": False}
    assert len(app.state.frame_window) == 1


def test_a_frame_for_another_cage_is_refused(client):
    response = client.post(
        "/ai/frames",
        headers=HEADERS,
        data=FORM | {"cageId": "cage-9"},
        files={"file": ("f.jpg", b"jpeg", "image/jpeg")},
    )

    assert response.status_code == 400
    assert len(app.state.frame_window) == 0


def test_a_huge_frame_is_refused(client):
    response = client.post(
        "/ai/frames",
        headers=HEADERS,
        data=FORM,
        files={"file": ("f.jpg", b"x" * (5 * 1024 * 1024 + 1), "image/jpeg")},
    )

    assert response.status_code == 413


def test_a_distress_clip_is_reported_every_time(client):
    for _ in range(2):
        response = client.post(
            "/ai/audio",
            headers=HEADERS,
            data=FORM,
            files={"file": ("c.wav", _wav(4000, 0.7), "audio/wav")},
        )
        assert response.status_code == 202

    events = app.state.backend_event_client.events
    assert [e.payload.label.value for e in events] == ["DISTRESS", "DISTRESS"]
    assert events[0].type.value == "AUDIO"
    assert events[0].source == "ai-service"


def test_calm_clips_are_reported_once_per_window(client):
    for _ in range(3):
        client.post(
            "/ai/audio",
            headers=HEADERS,
            data=FORM,
            files={"file": ("c.wav", _wav(200, 0.01), "audio/wav")},
        )

    assert [e.payload.label.value for e in app.state.backend_event_client.events] == ["NORMAL"]


def test_a_file_that_is_not_a_wav_gets_422(client):
    response = client.post(
        "/ai/audio",
        headers=HEADERS,
        data=FORM,
        files={"file": ("c.wav", b"not a wav", "audio/wav")},
    )

    assert response.status_code == 422
    assert app.state.backend_event_client.events == []


def test_real_mode_still_needs_the_api_key(client):
    response = client.post(
        "/ai/frames", data=FORM, files={"file": ("f.jpg", b"jpeg", "image/jpeg")}
    )

    assert response.status_code == 401
