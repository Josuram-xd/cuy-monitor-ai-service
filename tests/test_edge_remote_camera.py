"""Task 12.6: the backend client of the edge agent (httpx.MockTransport, no network)."""

import json
import time
from types import SimpleNamespace

import httpx
import pytest

from edge_agent.config import EdgeSettings
from edge_agent.remote_camera import (
    BackendUnreachable,
    CameraSource,
    camera_status,
    env_source,
    fetch_source,
    report_status,
)

BASE = "https://ai.example"
CAMERA_URL = "rtsp://tapo-user:s3cret@192.168.1.60:554/stream2"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _settings(**overrides) -> EdgeSettings:
    values = {"ai_url": BASE, "api_key": "k", "cage_id": "cage-1"} | overrides
    return EdgeSettings(_env_file=None, **values)


@pytest.mark.anyio
async def test_the_backend_camera_comes_with_its_version_and_the_key():
    seen = []

    def backend(request: httpx.Request):
        seen.append(request)
        return httpx.Response(200, json={"version": 3, "type": "RTSP", "url": CAMERA_URL})

    async with _client(backend) as client:
        source = await fetch_source(client, BASE, "secret")

    assert source == CameraSource("RTSP", CAMERA_URL, 3)
    [request] = seen
    assert request.url == f"{BASE}/api/v1/ingestion/station/camera"
    assert request.headers["X-API-Key"] == "secret"


@pytest.mark.anyio
@pytest.mark.parametrize("status", [204, 404])
async def test_no_camera_or_an_older_backend_means_none(status):
    async with _client(lambda r: httpx.Response(status)) as client:
        assert await fetch_source(client, BASE, "k") is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(500),
        httpx.Response(401),
        httpx.ConnectError("off"),
    ],
)
async def test_a_backend_that_fails_or_refuses_the_key_is_unreachable(answer):
    def backend(request):
        if isinstance(answer, Exception):
            raise answer
        return answer

    async with _client(backend) as client:
        with pytest.raises(BackendUnreachable):
            await fetch_source(client, BASE, "k")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {"version": 1, "type": "PHONE", "url": CAMERA_URL},  # rtsp url for a phone
        {"version": 1, "type": "RTSP", "url": "http://192.168.1.60/stream2"},
        {"version": 1, "type": "WEBCAM", "url": CAMERA_URL},
        {"version": "1", "type": "RTSP", "url": CAMERA_URL},
        {"type": "RTSP", "url": CAMERA_URL},
        ["not", "a", "camera"],
    ],
)
async def test_a_camera_that_cannot_be_used_is_ignored_and_not_printed(body, caplog):
    async with _client(lambda r: httpx.Response(200, json=body)) as client:
        assert await fetch_source(client, BASE, "k") is None

    assert "s3cret" not in caplog.text
    assert "cannot be used" in caplog.text


@pytest.mark.anyio
async def test_the_status_is_posted_with_the_version_and_the_key():
    seen = []

    def backend(request: httpx.Request):
        seen.append(request)
        return httpx.Response(204)

    async with _client(backend) as client:
        await report_status(client, BASE, "secret", 3, "CONNECTED")

    [request] = seen
    assert request.url == f"{BASE}/api/v1/ingestion/station/camera/status"
    assert request.headers["X-API-Key"] == "secret"
    assert json.loads(request.content) == {"version": 3, "status": "CONNECTED"}


@pytest.mark.anyio
async def test_a_failed_status_report_is_only_logged(caplog):
    def offline(request):
        raise httpx.ConnectError("no route")

    async with _client(offline) as client:
        await report_status(client, BASE, "k", 3, "CONNECTED")  # must not raise

    async with _client(lambda r: httpx.Response(500)) as client:
        await report_status(client, BASE, "k", 3, "CONNECTED")

    assert "Could not report the camera status" in caplog.text
    assert "HTTP 500" in caplog.text


def test_the_camera_from_env_prefers_the_rtsp_url_over_the_phone():
    both = _settings(camera_rtsp_url=CAMERA_URL, a12_url="http://phone:8080")
    phone = _settings(a12_url="http://phone:8080")

    assert env_source(both) == CameraSource("RTSP", CAMERA_URL)
    assert env_source(phone) == CameraSource("PHONE", "http://phone:8080")
    # model_construct skips the validators: a settings object with no camera at all
    bare = EdgeSettings.model_construct(a12_url="", camera_rtsp_url="", ai_url=BASE)
    assert env_source(bare) is None


@pytest.mark.parametrize(
    ("frame_age", "probed", "expected"),
    [
        (1.0, "UNREACHABLE", "CONNECTED"),  # frames arrive: it works
        (120.0, "AUTH_FAILED", "AUTH_FAILED"),  # old frames: say what the probe found
        (120.0, "CONNECTED", "UNREACHABLE"),  # probe worked but no frames now
        (0.0, "FFMPEG_MISSING", "FFMPEG_MISSING"),  # never had a frame
    ],
)
def test_the_status_says_what_the_dashboard_should_show(frame_age, probed, expected):
    frame_at = time.monotonic() - frame_age if frame_age else 0.0
    camera = SimpleNamespace(frame_at=frame_at)

    assert camera_status(camera, probed) == expected


def test_without_a_camera_object_the_probe_result_is_the_status():
    assert camera_status(None, "FFMPEG_MISSING") == "FFMPEG_MISSING"
    assert camera_status(None, "CONNECTED") == "UNREACHABLE"
