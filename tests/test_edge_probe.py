"""Task 12.6: testing the camera from the laptop (probe_rtsp, probe_phone), with stubbed ffmpeg."""

import asyncio
import shutil

import httpx
import pytest

import edge_agent.rtsp as rtsp_module
from edge_agent.camera import A12Camera, RtspCamera, probe_phone
from edge_agent.rtsp import probe_rtsp

RTSP = "rtsp://tapo-user:s3cret@192.168.1.60:554/stream2"
PHONE = "http://phone:8080"


class _Probe:
    """An ffmpeg that already finished with code and stderr, or hangs when hangs=True."""

    def __init__(self, code: int = 0, stderr: bytes = b"", hangs: bool = False):
        self.returncode = None
        self.args: tuple = ()
        self._code, self._stderr, self._hangs = code, stderr, hangs

    async def communicate(self):
        if self._hangs:
            await asyncio.sleep(10)
        self.returncode = self._code
        return None, self._stderr

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode


@pytest.fixture
def ffmpeg_installed(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "C:/ffmpeg/bin/ffmpeg.exe")


def _run_ffmpeg_with(monkeypatch, process: _Probe) -> list:
    seen = []

    async def fake_exec(*args, **kwargs):
        seen.append(args)
        process.args = args
        return process

    monkeypatch.setattr(rtsp_module.asyncio, "create_subprocess_exec", fake_exec)
    return seen


@pytest.mark.anyio
async def test_a_stream_that_opens_is_connected(monkeypatch, ffmpeg_installed):
    seen = _run_ffmpeg_with(monkeypatch, _Probe(code=0))

    assert await probe_rtsp(RTSP) == "CONNECTED"
    assert RTSP in seen[0]  # the URL is what ffmpeg reads


@pytest.mark.anyio
async def test_a_wrong_password_is_auth_failed(monkeypatch, ffmpeg_installed):
    _run_ffmpeg_with(
        monkeypatch, _Probe(code=1, stderr=b"method DESCRIBE failed: 401 Unauthorized")
    )

    assert await probe_rtsp(RTSP) == "AUTH_FAILED"


@pytest.mark.anyio
async def test_any_other_failure_is_unreachable(monkeypatch, ffmpeg_installed):
    _run_ffmpeg_with(monkeypatch, _Probe(code=1, stderr=b"Connection refused"))

    assert await probe_rtsp(RTSP) == "UNREACHABLE"


@pytest.mark.anyio
async def test_a_camera_that_never_answers_is_unreachable_and_ffmpeg_is_killed(
    monkeypatch, ffmpeg_installed
):
    process = _Probe(hangs=True)
    _run_ffmpeg_with(monkeypatch, process)

    assert await probe_rtsp(RTSP, timeout=0.01) == "UNREACHABLE"
    assert process.returncode == -9


@pytest.mark.anyio
async def test_without_ffmpeg_the_probe_says_so(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert await probe_rtsp(RTSP) == "FFMPEG_MISSING"


@pytest.mark.anyio
async def test_the_phone_is_connected_when_it_sends_a_frame():
    jpeg = b"\xff\xd8\xff\xe0"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=jpeg))
    ) as client:
        assert await probe_phone(client, PHONE) == "CONNECTED"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(503))
    ) as client:
        assert await probe_phone(client, PHONE) == "UNREACHABLE"


@pytest.mark.anyio
async def test_frame_at_tells_when_the_last_good_frame_was_taken():
    jpeg = b"\xff\xd8\xff\xe0"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=jpeg))
    ) as client:
        camera = A12Camera(client, PHONE)
        assert camera.frame_at == 0.0
        await camera.snapshot()
        assert camera.frame_at > 0.0

    rtsp = RtspCamera(RTSP, fps=1)
    assert rtsp.frame_at == 0.0
    rtsp._taken_at = 42.0
    assert rtsp.frame_at == 42.0
