"""RTSP cameras (TP-Link Tapo) for the edge agent: no real ffmpeg, the process is a stub."""

import asyncio
import shutil

import httpx
import pytest

import edge_agent.audio as audio_module
import edge_agent.camera as camera_module
from edge_agent.agent import choose_camera, choose_microphone
from edge_agent.audio import A12Microphone, AudioUnavailable, RtspMicrophone
from edge_agent.camera import (
    A12Camera,
    CameraAuthFailed,
    CameraUnavailable,
    RtspCamera,
    split_jpeg_frames,
)
from edge_agent.config import EdgeSettings, mask_url
from edge_agent.remote_camera import CameraSource
from edge_agent.rtsp import FfmpegMissing, require_ffmpeg

RTSP = "rtsp://tapo-user:s3cret@192.168.1.60:554/stream1"
FRAME_A = b"\xff\xd8\xff\xe0a\xff\x00\xff\xd9"  # 0xFF 0x00 inside the data is not a marker
FRAME_B = b"\xff\xd8b\xff\xd9"


def _settings(**overrides) -> EdgeSettings:
    values = {"ai_url": "https://ai.example", "api_key": "k", "cage_id": "cage-1"} | overrides
    return EdgeSettings(_env_file=None, **values)


# ---- the JPEG frame splitter


def test_frames_are_cut_between_soi_and_eoi_and_junk_is_dropped():
    frames, rest = split_jpeg_frames(b"junk" + FRAME_A + b"--" + FRAME_B + b"\xff\xd8partial")

    assert frames == [FRAME_A, FRAME_B]
    assert rest == b"\xff\xd8partial"


def test_a_frame_split_over_two_reads_is_completed_by_the_next_one():
    frames, rest = split_jpeg_frames(FRAME_B[:3])
    assert frames == [] and rest == FRAME_B[:3]

    frames, rest = split_jpeg_frames(rest + FRAME_B[3:])
    assert frames == [FRAME_B] and rest == b""


def test_a_trailing_ff_may_be_the_start_of_the_next_frame():
    assert split_jpeg_frames(b"junk\xff") == ([], b"\xff")
    assert split_jpeg_frames(b"junk") == ([], b"")
    assert split_jpeg_frames(b"\xff" + FRAME_B) == ([FRAME_B], b"")


# ---- masking the password in logs


@pytest.mark.parametrize(
    ("url", "masked"),
    [
        (RTSP, "rtsp://tapo-user:***@192.168.1.60:554/stream1"),
        ("rtsp://u:pa@ss@host:554/s", "rtsp://u:***@host:554/s"),
        ("rtsp://192.168.1.60:554/stream2", "rtsp://192.168.1.60:554/stream2"),
        ("rtsp://tapo-user@192.168.1.60/stream1", "rtsp://tapo-user@192.168.1.60/stream1"),
    ],
)
def test_the_password_never_reaches_the_logs(url, masked):
    assert mask_url(url) == masked
    assert "s3cret" not in mask_url(RTSP)


# ---- choosing the camera from the settings


def test_a_phone_source_is_read_as_before():
    phone = httpx.AsyncClient()
    settings = _settings(a12_url="http://phone:8080")
    source = CameraSource("PHONE", "http://phone:8080")

    assert isinstance(choose_camera(source, settings, phone), A12Camera)
    assert isinstance(choose_microphone(source, phone), A12Microphone)


def test_an_rtsp_source_gives_frames_and_audio():
    phone = httpx.AsyncClient()
    settings = _settings(a12_url="", camera_rtsp_url=RTSP, fps=2)
    source = CameraSource("RTSP", RTSP)

    camera = choose_camera(source, settings, phone)

    assert isinstance(camera, RtspCamera) and camera._url == RTSP and camera._fps == 2
    assert isinstance(choose_microphone(source, phone), RtspMicrophone)


def test_the_rtsp_url_must_be_rtsp_and_the_camera_is_optional():
    with pytest.raises(ValueError, match="rtsp://"):
        _settings(a12_url="", camera_rtsp_url="http://tapo/stream1")
    assert _settings(a12_url="").a12_url == ""
    assert _settings(a12_url="http://phone:8080").camera_rtsp_url == ""


def test_ffmpeg_missing_is_a_clear_startup_error(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(FfmpegMissing, match="winget install Gyan.FFmpeg"):
        require_ffmpeg()

    monkeypatch.setattr(shutil, "which", lambda name: "C:/ffmpeg/bin/ffmpeg.exe")
    require_ffmpeg()


# ---- the RTSP readers, with a stubbed ffmpeg


class _Stdout:
    def __init__(self, *chunks: bytes):
        self._chunks = list(chunks)

    async def read(self, size: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""


class _Stalled:
    async def read(self, size: int) -> bytes:
        await asyncio.sleep(10)
        return b""


class _Process:
    def __init__(self, stdout):
        self.stdout = stdout
        self.returncode = None

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode if self.returncode is not None else 0


@pytest.mark.anyio
async def test_the_latest_frame_is_served_and_an_old_one_is_not():
    camera = RtspCamera(RTSP, fps=1)
    with pytest.raises(CameraUnavailable):
        await camera.snapshot()

    assert await camera._read_frames(_Process(_Stdout(FRAME_A, FRAME_B))) == 2
    assert await camera.snapshot() == FRAME_B

    camera._taken_at -= camera_module.STALE_SECONDS + 1
    with pytest.raises(CameraUnavailable):
        await camera.snapshot()


@pytest.mark.anyio
async def test_a_stream_that_never_ends_a_frame_is_cut_off_instead_of_growing(monkeypatch):
    garbage = bytes([0xFF, 0xD8]) + b"x" * (camera_module.MAX_FRAME_BYTES + 1)
    camera = RtspCamera(RTSP, fps=1)

    assert await camera._read_frames(_Process(_Stdout(garbage))) == 0
    assert camera._latest is None


@pytest.mark.anyio
async def test_a_stopped_ffmpeg_is_restarted_with_backoff(monkeypatch):
    started, delays = [], []

    async def fake_ffmpeg(url, *args):
        started.append(url)
        if len(started) == 1:
            raise FileNotFoundError("ffmpeg")  # could not start: no frames
        if len(started) == 2:
            return _Process(_Stdout(FRAME_A))  # one frame, then the stream ends
        raise asyncio.CancelledError  # stop the loop after the third start

    async def sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(camera_module, "start_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(camera_module, "probe_rtsp", _unreachable)
    camera = RtspCamera(RTSP, fps=1)

    with pytest.raises(asyncio.CancelledError):
        await camera.run(sleep=sleep)

    assert started == [RTSP, RTSP, RTSP]
    # 1 s after the failed start; after the frame the backoff restarted, so 1 s again
    assert delays == [1, 1]
    assert await camera.snapshot() == FRAME_A


async def _unreachable(url, timeout=15.0):
    return "UNREACHABLE"


@pytest.mark.anyio
async def test_a_refused_login_is_not_retried(monkeypatch):
    started = []

    async def fake_ffmpeg(url, *args):
        started.append(url)
        raise FileNotFoundError("ffmpeg")  # no frames: the restart would follow

    async def refused(url, timeout=15.0):
        return "AUTH_FAILED"

    async def sleep(seconds):
        pass

    monkeypatch.setattr(camera_module, "start_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(camera_module, "probe_rtsp", refused)

    with pytest.raises(CameraAuthFailed):
        await RtspCamera(RTSP, fps=1).run(sleep=sleep)

    assert started == [RTSP]  # one ffmpeg, then the login is refused: no second attempt


@pytest.mark.anyio
async def test_a_stuck_stream_ends_the_read_so_ffmpeg_is_restarted(monkeypatch):
    monkeypatch.setattr(camera_module, "STALL_SECONDS", 0.01)

    assert await RtspCamera(RTSP, fps=1)._read_frames(_Process(_Stalled())) == 0


@pytest.mark.anyio
async def test_the_audio_is_cut_into_wav_clips_and_the_stream_end_is_reported(monkeypatch):
    second = b"\x01\x00" * 16_000
    processes, started = [], []

    async def fake_ffmpeg(url, *args):
        started.append(args)
        processes.append(_Process(_Stdout(second, second[:1000])))
        return processes[-1]

    monkeypatch.setattr(audio_module, "start_ffmpeg", fake_ffmpeg)

    clips = []
    with pytest.raises(AudioUnavailable, match="ended"):
        async for clip in RtspMicrophone(RTSP).clips(1.0):
            clips.append(clip)

    assert started[0] == ("-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-")
    assert len(clips) == 1 and clips[0][:4] == b"RIFF"
    assert processes[0].returncode is not None  # ffmpeg is stopped when the stream ends
