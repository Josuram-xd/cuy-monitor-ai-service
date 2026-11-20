"""Task 12: the edge agent against fake phones and a fake AI service (httpx.MockTransport)."""

import asyncio
import io
import logging
import struct
import wave
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

import edge_agent.agent as agent
from edge_agent.agent import ClipBuffer, record_audio, run, run_frames, send_audio
from edge_agent.audio import (
    A12Microphone,
    AudioUnavailable,
    PcmFormat,
    parse_wav_header,
    wav_clips,
)
from edge_agent.backoff import Backoff
from edge_agent.camera import A12Camera, CameraUnavailable
from edge_agent.config import EdgeSettings
from edge_agent.remote_camera import BackendUnreachable, CameraSource
from edge_agent.uploader import AiServiceUploader, SendResult, utc_iso

JPEG = b"\xff\xd8\xff\xe0fake-jpeg"
NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
PHONE = "http://phone:8080"
AI = "https://ai.example"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _stream_header(rate: int = 16_000, channels: int = 1, bits: int = 16) -> bytes:
    """The header IP Webcam sends: sizes unknown (0xFFFFFFFF) because the stream never ends."""
    byte_rate = rate * channels * bits // 8
    fmt = struct.pack("<HHIIHH", 1, channels, rate, byte_rate, channels * bits // 8, bits)
    return (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + fmt
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
    )


async def _chunks(*parts: bytes):
    for part in parts:
        yield part


class _Clock:
    def __init__(self):
        self.slept: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


# ---- configuration


def test_settings_need_the_phone_the_service_the_key_and_the_cage(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = EdgeSettings(
        _env_file=None,
        a12_url="http://phone:8080/",
        ai_url="https://ai.example/",
        api_key="k",
        cage_id="cage-1",
    )
    assert settings.a12_url == "http://phone:8080"
    assert settings.ai_url == "https://ai.example"
    assert settings.fps == 1.0
    assert settings.backend_url == "" and settings.camera_poll_seconds == 15.0

    backend = EdgeSettings(
        _env_file=None,
        ai_url=AI,
        api_key="k",
        cage_id="c",
        backend_url="https://api.example/",
        camera_poll_seconds=5,
    )
    assert backend.backend_url == "https://api.example"
    with pytest.raises(ValueError):
        EdgeSettings(_env_file=None, ai_url=AI, api_key="k", cage_id="c", camera_poll_seconds=1)

    with pytest.raises(ValueError):
        EdgeSettings(_env_file=None, a12_url="phone", ai_url=AI, api_key="k", cage_id="c")
    with pytest.raises(ValueError):
        EdgeSettings(_env_file=None, a12_url=PHONE, ai_url=AI, api_key="k", cage_id="c", fps=5)


def test_backoff_doubles_up_to_the_maximum_and_resets():
    backoff = Backoff(1, 8)
    assert [backoff.next_delay() for _ in range(5)] == [1, 2, 4, 8, 8]
    backoff.reset()
    assert backoff.next_delay() == 1


# ---- Task 12.1: the camera


@pytest.mark.anyio
async def test_the_camera_reads_the_last_frame_from_shot_jpg():
    seen = []

    def phone(request: httpx.Request):
        seen.append(request.url.path)
        return httpx.Response(200, content=JPEG)

    async with _client(phone) as client:
        assert await A12Camera(client, PHONE).snapshot() == JPEG
    assert seen == ["/shot.jpg"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(503),
        httpx.Response(200, content=b"<html>login</html>"),
        httpx.ConnectError("off"),
    ],
)
async def test_a_phone_that_is_off_or_answers_something_else_is_unavailable(answer):
    def phone(request):
        if isinstance(answer, Exception):
            raise answer
        return answer

    async with _client(phone) as client:
        with pytest.raises(CameraUnavailable):
            await A12Camera(client, PHONE).snapshot()


# ---- Task 12.2: sending, with retries and without a queue


@pytest.mark.anyio
async def test_a_frame_is_posted_with_the_key_the_cage_and_the_capture_time():
    received = []

    def service(request: httpx.Request):
        received.append(request)
        return httpx.Response(202)

    async with _client(service) as client:
        result = await AiServiceUploader(client, AI, "secret", "cage-7").send_frame(JPEG, NOW)

    assert result is SendResult.SENT
    [request] = received
    assert request.url == f"{AI}/ai/v1/frames"
    assert request.headers["X-API-Key"] == "secret"
    body = request.content
    assert b'name="cageId"\r\n\r\ncage-7' in body
    assert b'name="capturedAt"\r\n\r\n2026-11-05T12:00:00.000Z' in body
    assert b"image/jpeg" in body and JPEG in body


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (500, SendResult.RETRY),
        (503, SendResult.RETRY),
        (429, SendResult.RETRY),
        (401, SendResult.REJECTED),
        (400, SendResult.REJECTED),
        (413, SendResult.REJECTED),
    ],
)
async def test_what_to_do_after_each_answer(status, expected):
    async with _client(lambda r: httpx.Response(status)) as client:
        assert await AiServiceUploader(client, AI, "k", "c").send_audio(b"wav", NOW) is expected


@pytest.mark.anyio
async def test_no_internet_means_retry():
    def offline(request):
        raise httpx.ConnectError("no route")

    async with _client(offline) as client:
        assert (
            await AiServiceUploader(client, AI, "k", "c").send_frame(JPEG, NOW) is SendResult.RETRY
        )


def test_capture_times_are_utc_with_z():
    assert utc_iso(NOW) == "2026-11-05T12:00:00.000Z"


class _Camera:
    def __init__(self, frames):
        self.frames = list(frames)

    async def snapshot(self):
        item = self.frames.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _Uploader:
    def __init__(self, results):
        self.results = list(results)
        self.sent: list[bytes] = []

    async def send_frame(self, frame, captured_at):
        self.sent.append(frame)
        return self.results.pop(0)

    async def send_audio(self, clip, captured_at):
        self.sent.append(clip)
        return self.results.pop(0)


@pytest.mark.anyio
async def test_frames_keep_the_pace_and_a_failed_one_is_replaced_by_a_fresh_one():
    camera = _Camera([b"f1", b"f2", b"f3", b"f4"])
    uploader = _Uploader([SendResult.SENT, SendResult.RETRY, SendResult.RETRY, SendResult.SENT])
    clock = _Clock()

    sent = await run_frames(
        camera, uploader, fps=2, sleep=clock.sleep, monotonic=lambda: 0.0, max_ticks=4
    )

    assert sent == 2
    # every tick took a new frame: nothing was queued or sent twice
    assert uploader.sent == [b"f1", b"f2", b"f3", b"f4"]
    # 0.5 s between frames, 1 s and 2 s of backoff after the failures, then the pace again
    assert clock.slept == [0.5, 1, 2, 0.5]


@pytest.mark.anyio
async def test_a_silent_phone_backs_off_and_a_wrong_key_waits_a_minute():
    camera = _Camera([CameraUnavailable("off"), CameraUnavailable("off"), b"f"])
    uploader = _Uploader([SendResult.REJECTED])
    clock = _Clock()

    await run_frames(camera, uploader, fps=1, sleep=clock.sleep, monotonic=lambda: 0.0, max_ticks=3)

    assert clock.slept == [1, 2, 60]


# ---- Task 12.3: audio clips


def test_the_endless_wav_header_is_understood():
    header = _stream_header(rate=44_100, channels=2)

    fmt, start = parse_wav_header(header + b"\x00\x00")

    assert fmt == PcmFormat(channels=2, sample_rate=44_100, sample_width=2)
    assert start == len(header)
    assert parse_wav_header(header[:20]) is None  # more bytes needed


@pytest.mark.parametrize(
    "header",
    [
        b"NOPE" + b"\x00" * 40,
        _stream_header(bits=8),
        b"RIFF\x00\x00\x00\x00WAVEdata\x00\x00\x00\x00",
    ],
)
def test_a_stream_that_is_not_16_bit_pcm_wav_is_refused(header):
    with pytest.raises(AudioUnavailable):
        parse_wav_header(header)


@pytest.mark.anyio
async def test_the_stream_is_cut_into_wav_clips_of_one_second():
    second = b"\x01\x00" * 16_000
    stream = (
        _stream_header()[:30],
        _stream_header()[30:] + second[:1000],
        second[1000:] + second,
        b"\x02",
    )

    clips = [clip async for clip in wav_clips(_chunks(*stream), 1.0)]

    assert len(clips) == 2  # the last byte waits for the next second
    with wave.open(io.BytesIO(clips[0])) as clip:
        assert (clip.getnchannels(), clip.getframerate(), clip.getsampwidth()) == (1, 16_000, 2)
        assert clip.getnframes() == 16_000


@pytest.mark.anyio
async def test_the_microphone_streams_audio_wav_from_the_phone():
    body = _stream_header() + b"\x00\x00" * 16_000

    async with _client(lambda r: httpx.Response(200, content=body)) as client:
        microphone = A12Microphone(client, PHONE)
        clips = []
        with pytest.raises(AudioUnavailable, match="ended"):
            async for clip in microphone.clips(1.0):
                clips.append(clip)

    assert len(clips) == 1


def test_the_clip_buffer_keeps_only_the_newest():
    buffer = ClipBuffer(size=2)
    for i in range(4):
        buffer.put(bytes([i]), NOW)

    assert len(buffer) == 2
    assert buffer.dropped == 2


@pytest.mark.anyio
async def test_recorded_clips_are_sent_and_failures_back_off():
    buffer = ClipBuffer()
    for clip in (b"c1", b"c2", b"c3"):
        buffer.put(clip, NOW)
    uploader = _Uploader([SendResult.SENT, SendResult.RETRY, SendResult.SENT])
    clock = _Clock()

    sent = await send_audio(buffer, uploader, sleep=clock.sleep, max_clips=3)

    assert sent == 2
    assert uploader.sent == [b"c1", b"c2", b"c3"]
    assert clock.slept == [1]


@pytest.mark.anyio
async def test_a_dropped_audio_stream_reconnects_with_backoff():
    class _Microphone:
        def __init__(self):
            self.connections = 0

        async def clips(self, seconds):
            self.connections += 1
            yield b"clip"
            raise AudioUnavailable("the phone audio stream ended")

    microphone, buffer, clock = _Microphone(), ClipBuffer(), _Clock()

    async def sleep(seconds):
        await clock.sleep(seconds)
        if len(clock.slept) == 2:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await record_audio(microphone, buffer, 1.0, sleep=sleep)

    assert microphone.connections == 2
    assert len(buffer) == 2
    # a clip arrived after the first reconnection, so the backoff started again
    assert clock.slept == [1, 1]


@pytest.mark.anyio
async def test_the_warnings_name_the_camera_that_is_failing(caplog):
    class _Microphone:
        def clips(self, seconds):
            raise AudioUnavailable("the camera audio stream ended")

    class _Silent:
        async def snapshot(self):
            raise CameraUnavailable("no recent frame")

    async def stop(seconds):
        raise asyncio.CancelledError

    caplog.set_level(logging.WARNING, logger="edge_agent")
    with pytest.raises(asyncio.CancelledError):
        await record_audio(_Microphone(), ClipBuffer(), 1.0, sleep=stop, source="the RTSP camera")
    with pytest.raises(asyncio.CancelledError):
        await run_frames(
            _Silent(), _Uploader([]), 1, sleep=stop, max_ticks=1, source="the RTSP camera"
        )

    assert "No audio from the RTSP camera" in caplog.text
    assert "No frame from the RTSP camera" in caplog.text
    assert "from the phone" not in caplog.text


# ---- Task 12.6: the camera from the backend, switched while the agent runs


class _Backend:
    """Answers the camera questions of run() in order (a CameraSource, None or an exception).

    run() stops after the last answer: the sleep that follows it raises _Stop.
    """

    def __init__(self, *answers, probe="CONNECTED"):
        self.answers = list(answers)
        self.probe_answer = probe
        self.probed: list[CameraSource] = []

    async def fetch(self, client, base, key):
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def probe(self, source, phone):
        self.probed.append(source)
        return self.probe_answer

    async def sleep(self, seconds):
        await asyncio.sleep(0)  # lets the camera task start, as a real poll would
        if not self.answers:
            raise _Stop


class _Stop(Exception):
    pass


PHONE_CAMERA = CameraSource("PHONE", "http://192.168.1.50:8080", version=3)
RTSP_CAMERA = CameraSource("RTSP", "rtsp://tapo:pw@192.168.1.60:554/stream2", version=4)


def _supervised_settings(**overrides) -> EdgeSettings:
    values = {"ai_url": AI, "api_key": "k", "cage_id": "cage-1"} | overrides
    return EdgeSettings(_env_file=None, **values)


@pytest.fixture
def loops(monkeypatch):
    """Fake camera loops (they run until cancelled) and a recorder of the status reports."""
    started, stopped, reports = [], [], []

    async def fake_run_source(source, camera, settings, phone, uploader):
        started.append(source)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stopped.append(source)
            raise

    async def fake_report(client, base, key, version, status):
        reports.append((version, status))

    monkeypatch.setattr(agent, "run_source", fake_run_source)
    monkeypatch.setattr(agent, "report_status", fake_report)
    return SimpleNamespace(started=started, stopped=stopped, reports=reports)


@pytest.mark.anyio
async def test_a_new_camera_from_the_backend_replaces_the_old_one(loops):
    backend = _Backend(PHONE_CAMERA, RTSP_CAMERA)

    with pytest.raises(_Stop):
        await run(
            _supervised_settings(), fetch=backend.fetch, probe=backend.probe, sleep=backend.sleep
        )

    assert loops.started == [PHONE_CAMERA, RTSP_CAMERA]
    assert loops.stopped == [PHONE_CAMERA]


@pytest.mark.anyio
async def test_an_unreachable_backend_keeps_the_camera_that_works(loops):
    backend = _Backend(PHONE_CAMERA, BackendUnreachable("down"), BackendUnreachable("down"))

    with pytest.raises(_Stop):
        await run(
            _supervised_settings(), fetch=backend.fetch, probe=backend.probe, sleep=backend.sleep
        )

    assert loops.started == [PHONE_CAMERA] and loops.stopped == []
    assert backend.probed == [PHONE_CAMERA]  # no new probe while the backend is down


@pytest.mark.anyio
async def test_without_a_backend_camera_the_env_one_is_used(loops):
    backend = _Backend(None)
    settings = _supervised_settings(a12_url="http://phone:8080")

    with pytest.raises(_Stop):
        await run(settings, fetch=backend.fetch, probe=backend.probe, sleep=backend.sleep)

    assert loops.started == [CameraSource("PHONE", "http://phone:8080")]
    assert loops.reports == []  # .env cameras have no version to report


@pytest.mark.anyio
async def test_no_camera_anywhere_starts_no_loops(loops):
    backend = _Backend(None, None)

    with pytest.raises(_Stop):
        await run(
            _supervised_settings(), fetch=backend.fetch, probe=backend.probe, sleep=backend.sleep
        )

    assert loops.started == [] and backend.probed == []


@pytest.mark.anyio
@pytest.mark.parametrize("probed", ["AUTH_FAILED", "FFMPEG_MISSING"])
async def test_a_failed_test_is_reported_with_the_backend_version(loops, probed):
    backend = _Backend(PHONE_CAMERA, PHONE_CAMERA, probe=probed)

    with pytest.raises(_Stop):
        await run(
            _supervised_settings(), fetch=backend.fetch, probe=backend.probe, sleep=backend.sleep
        )

    assert loops.reports == [(3, probed), (3, probed)]
    # FFMPEG_MISSING means no ffmpeg loops; the phone camera is still started for AUTH_FAILED
    assert len(loops.started) == (0 if probed == "FFMPEG_MISSING" else 1)


@pytest.mark.anyio
async def test_right_after_a_switch_the_status_waits_for_the_first_frames(loops):
    backend = _Backend(RTSP_CAMERA, RTSP_CAMERA, probe="CONNECTED")

    with pytest.raises(_Stop):
        await run(
            _supervised_settings(), fetch=backend.fetch, probe=backend.probe, sleep=backend.sleep
        )

    # round 1: a fresh connection is not reported yet; round 2: no frames were produced by the fake
    assert loops.reports == [(4, "UNREACHABLE")]


@pytest.mark.anyio
async def test_a_camera_loop_that_died_is_started_again(loops, monkeypatch):
    calls = []

    async def dying_then_fine(source, camera, settings, phone, uploader):
        calls.append(source)
        if len(calls) == 1:
            raise RuntimeError("boom")
        await asyncio.Event().wait()

    monkeypatch.setattr(agent, "run_source", dying_then_fine)
    backend = _Backend(PHONE_CAMERA, PHONE_CAMERA)

    with pytest.raises(_Stop):
        await run(
            _supervised_settings(), fetch=backend.fetch, probe=backend.probe, sleep=backend.sleep
        )

    assert calls == [PHONE_CAMERA, PHONE_CAMERA]
