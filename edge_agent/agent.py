"""The loops of the edge agent: frames at 1-2 fps and audio clips, each one recovering on its own.

Nothing grows without limit (AGENTS: old Celeron, unstable internet):
- frames: one at a time; a frame that failed is dropped and the next tick takes a fresh one;
- audio: at most MAX_PENDING_CLIPS waiting to be sent; when the upload is slower than the
  microphone the oldest clip is dropped, the phone stream is never paused.

Task 12.6: run() is a supervisor. It asks the backend which camera the breeder chose (every
settings.camera_poll_seconds), switches to it without a restart, and reports the status back.
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import httpx

from edge_agent.audio import A12Microphone, AudioUnavailable, RtspMicrophone
from edge_agent.backoff import Backoff
from edge_agent.camera import (
    A12Camera,
    CameraAuthFailed,
    CameraUnavailable,
    RtspCamera,
    probe_phone,
)
from edge_agent.config import EdgeSettings, mask_url
from edge_agent.remote_camera import (
    BackendUnreachable,
    CameraSource,
    camera_status,
    env_source,
    fetch_source,
    report_status,
)
from edge_agent.rtsp import probe_rtsp
from edge_agent.uploader import AiServiceUploader, SendResult

logger = logging.getLogger("edge_agent")

MAX_PENDING_CLIPS = 5
# after a REJECTED answer (wrong key or cage) try only now and then: a person must fix the config
REJECTED_PAUSE_SECONDS = 60.0
# a refused login is not retried until the dashboard changes the camera (version or URL)
PARKED_WARNING = "Camera login failed; waiting for new settings from the dashboard"

Sleep = Callable[[float], Awaitable[None]]
Now = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def source_label(source: CameraSource) -> str:
    return "the RTSP camera" if source.kind == "RTSP" else "the phone"


def choose_camera(
    source: CameraSource, settings: EdgeSettings, phone: httpx.AsyncClient
) -> A12Camera | RtspCamera:
    if source.kind == "RTSP":
        return RtspCamera(source.url, settings.fps)
    return A12Camera(phone, source.url)


def choose_microphone(
    source: CameraSource, phone: httpx.AsyncClient
) -> A12Microphone | RtspMicrophone:
    if source.kind == "RTSP":
        return RtspMicrophone(source.url)
    return A12Microphone(phone, source.url)


async def probe_source(source: CameraSource, phone: httpx.AsyncClient) -> str:
    if source.kind == "RTSP":
        return await probe_rtsp(source.url)
    return await probe_phone(phone, source.url)


async def run_frames(
    camera: A12Camera | RtspCamera,
    uploader: AiServiceUploader,
    fps: float,
    sleep: Sleep = asyncio.sleep,
    now: Now = utc_now,
    monotonic: Callable[[], float] = time.monotonic,
    max_ticks: int | None = None,
    source: str = "the phone",
) -> int:
    """Send frames forever (or max_ticks times, for tests). Returns how many were accepted."""
    period = 1.0 / fps
    backoff = Backoff()
    sent = ticks = 0
    while max_ticks is None or ticks < max_ticks:
        ticks += 1
        started = monotonic()
        try:
            frame = await camera.snapshot()
        except CameraUnavailable as exception:
            delay = backoff.next_delay()
            logger.warning("No frame from %s (%s); next try in %.0f s", source, exception, delay)
            await sleep(delay)
            continue
        result = await uploader.send_frame(frame, now())
        if result is SendResult.SENT:
            sent += 1
            backoff.reset()
            await sleep(max(0.0, period - (monotonic() - started)))
        elif result is SendResult.RETRY:
            await sleep(backoff.next_delay())
        else:
            await sleep(REJECTED_PAUSE_SECONDS)
    return sent


class ClipBuffer:
    """The clips waiting to be sent; full = the oldest one goes."""

    def __init__(self, size: int = MAX_PENDING_CLIPS):
        self._clips: deque[tuple[bytes, datetime]] = deque(maxlen=size)
        self._ready = asyncio.Event()
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._clips)

    def put(self, clip: bytes, captured_at: datetime) -> None:
        if len(self._clips) == self._clips.maxlen:
            self.dropped += 1
        self._clips.append((clip, captured_at))
        self._ready.set()

    async def get(self) -> tuple[bytes, datetime]:
        while not self._clips:
            self._ready.clear()
            await self._ready.wait()
        return self._clips.popleft()


async def record_audio(
    microphone: A12Microphone | RtspMicrophone,
    buffer: ClipBuffer,
    clip_seconds: float,
    sleep: Sleep = asyncio.sleep,
    now: Now = utc_now,
    source: str = "the phone",
) -> None:
    """Read the microphone forever, reconnecting with backoff when the stream drops."""
    backoff = Backoff()
    while True:
        try:
            async for clip in microphone.clips(clip_seconds):
                buffer.put(clip, now())
                backoff.reset()
        except AudioUnavailable as exception:
            delay = backoff.next_delay()
            logger.warning(
                "No audio from %s (%s); reconnecting in %.0f s", source, exception, delay
            )
            await sleep(delay)


async def send_audio(
    buffer: ClipBuffer,
    uploader: AiServiceUploader,
    sleep: Sleep = asyncio.sleep,
    max_clips: int | None = None,
) -> int:
    """Send the recorded clips forever (or max_clips, for tests). Returns how many were accepted."""
    backoff = Backoff()
    sent = handled = 0
    while max_clips is None or handled < max_clips:
        clip, captured_at = await buffer.get()
        handled += 1
        result = await uploader.send_audio(clip, captured_at)
        if result is SendResult.SENT:
            sent += 1
            backoff.reset()
        elif result is SendResult.RETRY:
            await sleep(backoff.next_delay())
        else:
            await sleep(REJECTED_PAUSE_SECONDS)
    return sent


async def run_source(
    source: CameraSource,
    camera: A12Camera | RtspCamera,
    settings: EdgeSettings,
    phone: httpx.AsyncClient,
    uploader: AiServiceUploader,
) -> None:
    """The loops of one camera: frames, the RTSP reader and the audio. Runs until cancelled."""
    label = source_label(source)
    coros = [run_frames(camera, uploader, settings.fps, source=label)]
    if isinstance(camera, RtspCamera):
        coros.append(camera.run())
    if settings.audio_enabled:
        buffer = ClipBuffer()
        microphone = choose_microphone(source, phone)
        coros += [
            record_audio(microphone, buffer, settings.audio_clip_seconds, source=label),
            send_audio(buffer, uploader),
        ]
    tasks = [asyncio.create_task(coro) for coro in coros]
    try:
        await asyncio.gather(*tasks)
    finally:
        # one loop that raises (a refused login) must stop the others, not leave them running
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def stop(task: asyncio.Task | None) -> None:
    """Cancel the loops of the old camera and wait until ffmpeg has been stopped."""
    if task is not None:
        task.cancel()
        await asyncio.wait([task])


async def run(
    settings: EdgeSettings,
    fetch: Callable[..., Awaitable[CameraSource | None]] = fetch_source,
    probe: Callable[[CameraSource, httpx.AsyncClient], Awaitable[str]] = probe_source,
    sleep: Sleep = asyncio.sleep,
) -> None:
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    base = settings.backend_url or settings.ai_url
    key = settings.api_key.get_secret_value()
    # the phone is on the LAN, the service on the internet: separate clients, separate connection
    # pools
    async with (
        httpx.AsyncClient(timeout=timeout) as phone,
        httpx.AsyncClient(timeout=timeout) as service,
    ):
        uploader = AiServiceUploader(service, settings.ai_url, key, settings.cage_id)
        logger.info(
            "Edge agent for %s: %s fps to %s (audio %s)",
            settings.cage_id,
            settings.fps,
            settings.ai_url,
            "on" if settings.audio_enabled else "off",
        )
        current: CameraSource | None = None
        task: asyncio.Task | None = None
        camera = probed = None
        first = True
        while True:
            report_now = False
            if task is not None and task.done():
                error = task.exception()
                if isinstance(error, CameraAuthFailed):
                    # parked: `current` stays, so only a changed version or URL starts it again
                    logger.warning(PARKED_WARNING)
                    task, camera, probed = None, None, "AUTH_FAILED"
                    report_now = True
                else:
                    # a camera loop that died on an unexpected error is started again below
                    logger.error("The camera stopped unexpectedly: %s", error)
                    current, task = None, None
            try:
                source = await fetch(service, base, key) or env_source(settings)
            except BackendUnreachable as exception:
                logger.warning("Keeping the current camera: %s", exception)
                source = current or env_source(settings)
            changed = first or source != current
            first = False
            if changed:
                await stop(task)
                current, task, camera, probed = source, None, None, None
                if source is None:
                    logger.warning("No camera yet: set it up from the app (cage page > Cámara)")
                else:
                    probed = await probe(source, phone)
                    logger.info("Camera %s: %s", mask_url(source.url), probed)
                    if probed == "AUTH_FAILED":
                        logger.warning(PARKED_WARNING)
                    elif probed != "FFMPEG_MISSING":
                        camera = choose_camera(source, settings, phone)
                        task = asyncio.create_task(
                            run_source(source, camera, settings, phone, uploader)
                        )
            # right after a switch the first frames are still on their way: report next round
            # a parked camera is reported once, when it parks, not on every poll
            parked = probed == "AUTH_FAILED"
            if (
                current is not None
                and current.version is not None
                and not (changed and probed == "CONNECTED")
                and (changed or report_now or not parked)
            ):
                await report_status(
                    service, base, key, current.version, camera_status(camera, probed)
                )
            await sleep(settings.camera_poll_seconds)
