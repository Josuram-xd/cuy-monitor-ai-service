"""The loops of the edge agent: frames at 1-2 fps and audio clips, each one recovering on its own.

Nothing grows without limit (AGENTS: old Celeron, unstable internet):
- frames: one at a time; a frame that failed is dropped and the next tick takes a fresh one;
- audio: at most MAX_PENDING_CLIPS waiting to be sent; when the upload is slower than the
  microphone the oldest clip is dropped, the phone stream is never paused.
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import httpx

from edge_agent.audio import A12Microphone, AudioUnavailable
from edge_agent.backoff import Backoff
from edge_agent.camera import A12Camera, CameraUnavailable
from edge_agent.config import EdgeSettings
from edge_agent.uploader import AiServiceUploader, SendResult

logger = logging.getLogger("edge_agent")

MAX_PENDING_CLIPS = 5
# after a REJECTED answer (wrong key or cage) try only now and then: a person must fix the config
REJECTED_PAUSE_SECONDS = 60.0

Sleep = Callable[[float], Awaitable[None]]
Now = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


async def run_frames(
    camera: A12Camera,
    uploader: AiServiceUploader,
    fps: float,
    sleep: Sleep = asyncio.sleep,
    now: Now = utc_now,
    monotonic: Callable[[], float] = time.monotonic,
    max_ticks: int | None = None,
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
            logger.warning("No frame from the phone (%s); next try in %.0f s", exception, delay)
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
    microphone: A12Microphone,
    buffer: ClipBuffer,
    clip_seconds: float,
    sleep: Sleep = asyncio.sleep,
    now: Now = utc_now,
) -> None:
    """Read the phone microphone forever, reconnecting with backoff when the stream drops."""
    backoff = Backoff()
    while True:
        try:
            async for clip in microphone.clips(clip_seconds):
                buffer.put(clip, now())
                backoff.reset()
        except AudioUnavailable as exception:
            delay = backoff.next_delay()
            logger.warning("No audio from the phone (%s); reconnecting in %.0f s", exception, delay)
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


async def run(settings: EdgeSettings) -> None:
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    # the phone is on the LAN, the service on the internet: separate clients, separate connection
    # pools
    async with (
        httpx.AsyncClient(timeout=timeout) as phone,
        httpx.AsyncClient(timeout=timeout) as service,
    ):
        uploader = AiServiceUploader(
            service, settings.ai_url, settings.api_key.get_secret_value(), settings.cage_id
        )
        tasks = [run_frames(A12Camera(phone, settings.a12_url), uploader, settings.fps)]
        if settings.audio_enabled:
            buffer = ClipBuffer()
            microphone = A12Microphone(phone, settings.a12_url)
            tasks += [
                record_audio(microphone, buffer, settings.audio_clip_seconds),
                send_audio(buffer, uploader),
            ]
        logger.info(
            "Edge agent for %s: %s fps from %s to %s (audio %s)",
            settings.cage_id,
            settings.fps,
            settings.a12_url,
            settings.ai_url,
            "on" if settings.audio_enabled else "off",
        )
        await asyncio.gather(*tasks)
