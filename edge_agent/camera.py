"""Task 12.1: frames from the A12 running IP Webcam, or from an RTSP camera (TP-Link Tapo).

IP Webcam serves the last camera frame as a JPEG at /shot.jpg. Asking for one frame each tick is
simpler and lighter for an old Celeron than decoding the MJPEG stream at /video, and at 1-2 fps it
gives the same frames.

The Tapo C200 only streams RTSP: one ffmpeg reads it and writes JPEG frames to a pipe, RtspCamera
keeps the latest one and restarts ffmpeg when the stream stops.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import httpx

from edge_agent.backoff import Backoff
from edge_agent.rtsp import FFMPEG_READ_BYTES, start_ffmpeg, stop_ffmpeg

logger = logging.getLogger("edge_agent")

JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
# a frame never gets this big: more bytes without an end means a broken stream
MAX_FRAME_BYTES = 2 * 1024 * 1024
# no bytes from ffmpeg for this long: the stream is stuck, restart it
STALL_SECONDS = 15.0
# a frame older than this is not sent (the next tick asks again; the backoff waits)
STALE_SECONDS = 10.0


class CameraUnavailable(Exception):
    """The phone did not answer, or answered something that is not a JPEG."""


def split_jpeg_frames(data: bytes) -> tuple[list[bytes], bytes]:
    """The complete JPEG frames in data (SOI ... EOI) and the bytes to keep for the next read.

    Inside JPEG image data every 0xFF is followed by 0x00, so FFD8 and FFD9 are only markers.
    """
    frames = []
    while (start := data.find(JPEG_START)) >= 0:
        end = data.find(JPEG_END, start + 2)
        if end < 0:
            return frames, data[start:]
        frames.append(data[start : end + 2])
        data = data[end + 2 :]
    # a 0xFF at the very end may be the first byte of the next SOI
    return frames, data[-1:] if data.endswith(b"\xff") else b""


async def probe_phone(client: httpx.AsyncClient, base_url: str) -> str:
    """CONNECTED or UNREACHABLE: one frame from the phone is enough to know it works."""
    try:
        await A12Camera(client, base_url).snapshot()
    except CameraUnavailable:
        return "UNREACHABLE"
    return "CONNECTED"


class A12Camera:
    def __init__(self, client: httpx.AsyncClient, base_url: str):
        self._client = client
        self._url = f"{base_url}/shot.jpg"
        # time.monotonic() of the last good frame (0.0 = none yet)
        self.frame_at = 0.0

    async def snapshot(self) -> bytes:
        try:
            response = await self._client.get(self._url)
        except httpx.HTTPError as exception:
            raise CameraUnavailable(f"the phone did not answer: {exception}") from exception
        if response.status_code != 200:
            raise CameraUnavailable(f"the phone answered HTTP {response.status_code}")
        frame = response.content
        if not frame.startswith(JPEG_START):
            raise CameraUnavailable("the phone did not send a JPEG")
        self.frame_at = time.monotonic()
        return frame


class RtspCamera:
    """Same snapshot() as A12Camera. run() must be running next to the frame loop."""

    def __init__(self, url: str, fps: float):
        self._url = url
        self._fps = fps
        self._latest: bytes | None = None
        self._taken_at = 0.0

    @property
    def frame_at(self) -> float:
        """time.monotonic() of the last good frame (0.0 = none yet)."""
        return self._taken_at

    async def snapshot(self) -> bytes:
        if self._latest is None or time.monotonic() - self._taken_at > STALE_SECONDS:
            raise CameraUnavailable("no recent frame from the RTSP camera")
        return self._latest

    async def run(self, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        """Keep one ffmpeg reading the stream, restarting it with backoff when it stops."""
        backoff = Backoff()
        while True:
            frames, reason = 0, "ffmpeg could not start"
            try:
                process = await start_ffmpeg(
                    self._url,
                    "-vf",
                    f"fps={self._fps}",
                    "-f",
                    "image2pipe",
                    "-c:v",
                    "mjpeg",
                    "-q:v",
                    "5",
                    "-",
                )
                try:
                    frames = await self._read_frames(process)
                finally:
                    code = await stop_ffmpeg(process)
                reason = f"ffmpeg stopped with code {code}"
            except OSError as exception:
                reason = f"{reason}: {exception}"
            if frames:
                backoff.reset()
            delay = backoff.next_delay()
            logger.warning("No frames from the RTSP camera (%s); restart in %.0f s", reason, delay)
            await sleep(delay)

    async def _read_frames(self, process: asyncio.subprocess.Process) -> int:
        rest, count = b"", 0
        while True:
            try:
                chunk = await asyncio.wait_for(
                    process.stdout.read(FFMPEG_READ_BYTES), STALL_SECONDS
                )
            except TimeoutError:
                logger.warning("The RTSP camera sent nothing for %.0f s", STALL_SECONDS)
                return count
            if not chunk:
                return count
            frames, rest = split_jpeg_frames(rest + chunk)
            if frames:
                self._latest, self._taken_at = frames[-1], time.monotonic()
                count += len(frames)
            if len(rest) > MAX_FRAME_BYTES:
                rest = b""
