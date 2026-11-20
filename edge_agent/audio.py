"""Task 12.3: the A12 microphone as ~1 s WAV clips.

IP Webcam streams the microphone as one endless WAV at /audio.wav: a RIFF header (with unknown
sizes) and then raw PCM forever. The header is read once, then the PCM is cut into clips of N
seconds and each clip is wrapped in its own small WAV, which is what POST /ai/v1/audio expects.
"""

import io
import struct
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from edge_agent.rtsp import FFMPEG_READ_BYTES, start_ffmpeg, stop_ffmpeg


class AudioUnavailable(Exception):
    """The phone stream ended, failed, or is not 16-bit PCM WAV."""


@dataclass(frozen=True)
class PcmFormat:
    channels: int
    sample_rate: int
    sample_width: int

    @property
    def bytes_per_second(self) -> int:
        return self.channels * self.sample_rate * self.sample_width


# what ffmpeg writes for an RTSP camera: mono 16 kHz 16-bit PCM, the format the service takes
RTSP_PCM = PcmFormat(channels=1, sample_rate=16_000, sample_width=2)


def parse_wav_header(header: bytes) -> tuple[PcmFormat, int] | None:
    """(format, offset where the PCM starts) once the header is complete; None if more bytes are
    needed."""
    if len(header) < 12:
        return None
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise AudioUnavailable("the phone audio is not a WAV stream")
    offset, pcm = 12, None
    while offset + 8 <= len(header):
        chunk_id = header[offset : offset + 4]
        (size,) = struct.unpack("<I", header[offset + 4 : offset + 8])
        body = offset + 8
        if chunk_id == b"fmt ":
            if len(header) < body + 16:
                return None
            audio_format, channels, rate, _, _, bits = struct.unpack(
                "<HHIIHH", header[body : body + 16]
            )
            if audio_format != 1 or bits != 16:
                raise AudioUnavailable("the phone audio must be 16-bit PCM")
            pcm = PcmFormat(channels, rate, 2)
        elif chunk_id == b"data":
            if pcm is None:
                raise AudioUnavailable("the WAV stream has no format before its data")
            # the size of an endless stream is meaningless (0 or 0xFFFFFFFF): the data goes on
            # forever
            return pcm, body
        offset = body + size + (size & 1)
    return None


def to_wav(pcm: bytes, fmt: PcmFormat) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(fmt.channels)
        clip.setsampwidth(fmt.sample_width)
        clip.setframerate(fmt.sample_rate)
        clip.writeframes(pcm)
    return buffer.getvalue()


async def pcm_clips(chunks: AsyncIterator[bytes], fmt: PcmFormat, clip_seconds: float):
    """Raw PCM of a known format -> one WAV file every clip_seconds of audio."""
    buffer = b""
    frame_bytes = fmt.channels * fmt.sample_width
    clip_bytes = int(fmt.bytes_per_second * clip_seconds) // frame_bytes * frame_bytes
    async for chunk in chunks:
        buffer += chunk
        while len(buffer) >= clip_bytes:
            yield to_wav(buffer[:clip_bytes], fmt)
            buffer = buffer[clip_bytes:]


async def wav_clips(chunks: AsyncIterator[bytes], clip_seconds: float) -> AsyncIterator[bytes]:
    """The bytes of an endless WAV stream -> one complete WAV file every clip_seconds of audio."""
    buffer = b""
    fmt: PcmFormat | None = None
    async for chunk in chunks:
        buffer += chunk
        if fmt is None:
            parsed = parse_wav_header(buffer)
            if parsed is None:
                if len(buffer) > 64 * 1024:
                    raise AudioUnavailable("the WAV header never ended")
                continue
            fmt, start = parsed
            buffer = buffer[start:]
        frame_bytes = fmt.channels * fmt.sample_width
        clip_bytes = int(fmt.bytes_per_second * clip_seconds) // frame_bytes * frame_bytes
        while len(buffer) >= clip_bytes:
            yield to_wav(buffer[:clip_bytes], fmt)
            buffer = buffer[clip_bytes:]


class A12Microphone:
    def __init__(self, client: httpx.AsyncClient, base_url: str):
        self._client = client
        self._url = f"{base_url}/audio.wav"

    async def clips(self, clip_seconds: float) -> AsyncIterator[bytes]:
        try:
            async with self._client.stream(
                "GET", self._url, timeout=httpx.Timeout(10, read=30)
            ) as r:
                if r.status_code != 200:
                    raise AudioUnavailable(f"the phone answered HTTP {r.status_code}")
                async for clip in wav_clips(r.aiter_bytes(), clip_seconds):
                    yield clip
        except httpx.HTTPError as exception:
            raise AudioUnavailable(f"the phone audio stream failed: {exception}") from exception
        raise AudioUnavailable("the phone audio stream ended")


class RtspMicrophone:
    """The audio of an RTSP camera, read by one ffmpeg as raw PCM and cut like the phone clips."""

    def __init__(self, url: str):
        self._url = url

    async def clips(self, clip_seconds: float) -> AsyncIterator[bytes]:
        try:
            process = await start_ffmpeg(
                self._url, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"
            )
        except OSError as exception:
            raise AudioUnavailable(f"ffmpeg could not start: {exception}") from exception

        async def chunks() -> AsyncIterator[bytes]:
            while chunk := await process.stdout.read(FFMPEG_READ_BYTES):
                yield chunk

        try:
            async for clip in pcm_clips(chunks(), RTSP_PCM, clip_seconds):
                yield clip
        finally:
            await stop_ffmpeg(process)
        raise AudioUnavailable("the camera audio stream ended")
