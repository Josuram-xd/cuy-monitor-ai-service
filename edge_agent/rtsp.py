"""ffmpeg for RTSP cameras (TP-Link Tapo): one long-lived process per stream, read from stdout.

ffmpeg's own stderr goes nowhere on purpose: it can print the RTSP URL with its password.
"""

import asyncio
import shutil

FFMPEG_READ_BYTES = 64 * 1024


class FfmpegMissing(RuntimeError):
    """ffmpeg is not on PATH, so an RTSP camera cannot be read."""


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise FfmpegMissing(
            "ffmpeg is not on PATH: install it (Windows: winget install Gyan.FFmpeg; "
            "Linux: apt install ffmpeg) or leave CAMERA_RTSP_URL empty to use the phone"
        )


async def start_ffmpeg(url: str, *output_args: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-loglevel",
        "quiet",
        "-rtsp_transport",
        "tcp",
        "-i",
        url,
        *output_args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )


async def stop_ffmpeg(process: asyncio.subprocess.Process) -> int | None:
    if process.returncode is None:
        process.kill()
    return await process.wait()
