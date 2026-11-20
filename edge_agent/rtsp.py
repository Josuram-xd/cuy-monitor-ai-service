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


async def probe_rtsp(url: str, timeout: float = 15.0) -> str:
    """CONNECTED, AUTH_FAILED, UNREACHABLE or FFMPEG_MISSING. ffmpeg's stderr is read but never
    logged (it can contain the password)."""
    try:
        require_ffmpeg()
    except FfmpegMissing:
        return "FFMPEG_MISSING"
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            url,
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError:
        return "FFMPEG_MISSING"
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        await stop_ffmpeg(process)
        return "UNREACHABLE"
    if process.returncode == 0:
        return "CONNECTED"
    return "AUTH_FAILED" if b"401" in stderr else "UNREACHABLE"
