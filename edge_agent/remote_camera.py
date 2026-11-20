"""Task 12.6: the camera the breeder chose in the dashboard, read by the edge agent from the
backend.

GET /api/v1/ingestion/station/camera answers 200 with the camera, 204 when none is set (404 from an
older backend). Then the camera from edge_agent/.env is used. The status goes back with
POST /api/v1/ingestion/station/camera/status. Camera URLs can hold a password: never logged as
they are.
"""

import logging
import time
from dataclasses import dataclass

import httpx

from edge_agent.config import EdgeSettings

logger = logging.getLogger("edge_agent")

CAMERA_PATH = "/api/v1/ingestion/station/camera"
# a frame this old still means the camera works (the status is reported every poll)
FRESH_FRAME_SECONDS = 30.0
URL_SCHEMES = {"PHONE": ("http://", "https://"), "RTSP": ("rtsp://",)}


@dataclass(frozen=True)
class CameraSource:
    kind: str  # "PHONE" | "RTSP"
    url: str  # http://ip:port  |  rtsp://user:pass@ip:554/stream2
    version: int | None = None  # None = from edge_agent/.env: nothing to report


class BackendUnreachable(Exception):
    """The backend could not give the camera now; the current one keeps running."""


def env_source(settings: EdgeSettings) -> CameraSource | None:
    if settings.camera_rtsp_url:
        return CameraSource("RTSP", settings.camera_rtsp_url)
    if settings.a12_url:
        return CameraSource("PHONE", settings.a12_url)
    return None


async def fetch_source(
    client: httpx.AsyncClient, base_url: str, api_key: str
) -> CameraSource | None:
    try:
        response = await client.get(f"{base_url}{CAMERA_PATH}", headers={"X-API-Key": api_key})
    except httpx.HTTPError as exception:
        raise BackendUnreachable(f"no connection: {exception}") from exception
    if response.status_code in (204, 404):
        return None
    if response.status_code == 401:
        raise BackendUnreachable("the backend rejected API_KEY (401): check edge_agent/.env")
    if response.status_code != 200:
        raise BackendUnreachable(f"the backend answered HTTP {response.status_code}")
    try:
        body = response.json()
        kind, url, version = body["type"], body["url"], body["version"]
    except ValueError, KeyError, TypeError:
        kind = url = version = None
    if (
        isinstance(kind, str)
        and kind in URL_SCHEMES
        and isinstance(url, str)
        and url.startswith(URL_SCHEMES[kind])
        and isinstance(version, int)
    ):
        return CameraSource(kind, url, version)
    # the body holds the password: say what is wrong, never print it
    logger.error("The backend sent a camera that cannot be used (type %r)", kind)
    return None


async def report_status(
    client: httpx.AsyncClient, base_url: str, api_key: str, version: int, status: str
) -> None:
    try:
        response = await client.post(
            f"{base_url}{CAMERA_PATH}/status",
            headers={"X-API-Key": api_key},
            json={"version": version, "status": status},
        )
    except httpx.HTTPError as exception:
        logger.warning("Could not report the camera status: %s", exception)
        return
    if response.is_error:
        logger.warning("The backend answered HTTP %s to the camera status", response.status_code)


def camera_status(camera, probed: str) -> str:
    """What the dashboard should show: CONNECTED while frames arrive, else what the probe found."""
    if (
        camera is not None
        and camera.frame_at
        and time.monotonic() - camera.frame_at <= FRESH_FRAME_SECONDS
    ):
        return "CONNECTED"
    return probed if probed != "CONNECTED" else "UNREACHABLE"
