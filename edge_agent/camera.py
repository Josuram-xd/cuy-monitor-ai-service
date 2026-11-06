"""Task 12.1: frames from the A12 running IP Webcam.

IP Webcam serves the last camera frame as a JPEG at /shot.jpg. Asking for one frame each tick is
simpler and lighter for an old Celeron than decoding the MJPEG stream at /video, and at 1-2 fps it
gives the same frames.
"""

import httpx

JPEG_START = b"\xff\xd8"


class CameraUnavailable(Exception):
    """The phone did not answer, or answered something that is not a JPEG."""


class A12Camera:
    def __init__(self, client: httpx.AsyncClient, base_url: str):
        self._client = client
        self._url = f"{base_url}/shot.jpg"

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
        return frame
