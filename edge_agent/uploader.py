"""Task 12.2: sends frames and clips to the AI service (POST /ai/v1/frames, /ai/v1/audio with X-API-Key).

There is no queue: a frame that could not be sent is dropped and the next one is a fresh frame, so
a long internet cut never piles up minutes of old frames on the laptop. What to do next is in the
result:

- SENT: 202, go on at the normal pace.
- RETRY: the network failed or the service answered 5xx / 429: wait with backoff and send a new one.
- REJECTED: 400 / 401 / 413 / 415: sending the same thing again will not help (wrong key, wrong
  cage...).
"""

import logging
from datetime import datetime
from enum import Enum, auto

import httpx

logger = logging.getLogger(__name__)


class SendResult(Enum):
    SENT = auto()
    RETRY = auto()
    REJECTED = auto()


def utc_iso(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AiServiceUploader:
    def __init__(self, client: httpx.AsyncClient, ai_url: str, api_key: str, cage_id: str):
        self._client = client
        self._ai_url = ai_url
        self._headers = {"X-API-Key": api_key}
        self._cage_id = cage_id

    async def send_frame(self, jpeg: bytes, captured_at: datetime) -> SendResult:
        return await self._send("/ai/v1/frames", ("frame.jpg", jpeg, "image/jpeg"), captured_at)

    async def send_audio(self, wav: bytes, captured_at: datetime) -> SendResult:
        return await self._send("/ai/v1/audio", ("clip.wav", wav, "audio/wav"), captured_at)

    async def _send(
        self, path: str, file: tuple[str, bytes, str], captured_at: datetime
    ) -> SendResult:
        try:
            response = await self._client.post(
                f"{self._ai_url}{path}",
                headers=self._headers,
                data={"capturedAt": utc_iso(captured_at), "cageId": self._cage_id},
                files={"file": file},
            )
        except httpx.HTTPError as exception:
            logger.warning("Could not reach the AI service (%s): %s", path, exception)
            return SendResult.RETRY
        if response.status_code == 202:
            return SendResult.SENT
        if response.status_code == 429 or response.status_code >= 500:
            logger.warning("The AI service answered %s on %s", response.status_code, path)
            return SendResult.RETRY
        # the key, the cage code or the file is wrong: the log says which, a person has to fix it
        logger.error(
            "The AI service rejected %s with %s: %s",
            path,
            response.status_code,
            response.text[:200],
        )
        return SendResult.REJECTED
