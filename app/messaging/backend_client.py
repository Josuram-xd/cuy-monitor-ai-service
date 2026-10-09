import asyncio
import logging
from collections import deque
from enum import Enum, auto
from typing import Protocol

import httpx

from app.config import Settings
from app.contracts.events import EventEnvelope

logger = logging.getLogger(__name__)

BACKEND_EVENTS_PATH = "/api/v1/ingestion/events"
MAX_PENDING_EVENTS = 500
MAX_BACKOFF_SECONDS = 30


class _DeliveryResult(Enum):
    DELIVERED = auto()
    DROP = auto()
    RETRY = auto()


class AsyncSleep(Protocol):
    async def __call__(self, delay: float) -> None: ...


class BackendEventClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
        sleep: AsyncSleep = asyncio.sleep,
    ):
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            base_url=settings.backend_url,
            headers={"X-API-Key": settings.api_key.get_secret_value()},
            timeout=5.0,
        )
        self._sleep = sleep
        self._pending: deque[EventEnvelope] = deque()
        self._condition = asyncio.Condition()
        self._flush_lock = asyncio.Lock()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def enqueue_event(self, event: EventEnvelope) -> None:
        async with self._condition:
            if len(self._pending) == MAX_PENDING_EVENTS:
                dropped_event = self._pending.popleft()
                logger.warning("Dropping oldest buffered event id=%s", dropped_event.event_id)
            self._pending.append(event)
            self._condition.notify_all()

    async def flush_pending_once(self) -> bool:
        async with self._flush_lock:
            async with self._condition:
                if not self._pending:
                    return True
                event = self._pending[0]

            result = await self._deliver_once(event)
            if result is _DeliveryResult.RETRY:
                return False

            async with self._condition:
                if self._pending and self._pending[0].event_id == event.event_id:
                    self._pending.popleft()
                    self._condition.notify_all()
            return True

    async def run(self) -> None:
        backoff_seconds = 1
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: bool(self._pending))

            completed = await self.flush_pending_once()
            if completed:
                backoff_seconds = 1
                continue

            await self._sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, MAX_BACKOFF_SECONDS)

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()

    async def _deliver_once(self, event: EventEnvelope) -> _DeliveryResult:
        try:
            response = await self._http_client.post(
                BACKEND_EVENTS_PATH,
                json=event.model_dump(mode="json", by_alias=True),
            )
        except httpx.RequestError as exception:
            logger.warning("Backend request failed for event id=%s: %s", event.event_id, exception)
            return _DeliveryResult.RETRY

        if response.status_code == 202:
            logger.debug("Backend accepted event id=%s", event.event_id)
            return _DeliveryResult.DELIVERED
        if response.status_code >= 500:
            logger.warning(
                "Backend returned status=%s for event id=%s",
                response.status_code,
                event.event_id,
            )
            return _DeliveryResult.RETRY

        logger.error(
            "Backend rejected event id=%s with status=%s",
            event.event_id,
            response.status_code,
        )
        return _DeliveryResult.DROP
