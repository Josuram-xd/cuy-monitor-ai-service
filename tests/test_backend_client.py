import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.contracts import EventEnvelope
from app.messaging.backend_client import (
    BACKEND_EVENTS_PATH,
    MAX_PENDING_EVENTS,
    BackendEventClient,
)

EVENT_FIXTURE = Path(__file__).parent / "fixtures" / "backend-events" / "behavior.json"


def load_event() -> EventEnvelope:
    return EventEnvelope.model_validate_json(EVENT_FIXTURE.read_text(encoding="utf-8"))


def test_backend_client_posts_camel_case_event_with_api_key():
    async def run_test():
        requests = []

        async def handle_request(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(202)

        settings = Settings(api_key=SecretStr("test-secret"), _env_file=None)
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request),
            base_url=settings.backend_url,
            headers={"X-API-Key": settings.api_key.get_secret_value()},
        )
        client = BackendEventClient(settings, http_client=http_client)
        event = load_event()

        await client.enqueue_event(event)
        assert await client.flush_pending_once()

        assert client.pending_count == 0
        assert len(requests) == 1
        assert requests[0].url.path == BACKEND_EVENTS_PATH
        assert requests[0].headers["X-API-Key"] == "test-secret"
        assert json.loads(requests[0].read()) == event.model_dump(mode="json", by_alias=True)
        await http_client.aclose()

    asyncio.run(run_test())


@pytest.mark.parametrize("status_code", [400, 401])
def test_backend_client_drops_non_retryable_responses(status_code):
    async def run_test():
        attempts = 0

        async def handle_request(_: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(status_code)

        settings = Settings(api_key=SecretStr("test-secret"), _env_file=None)
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request),
            base_url=settings.backend_url,
        )
        client = BackendEventClient(settings, http_client=http_client)
        await client.enqueue_event(load_event())

        assert await client.flush_pending_once()
        assert attempts == 1
        assert client.pending_count == 0
        await http_client.aclose()

    asyncio.run(run_test())


def test_backend_client_retries_server_errors_with_same_event_id_and_backoff():
    async def run_test():
        attempts = 0
        accepted = asyncio.Event()
        observed_event_ids = []
        delays = []

        async def handle_request(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            observed_event_ids.append(json.loads(request.read())["eventId"])
            if attempts < 3:
                return httpx.Response(503)
            accepted.set()
            return httpx.Response(202)

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        settings = Settings(api_key=SecretStr("test-secret"), _env_file=None)
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request),
            base_url=settings.backend_url,
        )
        client = BackendEventClient(settings, http_client=http_client, sleep=record_sleep)
        event = load_event()
        await client.enqueue_event(event)
        worker = asyncio.create_task(client.run())
        try:
            await asyncio.wait_for(accepted.wait(), timeout=1)
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker

        assert observed_event_ids == [str(event.event_id)] * 3
        assert delays == [1, 2]
        assert client.pending_count == 0
        await http_client.aclose()

    asyncio.run(run_test())


def test_backend_client_retries_network_errors():
    async def run_test():
        attempts = 0

        async def handle_request(_: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("backend unavailable")
            return httpx.Response(202)

        settings = Settings(api_key=SecretStr("test-secret"), _env_file=None)
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request),
            base_url=settings.backend_url,
        )
        client = BackendEventClient(settings, http_client=http_client)
        await client.enqueue_event(load_event())

        assert not await client.flush_pending_once()
        assert client.pending_count == 1
        assert await client.flush_pending_once()
        assert attempts == 2
        assert client.pending_count == 0
        await http_client.aclose()

    asyncio.run(run_test())


def test_backend_client_drops_oldest_event_when_buffer_is_full():
    async def run_test():
        settings = Settings(api_key=SecretStr("test-secret"), _env_file=None)
        requests = []

        async def handle_request(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.read()))
            return httpx.Response(202)

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request),
            base_url=settings.backend_url,
        )
        client = BackendEventClient(settings, http_client=http_client)
        events = [
            load_event().model_copy(update={"event_id": uuid4()})
            for _ in range(MAX_PENDING_EVENTS + 1)
        ]

        for event in events:
            await client.enqueue_event(event)

        assert client.pending_count == MAX_PENDING_EVENTS
        assert await client.flush_pending_once()
        assert requests[0]["eventId"] == str(events[1].event_id)
        assert client.pending_count == MAX_PENDING_EVENTS - 1
        await http_client.aclose()

    asyncio.run(run_test())
