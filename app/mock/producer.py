import asyncio
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

from app.config import Settings
from app.contracts import (
    AudioEventPayload,
    BehaviorEventPayload,
    EventEnvelope,
    EventType,
    MarkColor,
)
from app.contracts.events import AudioLabel
from app.messaging.backend_client import BackendEventClient

Sleep = Callable[[float], Awaitable[None]]


def create_mock_events(
    settings: Settings, cage_id: str | None = None
) -> tuple[EventEnvelope, EventEnvelope]:
    cage_id = cage_id or settings.cage_id
    timestamp = datetime.now(UTC)
    behavior_event = EventEnvelope(
        event_id=uuid4(),
        type=EventType.BEHAVIOR,
        cage_id=cage_id,
        timestamp=timestamp,
        source="ai-service",
        schema_version=1,
        payload=BehaviorEventPayload(
            color=random.choice(list(MarkColor)),
            window_seconds=settings.window_seconds,
            still_seconds=round(random.uniform(0, settings.window_seconds), 2),
            feeder_visits=random.randint(0, 3),
            waterer_visits=random.randint(0, 3),
            avg_group_distance=round(random.random(), 2),
            prob_anomaly=round(random.random(), 2),
            detection_confidence=round(random.random(), 2),
        ),
    )
    audio_event = EventEnvelope(
        event_id=uuid4(),
        type=EventType.AUDIO,
        cage_id=cage_id,
        timestamp=timestamp,
        source="ai-service",
        schema_version=1,
        payload=AudioEventPayload(
            label=random.choice(list(AudioLabel)),
            probability=round(random.random(), 2),
            duration_ms=960,
        ),
    )
    return behavior_event, audio_event


async def run_mock_producer(
    client: BackendEventClient,
    settings: Settings,
    sleep: Sleep = asyncio.sleep,
) -> None:
    while True:
        await sleep(settings.window_seconds)
        for cage_id in settings.allowed_cages:
            for event in create_mock_events(settings, cage_id):
                await client.enqueue_event(event)
