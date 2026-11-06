import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from app.bedrock.runtime import BedrockError
from app.config import Settings
from app.contracts import BehaviorEventPayload, EventEnvelope, EventType
from app.messaging.backend_client import BackendEventClient
from app.vision.analysis import BedrockBehaviorAnalyzer
from app.vision.window import FrameWindow, FrameWindows

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


class BehaviorEngine(Protocol):
    def cage_ids(self) -> list[str]: ...

    async def close_window(self, cage_id: str) -> list[BehaviorEventPayload]: ...


async def send_behavior_events(
    client: BackendEventClient, payloads: list[BehaviorEventPayload], cage_id: str
) -> None:
    """One BEHAVIOR event per payload, all with the time the window closed."""
    timestamp = datetime.now(UTC)
    for payload in payloads:
        await client.enqueue_event(
            EventEnvelope(
                event_id=uuid4(),
                type=EventType.BEHAVIOR,
                cage_id=cage_id,
                timestamp=timestamp,
                source="ai-service",
                schema_version=1,
                payload=payload,
            )
        )


async def close_trained_window_once(
    client: BackendEventClient, engine: BehaviorEngine, cage_id: str
) -> int:
    """Send the BEHAVIOR events of the trained chain for the window of a cage that just ended."""
    try:
        payloads = await engine.close_window(cage_id)
    except Exception:  # one bad window is lost, the loop and the next window go on
        logger.exception("Skipping a window of %s", cage_id)
        return 0
    await send_behavior_events(client, payloads, cage_id)
    logger.info("Window of %s closed: %s guinea pigs", cage_id, len(payloads))
    return len(payloads)


async def run_trained_behavior_pipeline(
    client: BackendEventClient,
    engine: BehaviorEngine,
    settings: Settings,
    sleep: Sleep = asyncio.sleep,
) -> None:
    while True:
        await sleep(settings.window_seconds)
        for cage_id in engine.cage_ids():
            await close_trained_window_once(client, engine, cage_id)


async def analyze_window_once(
    client: BackendEventClient,
    analyzer: BedrockBehaviorAnalyzer,
    window: FrameWindow,
    settings: Settings,
    cage_id: str | None = None,
) -> int:
    """Look at the frames gathered so far and send one BEHAVIOR event per guinea pig seen.

    cage_id is the cage the frames came from (settings.cage_id when not given)."""
    frames = window.drain(settings.window_max_frames)
    if not frames:
        return 0
    try:
        payloads = await analyzer.analyze(frames)
    except BedrockError as exception:
        # the window is lost, the next one starts clean: better than reporting made-up data
        logger.error("Skipping a window: %s", exception)
        return 0
    await send_behavior_events(client, payloads, cage_id or settings.cage_id)
    logger.info(
        "Window of %s analysed: %s frames, %s guinea pigs",
        cage_id or settings.cage_id,
        len(frames),
        len(payloads),
    )
    return len(payloads)


async def run_behavior_pipeline(
    client: BackendEventClient,
    analyzer: BedrockBehaviorAnalyzer,
    windows: FrameWindows,
    settings: Settings,
    sleep: Sleep = asyncio.sleep,
) -> None:
    while True:
        await sleep(settings.window_seconds)
        # every cage is analysed on its own: one cage's guinea pigs never end up in another's events
        for cage_id, window in windows.items():
            await analyze_window_once(client, analyzer, window, settings, cage_id)
