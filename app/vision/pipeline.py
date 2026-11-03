import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

from app.bedrock.runtime import BedrockError
from app.config import Settings
from app.contracts import EventEnvelope, EventType
from app.messaging.backend_client import BackendEventClient
from app.vision.analysis import BedrockBehaviorAnalyzer
from app.vision.window import FrameWindow, FrameWindows

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


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
    timestamp = datetime.now(UTC)
    for payload in payloads:
        await client.enqueue_event(
            EventEnvelope(
                event_id=uuid4(),
                type=EventType.BEHAVIOR,
                cage_id=cage_id or settings.cage_id,
                timestamp=timestamp,
                source="ai-service",
                schema_version=1,
                payload=payload,
            )
        )
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
