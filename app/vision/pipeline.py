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
from app.vision.window import FrameWindow

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


async def analyze_window_once(
    client: BackendEventClient,
    analyzer: BedrockBehaviorAnalyzer,
    window: FrameWindow,
    settings: Settings,
) -> int:
    """Look at the frames gathered so far and send one BEHAVIOR event per guinea pig seen."""
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
                cage_id=settings.cage_id,
                timestamp=timestamp,
                source="ai-service",
                schema_version=1,
                payload=payload,
            )
        )
    logger.info("Window analysed: %s frames, %s guinea pigs", len(frames), len(payloads))
    return len(payloads)


async def run_behavior_pipeline(
    client: BackendEventClient,
    analyzer: BedrockBehaviorAnalyzer,
    window: FrameWindow,
    settings: Settings,
    sleep: Sleep = asyncio.sleep,
) -> None:
    while True:
        await sleep(settings.window_seconds)
        await analyze_window_once(client, analyzer, window, settings)
