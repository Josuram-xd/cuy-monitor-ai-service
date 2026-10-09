import asyncio

import pytest

from app.bedrock.runtime import BedrockError
from app.config import Settings
from app.contracts import BehaviorEventPayload, EventType, MarkColor
from app.vision.pipeline import analyze_window_once, run_behavior_pipeline
from app.vision.window import FrameWindow


def _settings() -> Settings:
    return Settings(
        _env_file=None, api_key="k", cage_id="cage-1", window_seconds=60, mock_mode=False
    )


def _payload(color: MarkColor) -> BehaviorEventPayload:
    return BehaviorEventPayload(
        color=color,
        window_seconds=60,
        still_seconds=30,
        feeder_visits=1,
        waterer_visits=0,
        avg_group_distance=0.4,
        prob_anomaly=0.2,
        detection_confidence=0.8,
    )


class _Analyzer:
    def __init__(self, payloads=None, error=None):
        self.payloads = payloads or []
        self.error = error
        self.seen: list[list[bytes]] = []

    async def analyze(self, frames):
        self.seen.append(frames)
        if self.error:
            raise self.error
        return self.payloads


class _Client:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


@pytest.mark.anyio
async def test_an_empty_window_does_not_call_the_model():
    analyzer, client = _Analyzer(), _Client()

    sent = await analyze_window_once(client, analyzer, FrameWindow(), _settings())

    assert sent == 0
    assert analyzer.seen == []


@pytest.mark.anyio
async def test_each_guinea_pig_seen_becomes_a_behavior_event_for_the_cage():
    window = FrameWindow()
    for frame in (b"1", b"2", b"3"):
        window.add(frame)
    analyzer = _Analyzer([_payload(MarkColor.RED), _payload(MarkColor.BLUE)])
    client = _Client()

    sent = await analyze_window_once(client, analyzer, window, _settings())

    assert sent == 2
    assert analyzer.seen == [[b"1", b"2", b"3"]]
    assert [e.payload.color for e in client.events] == [MarkColor.RED, MarkColor.BLUE]
    for event in client.events:
        assert event.type is EventType.BEHAVIOR
        assert event.cage_id == "cage-1"
        assert event.source == "ai-service"
    assert len({e.event_id for e in client.events}) == 2
    assert len(window) == 0


@pytest.mark.anyio
async def test_a_model_failure_loses_the_window_and_reports_nothing():
    window = FrameWindow()
    window.add(b"1")
    client = _Client()

    sent = await analyze_window_once(
        client, _Analyzer(error=BedrockError("down")), window, _settings()
    )

    assert sent == 0
    assert client.events == []
    assert len(window) == 0


@pytest.mark.anyio
async def test_the_loop_waits_a_window_then_analyses():
    window = FrameWindow()
    window.add(b"1")
    analyzer, client = _Analyzer([_payload(MarkColor.GREEN)]), _Client()
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)
        if len(sleeps) == 2:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_behavior_pipeline(client, analyzer, window, _settings(), sleep=sleep)

    assert sleeps == [60, 60]
    assert len(client.events) == 1
