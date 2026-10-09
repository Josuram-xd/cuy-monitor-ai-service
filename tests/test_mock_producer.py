import asyncio

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.contracts import AudioEventPayload, BehaviorEventPayload, EventType
from app.mock.producer import run_mock_producer


def test_mock_producer_enqueues_behavior_and_audio_events_each_window():
    async def run_test():
        settings = Settings(
            api_key=SecretStr("test-secret"),
            cage_id="cage-test",
            window_seconds=60,
            _env_file=None,
        )

        class RecordingClient:
            def __init__(self):
                self.events = []

            async def enqueue_event(self, event):
                self.events.append(event)

        client = RecordingClient()
        sleep_count = 0

        async def stop_after_one_window(delay: float) -> None:
            nonlocal sleep_count
            assert delay == settings.window_seconds
            sleep_count += 1
            if sleep_count == 2:
                raise StopAsyncIteration

        with pytest.raises(StopAsyncIteration):
            await run_mock_producer(client, settings, sleep=stop_after_one_window)

        assert len(client.events) == 2
        behavior_event, audio_event = client.events
        assert behavior_event.type is EventType.BEHAVIOR
        assert isinstance(behavior_event.payload, BehaviorEventPayload)
        assert behavior_event.cage_id == settings.cage_id
        assert behavior_event.payload.window_seconds == settings.window_seconds
        assert 0 <= behavior_event.payload.still_seconds <= settings.window_seconds
        assert audio_event.type is EventType.AUDIO
        assert isinstance(audio_event.payload, AudioEventPayload)
        assert audio_event.cage_id == settings.cage_id
        assert audio_event.payload.duration_ms == 960

    asyncio.run(run_test())
