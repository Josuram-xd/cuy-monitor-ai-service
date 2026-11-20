import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from app.vision.frame_sink import BufferedFrameSink, frame_sink_for
from app.vision.window import FrameWindows

NOW = datetime(2026, 10, 5, 14, 32, tzinfo=UTC)


def test_buffered_sink_keeps_the_frame_in_the_cage_window_and_accepts_it():
    windows = FrameWindows()

    accepted = asyncio.run(BufferedFrameSink(windows).submit_frame("cage-1", b"frame", NOW))

    assert accepted is True
    assert len(windows.for_cage("cage-1")) == 1
    assert len(windows.for_cage("cage-2")) == 0


def test_the_trained_engine_is_the_sink_when_it_is_loaded():
    engine = SimpleNamespace(behavior_engine=object(), frame_windows=FrameWindows())

    assert frame_sink_for(engine) is engine.behavior_engine


def test_without_the_engine_frames_are_buffered_for_bedrock():
    windows = FrameWindows()
    state = SimpleNamespace(behavior_engine=None, frame_windows=windows)

    sink = frame_sink_for(state)

    assert isinstance(sink, BufferedFrameSink)
    asyncio.run(sink.submit_frame("cage-1", b"frame", NOW))
    assert len(windows.for_cage("cage-1")) == 1
