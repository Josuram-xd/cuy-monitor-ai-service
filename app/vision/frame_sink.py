from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from app.vision.window import FrameWindows


class FrameSink(Protocol):
    async def submit_frame(self, cage_id: str, frame: bytes, captured_at: datetime) -> bool: ...


class BufferedFrameSink:
    """Adapter: FrameWindows (frames kept for Bedrock) behind the engine's submit_frame call."""

    def __init__(self, windows: FrameWindows):
        self._windows = windows

    async def submit_frame(self, cage_id: str, frame: bytes, captured_at: datetime) -> bool:
        self._windows.for_cage(cage_id).add(frame)
        return True


def frame_sink_for(state: Any) -> FrameSink:
    """The trained behavior engine when it is loaded, else the frames buffered for Bedrock."""
    engine = getattr(state, "behavior_engine", None)
    return engine if engine is not None else BufferedFrameSink(state.frame_windows)
