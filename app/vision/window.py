from __future__ import annotations

from collections import deque

# a minute at 2 fps is 120 frames; keep a bit more than that, never grow without limit
MAX_BUFFERED_FRAMES = 240


class FrameWindows:
    """One FrameWindow per cage, created the first time a frame of that cage arrives."""

    def __init__(self, max_frames: int = MAX_BUFFERED_FRAMES):
        self._max_frames = max_frames
        self._windows: dict[str, FrameWindow] = {}

    def __len__(self) -> int:
        return sum(len(window) for window in self._windows.values())

    def for_cage(self, cage_id: str) -> FrameWindow:
        if cage_id not in self._windows:
            self._windows[cage_id] = FrameWindow(self._max_frames)
        return self._windows[cage_id]

    def items(self) -> list[tuple[str, FrameWindow]]:
        return list(self._windows.items())


class FrameWindow:
    """Frames received since the last analysis. Old ones are dropped if the loop falls behind."""

    def __init__(self, max_frames: int = MAX_BUFFERED_FRAMES):
        self._frames: deque[bytes] = deque(maxlen=max_frames)

    def __len__(self) -> int:
        return len(self._frames)

    def add(self, frame: bytes) -> None:
        self._frames.append(frame)

    def drain(self, sample_size: int) -> list[bytes]:
        """Empty the window and return up to sample_size frames evenly spread over it."""
        frames = list(self._frames)
        self._frames.clear()
        if len(frames) <= sample_size:
            return frames
        if sample_size == 1:
            return [frames[len(frames) // 2]]
        last = len(frames) - 1
        return [frames[round(i * last / (sample_size - 1))] for i in range(sample_size)]
