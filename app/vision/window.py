from collections import deque

# a minute at 2 fps is 120 frames; keep a bit more than that, never grow without limit
MAX_BUFFERED_FRAMES = 240


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
