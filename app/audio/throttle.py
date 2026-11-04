import time
from collections.abc import Callable

Clock = Callable[[], float]


class CalmAudioThrottle:
    """Lets one calm (NORMAL) clip per cage through every window, so the backend is not flooded with
    "all quiet" events. Distress clips never go through here: every one of them is reported."""

    def __init__(self, window_seconds: int, clock: Clock = time.monotonic):
        self._window_seconds = window_seconds
        self._clock = clock
        self._last_reported_at: dict[str, float] = {}

    def should_report(self, cage_id: str) -> bool:
        now = self._clock()
        last = self._last_reported_at.get(cage_id)
        if last is not None and now - last < self._window_seconds:
            return False
        self._last_reported_at[cage_id] = now
        return True
