"""Waits that grow after each failure (1 s, 2 s, 4 s ... up to 30 s) and reset after a success."""


class Backoff:
    def __init__(self, first: float = 1.0, maximum: float = 30.0):
        if not 0 < first <= maximum:
            raise ValueError("first must be positive and not above maximum")
        self._first = first
        self._maximum = maximum
        self._next = first

    def next_delay(self) -> float:
        delay = self._next
        self._next = min(self._maximum, self._next * 2)
        return delay

    def reset(self) -> None:
        self._next = self._first
