"""What the tracker saw in each cage since the last BEHAVIOR events (Task 9.2).

Every analysed frame adds one FrameSample with the guinea pigs seen in it. When the window closes
(every WINDOW_SECONDS) it is drained and the features of each guinea pig are computed from its
samples. Old frames are dropped if the window is not closed in time: it never grows without limit.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from app.contracts import MarkColor
from app.vision.tracker import TrackedMark
from app.vision.zones import CageZones

# a minute at 2 fps is 120 frames; keep a bit more than that
MAX_WINDOW_FRAMES = 240


@dataclass(frozen=True)
class MarkSample:
    """One guinea pig in one frame, already placed in the cage."""

    color: MarkColor
    confidence: float
    # centre of the box divided by the image diagonal: distances between samples are then a
    # fraction of the diagonal (0-1), as the contract asks for avgGroupDistance
    x: float
    y: float
    at_feeder: bool
    at_waterer: bool

    def distance_to(self, other: MarkSample) -> float:
        return min(1.0, math.hypot(self.x - other.x, self.y - other.y))


@dataclass(frozen=True)
class FrameSample:
    captured_at: datetime
    marks: tuple[MarkSample, ...]

    def mark(self, color: MarkColor) -> MarkSample | None:
        return next((m for m in self.marks if m.color == color), None)


def sample_frame(
    captured_at: datetime,
    marks: list[TrackedMark],
    frame_width: int,
    frame_height: int,
    zones: CageZones,
) -> FrameSample:
    """Tracker output of one frame -> the sample the window keeps."""
    diagonal = math.hypot(frame_width, frame_height)
    samples = []
    for mark in marks:
        cx, cy = mark.center
        at_feeder, at_waterer = zones.locate(cx / frame_width, cy / frame_height)
        samples.append(
            MarkSample(
                color=mark.color,
                confidence=mark.confidence,
                x=cx / diagonal,
                y=cy / diagonal,
                at_feeder=at_feeder,
                at_waterer=at_waterer,
            )
        )
    return FrameSample(captured_at, tuple(samples))


class BehaviorWindow:
    def __init__(self, max_frames: int = MAX_WINDOW_FRAMES):
        self._frames: deque[FrameSample] = deque(maxlen=max_frames)

    def __len__(self) -> int:
        return len(self._frames)

    def add(self, frame: FrameSample) -> None:
        self._frames.append(frame)

    def drain(self) -> list[FrameSample]:
        """Empty the window; frames in capture order (they can arrive slightly out of order)."""
        frames = sorted(self._frames, key=lambda f: f.captured_at)
        self._frames.clear()
        return frames


class BehaviorWindows:
    """One BehaviorWindow per cage, created with its first frame."""

    def __init__(self, max_frames: int = MAX_WINDOW_FRAMES):
        self._max_frames = max_frames
        self._windows: dict[str, BehaviorWindow] = {}

    def for_cage(self, cage_id: str) -> BehaviorWindow:
        if cage_id not in self._windows:
            self._windows[cage_id] = BehaviorWindow(self._max_frames)
        return self._windows[cage_id]

    def cage_ids(self) -> list[str]:
        return list(self._windows)
