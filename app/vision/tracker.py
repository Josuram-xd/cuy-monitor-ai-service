"""Keeps the identity of every marked guinea pig from one frame to the next (Task 8).

The detector already says one colour per box, but at 1-2 fps and with bedding, shadows and two
guinea pigs side by side it sometimes misreads a mark for a frame or misses it for a few. The
tracker smooths that out:

- each track is one guinea pig; its colour is a vote over its recent detections (weighted by
  confidence, older votes fading), so a single misread frame never renames it;
- detections are assigned to tracks with the Hungarian algorithm
  (``scipy.optimize.linear_sum_assignment``), the cheapest overall pairing;
- a detection of the same colour as a track can always join it (a guinea pig can cross the cage
  between two frames); one of another colour only when it is close (a misread mark);
- a track survives a few frames without detections (hidden behind the house or the feeder).

The colour reported is never invented: it is always one the detector gave for that track, and the
confidence goes down when the votes of the track disagree.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.contracts import MarkColor
from app.vision.classes import DETECTOR_CLASSES
from app.vision.detector import Detection

# how far (fraction of the image diagonal) a box with another colour can be from a track and still
# be taken as that guinea pig with a misread mark
DEFAULT_MAX_JUMP = 0.15
# frames a track survives without a detection: 3 s at 2 fps
DEFAULT_MAX_MISSES = 6
# every new frame the old votes count this much less, so the colour can recover from bad frames
DEFAULT_VOTE_DECAY = 0.8
# joining a track of another colour costs more than any distance, so the same colour always wins
_COLOR_MISMATCH_COST = 2.0
_FORBIDDEN = 1e6


@dataclass(frozen=True)
class TrackedMark:
    """One guinea pig seen in this frame, with the identity the tracker keeps for it."""

    track_id: int
    color: MarkColor
    confidence: float
    box: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class _Track:
    track_id: int
    box: tuple[float, float, float, float]
    votes: Counter[MarkColor] = field(default_factory=Counter)
    misses: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def color(self) -> MarkColor:
        # ties go to the class order, so the result never depends on dict order
        return max(self.votes, key=lambda c: (self.votes[c], -DETECTOR_CLASSES.index(c)))

    @property
    def agreement(self) -> float:
        """Share of the votes that agree with the colour of the track (1 = all of them)."""
        total = sum(self.votes.values())
        return self.votes[self.color] / total if total else 0.0

    def absorb(self, detection: Detection, decay: float) -> None:
        for color in self.votes:
            self.votes[color] *= decay
        self.votes[detection.color] += max(detection.confidence, 1e-3)
        self.box = detection.box
        self.misses = 0


class HungarianTracker:
    """One per cage. update() gets the detections of a frame and returns who is who."""

    def __init__(
        self,
        max_jump: float = DEFAULT_MAX_JUMP,
        max_misses: int = DEFAULT_MAX_MISSES,
        vote_decay: float = DEFAULT_VOTE_DECAY,
    ):
        if not 0 < max_jump <= 1:
            raise ValueError("max_jump must be a fraction of the diagonal (0-1]")
        if max_misses < 0:
            raise ValueError("max_misses cannot be negative")
        if not 0 < vote_decay <= 1:
            raise ValueError("vote_decay must be in (0, 1]")
        self._max_jump = max_jump
        self._max_misses = max_misses
        self._vote_decay = vote_decay
        self._tracks: list[_Track] = []
        self._next_id = 1

    def __len__(self) -> int:
        return len(self._tracks)

    def update(
        self, detections: list[Detection], frame_width: int, frame_height: int
    ) -> list[TrackedMark]:
        """The guinea pigs seen in this frame, at most one per colour."""
        diagonal = math.hypot(frame_width, frame_height)
        if diagonal <= 0:
            raise ValueError("The frame has no size")

        matched: list[tuple[_Track, Detection]] = []
        free_detections = list(detections)
        if self._tracks and detections:
            cost = self._cost_matrix(detections, diagonal)
            rows, cols = linear_sum_assignment(cost)
            taken = set()
            for row, col in zip(rows, cols, strict=True):
                if cost[row, col] < _FORBIDDEN:
                    matched.append((self._tracks[row], detections[col]))
                    taken.add(col)
            free_detections = [d for i, d in enumerate(detections) if i not in taken]

        seen_tracks = set()
        for track, detection in matched:
            track.absorb(detection, self._vote_decay)
            seen_tracks.add(track.track_id)
        for detection in free_detections:
            track = _Track(self._next_id, detection.box)
            track.absorb(detection, self._vote_decay)
            self._next_id += 1
            self._tracks.append(track)
            matched.append((track, detection))
            seen_tracks.add(track.track_id)

        for track in self._tracks:
            if track.track_id not in seen_tracks:
                track.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= self._max_misses]

        return self._report(matched)

    def _cost_matrix(self, detections: list[Detection], diagonal: float) -> np.ndarray:
        cost = np.full((len(self._tracks), len(detections)), _FORBIDDEN)
        for row, track in enumerate(self._tracks):
            tx, ty = track.center
            for col, detection in enumerate(detections):
                dx, dy = detection.center
                distance = math.hypot(dx - tx, dy - ty) / diagonal
                if detection.color == track.color:
                    cost[row, col] = distance
                elif distance <= self._max_jump:
                    cost[row, col] = distance + _COLOR_MISMATCH_COST
        return cost

    @staticmethod
    def _report(matched: list[tuple[_Track, Detection]]) -> list[TrackedMark]:
        best: dict[MarkColor, TrackedMark] = {}
        for track, detection in matched:
            # disagreeing votes mean the identity is less sure: say so in the confidence
            mark = TrackedMark(
                track_id=track.track_id,
                color=track.color,
                confidence=round(detection.confidence * track.agreement, 3),
                box=detection.box,
            )
            # a cage has one guinea pig per colour: two tracks voting the same colour keep the surer
            known = best.get(mark.color)
            if known is None or mark.confidence > known.confidence:
                best[mark.color] = mark
        return sorted(best.values(), key=lambda m: DETECTOR_CLASSES.index(m.color))
