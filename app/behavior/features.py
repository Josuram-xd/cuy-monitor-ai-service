"""Behavior of each guinea pig over one window, from its tracked samples (Task 9.3).

- stillSeconds: share of the observed time it did not move, scaled to the window (the same
  convention as the Bedrock engine, so the backend baseline works with both).
- feederVisits / watererVisits: times it entered the zone (a stay of several frames is one visit).
- avgGroupDistance: mean distance to the other guinea pigs seen in the same frames, as a fraction
  of the image diagonal; 0.5 (neutral) when it was never seen with another one.
- detectionConfidence: mean confidence x share of the window's frames it was seen in, so a
  guinea pig seen in few frames gets a low value and the backend can ignore that window.

Nothing here decides health: these are measurements (AGENTS rule 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.behavior.window import FrameSample, MarkSample
from app.contracts import BehaviorEventPayload, MarkColor
from app.vision.classes import DETECTOR_CLASSES

# moving less than this (fraction of the diagonal) between two frames counts as still
STILL_DISTANCE = 0.02
# a longer gap between two samples of a guinea pig is not counted: it was hidden, not still
MAX_GAP_SECONDS = 10.0
NEUTRAL_GROUP_DISTANCE = 0.5

# The input of the behavior classifier, in this order. Training (training/train_behavior.py) and
# inference read the same tuple: a model is never trained with one order and read with another.
FEATURE_NAMES: tuple[str, ...] = ("stillRatio", "feederVisits", "watererVisits", "avgGroupDistance")


@dataclass(frozen=True)
class BehaviorFeatures:
    color: MarkColor
    window_seconds: int
    still_seconds: float
    feeder_visits: int
    waterer_visits: int
    avg_group_distance: float
    detection_confidence: float
    frames_seen: int

    def vector(self) -> list[float]:
        """The classifier input, in FEATURE_NAMES order (still time as a share of the window)."""
        return [
            self.still_seconds / self.window_seconds,
            float(self.feeder_visits),
            float(self.waterer_visits),
            self.avg_group_distance,
        ]

    def to_payload(self, prob_anomaly: float) -> BehaviorEventPayload:
        return BehaviorEventPayload(
            color=self.color,
            window_seconds=self.window_seconds,
            still_seconds=self.still_seconds,
            feeder_visits=self.feeder_visits,
            waterer_visits=self.waterer_visits,
            avg_group_distance=self.avg_group_distance,
            prob_anomaly=round(min(1.0, max(0.0, prob_anomaly)), 2),
            detection_confidence=self.detection_confidence,
        )


def _still_ratio(track: list[tuple[float, MarkSample]]) -> float:
    observed = still = 0.0
    for (t0, a), (t1, b) in zip(track, track[1:], strict=False):
        gap = t1 - t0
        if gap <= 0 or gap > MAX_GAP_SECONDS:
            continue
        observed += gap
        if a.distance_to(b) < STILL_DISTANCE:
            still += gap
    return still / observed if observed else 0.0


def _visits(track: list[tuple[float, MarkSample]], zone: str) -> int:
    visits, inside = 0, False
    for _, sample in track:
        now = getattr(sample, zone)
        if now and not inside:
            visits += 1
        inside = now
    return visits


def _group_distance(frames: list[FrameSample], color: MarkColor) -> float:
    distances = []
    for frame in frames:
        mine = frame.mark(color)
        others = [m for m in frame.marks if m.color != color]
        if mine is not None and others:
            distances.append(sum(mine.distance_to(o) for o in others) / len(others))
    return sum(distances) / len(distances) if distances else NEUTRAL_GROUP_DISTANCE


def compute_features(frames: list[FrameSample], window_seconds: int) -> list[BehaviorFeatures]:
    """One BehaviorFeatures per guinea pig seen in the window, in class order."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not frames:
        return []
    start = frames[0].captured_at
    tracks: dict[MarkColor, list[tuple[float, MarkSample]]] = {}
    for frame in frames:
        offset = (frame.captured_at - start).total_seconds()
        for mark in frame.marks:
            tracks.setdefault(mark.color, []).append((offset, mark))

    features = []
    for color in sorted(tracks, key=DETECTOR_CLASSES.index):
        track = tracks[color]
        mean_confidence = sum(s.confidence for _, s in track) / len(track)
        seen_ratio = len(track) / len(frames)
        features.append(
            BehaviorFeatures(
                color=color,
                window_seconds=window_seconds,
                still_seconds=round(window_seconds * _still_ratio(track), 1),
                feeder_visits=_visits(track, "at_feeder"),
                waterer_visits=_visits(track, "at_waterer"),
                avg_group_distance=round(_group_distance(frames, color), 2),
                detection_confidence=round(min(1.0, seen_ratio * mean_confidence), 2),
                frames_seen=len(track),
            )
        )
    return features
