"""Task 9.4: the features of a window, computed from synthetic tracks (no model involved)."""

import math
from datetime import UTC, datetime, timedelta

import pytest

from app.behavior.features import FEATURE_NAMES, NEUTRAL_GROUP_DISTANCE, compute_features
from app.behavior.window import BehaviorWindow, BehaviorWindows, FrameSample, sample_frame
from app.contracts import MarkColor
from app.vision.tracker import TrackedMark
from app.vision.zones import CageZones, Zone

W, H = 640, 480
DIAGONAL = math.hypot(W, H)
START = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
FEEDER = Zone(((0.0, 0.0), (0.25, 0.0), (0.25, 0.25), (0.0, 0.25)))
WATERER = Zone(((0.75, 0.75), (1.0, 0.75), (1.0, 1.0), (0.75, 1.0)))
ZONES = CageZones(feeder=FEEDER, waterer=WATERER)


def _mark(color: MarkColor, x: float, y: float, confidence: float = 0.9) -> TrackedMark:
    return TrackedMark(1, color, confidence, (x - 10, y - 10, x + 10, y + 10))


def _frames(track: dict[MarkColor, list[tuple[float, float] | None]], step: float = 0.5):
    """One FrameSample per position; None = not seen in that frame."""
    length = max(len(p) for p in track.values())
    frames = []
    for i in range(length):
        marks = [
            _mark(color, *positions[i])
            for color, positions in track.items()
            if i < len(positions) and positions[i] is not None
        ]
        frames.append(sample_frame(START + timedelta(seconds=i * step), marks, W, H, ZONES))
    return frames


def test_a_guinea_pig_that_never_moves_is_still_the_whole_window():
    frames = _frames({MarkColor.RED: [(320, 240)] * 120})

    [red] = compute_features(frames, 60)

    assert red.still_seconds == 60
    assert red.feeder_visits == red.waterer_visits == 0
    assert red.avg_group_distance == NEUTRAL_GROUP_DISTANCE  # alone in view
    assert red.detection_confidence == 0.9
    assert red.frames_seen == 120


def test_a_guinea_pig_that_keeps_moving_is_never_still():
    # 40 px back and forth every frame: well over STILL_DISTANCE
    frames = _frames({MarkColor.BLUE: [(100 if i % 2 else 140, 240) for i in range(120)]})

    [blue] = compute_features(frames, 60)

    assert blue.still_seconds == 0


def test_still_time_is_the_share_of_the_observed_time_scaled_to_the_window():
    # moves during the first half, then rests
    positions = [(100 if i % 2 else 140, 240) for i in range(61)] + [(140, 240)] * 59
    frames = _frames({MarkColor.RED: positions})

    [red] = compute_features(frames, 60)

    assert red.still_seconds == pytest.approx(29.5, abs=1)


def test_a_long_gap_is_not_counted_as_still_time():
    positions = [(320, 240)] * 3 + [None] * 40 + [(330, 240)] * 3
    frames = _frames({MarkColor.RED: positions})

    [red] = compute_features(frames, 60)

    # the guinea pig was hidden for 20 s: only the time it was seen counts
    assert red.still_seconds == 60  # still in every pair it was seen in
    assert red.detection_confidence == round(0.9 * 6 / 46, 2)


def test_feeder_and_waterer_visits_count_entries_not_frames():
    outside, feeder, waterer = (320, 240), (60, 50), (600, 450)
    positions = [outside, feeder, feeder, feeder, outside, feeder, outside, waterer, waterer]
    frames = _frames({MarkColor.GREEN: positions})

    [green] = compute_features(frames, 60)

    assert green.feeder_visits == 2
    assert green.waterer_visits == 1


def test_group_distance_is_a_fraction_of_the_diagonal():
    frames = _frames({MarkColor.RED: [(0 + 10, 240)] * 4, MarkColor.BLUE: [(320 + 10, 240)] * 4})

    red, blue = compute_features(frames, 60)

    expected = round(320 / DIAGONAL, 2)
    assert red.avg_group_distance == blue.avg_group_distance == expected


def test_group_distance_only_counts_frames_where_both_were_seen():
    frames = _frames(
        {
            MarkColor.RED: [(10, 240), (10, 240), (10, 240)],
            MarkColor.BLUE: [None, None, (330, 240)],
        }
    )

    red, _ = compute_features(frames, 60)

    assert red.avg_group_distance == round(320 / DIAGONAL, 2)


def test_a_rarely_seen_guinea_pig_gets_a_low_confidence():
    frames = _frames({MarkColor.RED: [(320, 240)] * 100, MarkColor.WHITE: [(50, 50)] * 10})

    _, white = compute_features(frames, 60)

    assert white.detection_confidence == round(0.9 * 10 / 100, 2)
    assert white.detection_confidence < 0.5  # under the backend's minDetectionConfidence


def test_features_give_a_contract_valid_payload_and_the_classifier_vector():
    [red] = compute_features(_frames({MarkColor.RED: [(320, 240)] * 10}), 60)

    payload = red.to_payload(0.873)

    assert payload.color is MarkColor.RED
    assert payload.window_seconds == 60
    assert payload.prob_anomaly == 0.87
    assert len(red.vector()) == len(FEATURE_NAMES)
    assert red.vector()[0] == pytest.approx(red.still_seconds / 60)


def test_an_empty_window_gives_nothing_and_a_bad_window_length_is_refused():
    assert compute_features([], 60) == []
    with pytest.raises(ValueError):
        compute_features([], 0)


def test_the_window_returns_frames_in_capture_order_and_starts_empty():
    window = BehaviorWindow()
    late = FrameSample(START + timedelta(seconds=2), ())
    early = FrameSample(START, ())
    window.add(late)
    window.add(early)

    assert window.drain() == [early, late]
    assert len(window) == 0


def test_the_window_is_bounded_and_there_is_one_per_cage():
    windows = BehaviorWindows(max_frames=3)
    for i in range(5):
        windows.for_cage("cage-1").add(FrameSample(START + timedelta(seconds=i), ()))
    windows.for_cage("cage-2")

    assert len(windows.for_cage("cage-1")) == 3
    assert windows.cage_ids() == ["cage-1", "cage-2"]
