import pytest

from app.contracts import MarkColor
from app.vision.detector import Detection
from app.vision.tracker import HungarianTracker

W, H = 640, 480


def _det(color: MarkColor, x: float, y: float, confidence: float = 0.9, size: float = 40):
    return Detection(color, confidence, (x - size / 2, y - size / 2, x + size / 2, y + size / 2))


def test_each_colour_keeps_its_track_across_frames():
    tracker = HungarianTracker()
    first = tracker.update([_det(MarkColor.RED, 100, 100), _det(MarkColor.BLUE, 400, 300)], W, H)
    second = tracker.update([_det(MarkColor.BLUE, 410, 305), _det(MarkColor.RED, 105, 98)], W, H)

    ids_first = {m.color: m.track_id for m in first}
    ids_second = {m.color: m.track_id for m in second}
    assert ids_first == ids_second
    assert [m.color for m in second] == [MarkColor.RED, MarkColor.BLUE]  # class order


def test_a_guinea_pig_that_crosses_the_cage_keeps_its_identity():
    tracker = HungarianTracker()
    start = tracker.update([_det(MarkColor.RED, 50, 50)], W, H)
    # at 1-2 fps a guinea pig can run far between two frames: the colour still matches it
    far = tracker.update([_det(MarkColor.RED, 600, 440)], W, H)

    assert far[0].track_id == start[0].track_id


def test_one_misread_frame_does_not_rename_the_guinea_pig():
    tracker = HungarianTracker()
    for x in range(100, 160, 10):
        tracker.update([_det(MarkColor.RED, x, 100)], W, H)

    misread = tracker.update([_det(MarkColor.ORANGE, 165, 102, confidence=0.6)], W, H)

    assert len(misread) == 1
    assert misread[0].color is MarkColor.RED
    # the vote disagreed, so the identity is less sure than the box itself
    assert misread[0].confidence < 0.6


def test_a_far_box_of_another_colour_is_a_new_guinea_pig():
    tracker = HungarianTracker()
    red = tracker.update([_det(MarkColor.RED, 50, 50)], W, H)[0]

    marks = tracker.update([_det(MarkColor.GREEN, 600, 440)], W, H)

    assert marks[0].color is MarkColor.GREEN
    assert marks[0].track_id != red.track_id


def test_a_hidden_guinea_pig_comes_back_with_the_same_id_within_max_misses():
    tracker = HungarianTracker(max_misses=3)
    red = tracker.update([_det(MarkColor.RED, 100, 100)], W, H)[0]
    for _ in range(3):
        assert tracker.update([], W, H) == []

    back = tracker.update([_det(MarkColor.RED, 120, 110)], W, H)[0]

    assert back.track_id == red.track_id


def test_a_track_hidden_too_long_is_forgotten():
    tracker = HungarianTracker(max_misses=2)
    red = tracker.update([_det(MarkColor.RED, 100, 100)], W, H)[0]
    for _ in range(3):
        tracker.update([], W, H)

    assert len(tracker) == 0
    assert tracker.update([_det(MarkColor.RED, 100, 100)], W, H)[0].track_id != red.track_id


def test_assignment_is_the_cheapest_overall_not_greedy():
    tracker = HungarianTracker()
    tracker.update([_det(MarkColor.RED, 100, 100), _det(MarkColor.BLUE, 160, 100)], W, H)

    # both moved right: the BLUE box is now closest to where RED was, the colours still decide
    marks = tracker.update([_det(MarkColor.RED, 150, 100), _det(MarkColor.BLUE, 210, 100)], W, H)

    assert {m.color for m in marks} == {MarkColor.RED, MarkColor.BLUE}
    assert len({m.track_id for m in marks}) == 2


def test_at_most_one_mark_per_colour_is_reported():
    tracker = HungarianTracker()
    marks = tracker.update(
        [_det(MarkColor.RED, 100, 100, 0.5), _det(MarkColor.RED, 500, 400, 0.9)], W, H
    )

    assert len(marks) == 1
    assert marks[0].confidence == pytest.approx(0.9)


@pytest.mark.parametrize(
    "options", [{"max_jump": 0}, {"max_jump": 1.5}, {"max_misses": -1}, {"vote_decay": 0}]
)
def test_bad_options_are_refused(options):
    with pytest.raises(ValueError):
        HungarianTracker(**options)


def test_a_frame_without_size_is_refused():
    with pytest.raises(ValueError):
        HungarianTracker().update([_det(MarkColor.RED, 1, 1)], 0, 0)
