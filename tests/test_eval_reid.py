"""Task A10: the identification evaluation scores what the frames say against the clicks."""

import json

from app.contracts import MarkColor
from app.vision.animal_detector import AnimalBox
from app.vision.detector import Detection
from app.vision.tracker import TrackedMark
from dev.eval_reid import evaluate, format_report, score_frame, window_confidences
from tests.clip_helpers import write_clip

BIG = (0.0, 0.0, 40.0, 40.0)  # an animal box covering the clicks of the test clip


def _det(color: MarkColor, x: float, y: float) -> Detection:
    return Detection(color, 0.9, (x - 5, y - 5, x + 5, y + 5))


def test_a_detection_that_holds_its_own_click_is_correct():
    tallies = score_frame(
        {"RED": (10.0, 10.0), "BLUE": (100.0, 100.0)},
        [_det(MarkColor.RED, 10, 10)],
        [AnimalBox(0.5, (0, 0, 40, 40))],
    )

    assert tallies["RED"].correct == 1 and tallies["RED"].animal_found == 1
    assert tallies["BLUE"].unidentified == 1 and tallies["BLUE"].animal_found == 0


def test_a_colour_on_another_guinea_pig_is_a_wrong_colour():
    tallies = score_frame(
        {"RED": (10.0, 10.0), "BLUE": (100.0, 100.0)},
        [_det(MarkColor.BLUE, 10, 10)],
        [],
    )

    assert tallies["BLUE"].wrong == 1 and tallies["BLUE"].correct == 0
    assert tallies["RED"].unidentified == 1


def test_a_detection_on_nothing_is_not_counted_either_way():
    tallies = score_frame({"RED": (10.0, 10.0)}, [_det(MarkColor.RED, 200, 200)], [])

    assert tallies["RED"].correct == 0 and tallies["RED"].wrong == 0


def test_the_window_confidence_is_the_share_of_frames_times_the_confidence():
    def mark(color, confidence):
        return TrackedMark(1, color, confidence, (0, 0, 1, 1))

    frames = [[mark(MarkColor.RED, 0.9)], [], [mark(MarkColor.RED, 0.9)], []]

    assert window_confidences(frames) == {"RED": [0.45]}


def test_the_clip_is_evaluated_frame_by_frame_against_its_clicks(tmp_path):
    write_clip(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["marks"] = {"RED": [10, 10]}
    manifest["frames"][1]["marks"] = {"BLUE": [30, 30]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class Animals:
        def detect_image(self, image):
            return [AnimalBox(0.5, BIG)]

    class Marks:
        def __init__(self):
            self.calls = 0

        def detect_image(self, image):
            self.calls += 1
            return (
                [_det(MarkColor.RED, 10, 10)] if self.calls == 1 else [_det(MarkColor.BLUE, 30, 30)]
            )

    tallies = evaluate(tmp_path, Animals(), Marks())

    assert tallies["RED"].visible == 1 and tallies["RED"].correct == 1
    assert tallies["BLUE"].visible == 1 and tallies["BLUE"].correct == 1
    assert tallies["RED"].window_confidences == [0.45]  # seen in 1 of the 2 frames
    assert "RED" in format_report(tallies)
