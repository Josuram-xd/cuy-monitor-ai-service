from types import SimpleNamespace

import numpy as np
import pytest

from app.vision.animal_detector import OnnxAnimalDetector

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


class FakeSession:
    """Answers with a fixed raw output (cx, cy, w, h, one score per prompt) in model pixels."""

    def __init__(self, output: np.ndarray):
        self.output = output

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def run(self, output_names, input_feed):
        return [self.output]


def raw(columns: list[list[float]]) -> np.ndarray:
    """One row per box, [cx, cy, w, h, *scores], as the (1, 4 + prompts, N) model writes it."""
    if not columns:
        return np.zeros((1, 5, 0), dtype=np.float32)
    return np.array(columns, dtype=np.float32).T[None]


def test_an_animal_is_a_box_in_frame_pixels_above_the_threshold():
    # letterbox of a 640x480 frame into 640: scale 1, 80 px of grey padding on top
    session = FakeSession(raw([[320, 400, 100, 100, 0.6], [100, 100, 60, 60, 0.01]]))

    [animal] = OnnxAnimalDetector(session).detect_image(FRAME)

    assert animal.confidence == pytest.approx(0.6)
    assert animal.box == pytest.approx((270, 270, 370, 370), abs=0.5)


def test_several_prompts_for_one_animal_leave_one_box():
    # the same animal scored by three prompts: per-class NMS keeps three, the animal NMS one
    session = FakeSession(raw([[320, 400, 100, 100, 0.5, 0.4, 0.3]]))

    animals = OnnxAnimalDetector(session).detect_image(FRAME)

    assert len(animals) == 1 and animals[0].confidence == pytest.approx(0.5)


def test_crumbs_and_the_whole_cage_are_not_animals():
    session = FakeSession(
        raw(
            [
                [100, 300, 5, 5, 0.9],  # a crumb of bedding
                [320, 240, 600, 460, 0.9],  # the whole cage
                [400, 300, 80, 80, 0.2],  # a normal guinea pig
            ]
        )
    )

    animals = OnnxAnimalDetector(session).detect_image(FRAME)

    assert [round(a.confidence, 1) for a in animals] == [0.2]


def test_a_frame_without_animals_gives_an_empty_list():
    assert OnnxAnimalDetector(FakeSession(raw([]))).detect_image(FRAME) == []


def test_the_confidence_must_be_a_fraction():
    with pytest.raises(ValueError):
        OnnxAnimalDetector(FakeSession(raw([])), confidence=1.0)
