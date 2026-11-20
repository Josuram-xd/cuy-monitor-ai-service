import numpy as np

from app.contracts import MarkColor
from app.vision.animal_detector import AnimalBox
from app.vision.appearance import describe, similarity
from app.vision.detector import Detection
from app.vision.identity import Reference, identify
from app.vision.tracker import TrackedMark

WHITE_BOX = (50.0, 50.0, 150.0, 150.0)
BLACK_BOX = (300.0, 200.0, 400.0, 300.0)
BROWN_BOX = (450.0, 300.0, 550.0, 400.0)
BGR = {"white": (235, 235, 235), "black": (20, 20, 20), "brown": (40, 80, 150)}


def scene() -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for (x1, y1, x2, y2), colour in (
        (WHITE_BOX, "white"),
        (BLACK_BOX, "black"),
        (BROWN_BOX, "brown"),
    ):
        image[int(y1) : int(y2), int(x1) : int(x2)] = BGR[colour]
    return image


IMAGE = scene()
WHITE_COAT = describe(IMAGE, WHITE_BOX)
BLACK_COAT = describe(IMAGE, BLACK_BOX)
BROWN_COAT = describe(IMAGE, BROWN_BOX)


def _around(box, size=10.0):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (cx - size, cy - size, cx + size, cy + size)


def _mark(color, box, confidence=0.9):
    return Detection(color, confidence, _around(box))


def _track(color, box, confidence=0.8):
    return TrackedMark(track_id=1, color=color, confidence=confidence, box=box)


ANIMALS = [
    AnimalBox(0.5, WHITE_BOX),
    AnimalBox(0.4, BLACK_BOX),
    AnimalBox(0.3, BROWN_BOX),
]


def test_a_mark_inside_an_animal_gives_its_colour_and_the_body_box_and_teaches_the_gallery():
    result = identify(IMAGE, [_mark(MarkColor.RED, WHITE_BOX)], ANIMALS[:1], [], [])

    [detection] = result.detections
    assert detection.color is MarkColor.RED and detection.confidence == 0.9
    assert detection.box == WHITE_BOX
    [(color, descriptor)] = result.enrol
    assert color is MarkColor.RED and similarity_to(descriptor, WHITE_COAT)


def test_a_mark_without_an_animal_passes_through_unchanged_and_teaches_nothing():
    mark = _mark(MarkColor.RED, (600.0, 20.0, 620.0, 40.0))

    result = identify(IMAGE, [mark], ANIMALS[:1], [], [])

    assert result.detections == [mark]
    assert result.enrol == []


def test_a_weak_mark_inside_an_animal_does_not_teach_the_gallery():
    result = identify(IMAGE, [_mark(MarkColor.RED, WHITE_BOX, 0.4)], ANIMALS[:1], [], [])

    assert result.detections[0].confidence == 0.4 and result.enrol == []


def test_an_animal_keeps_the_colour_of_the_track_it_overlaps_with_a_decaying_confidence():
    previous = [_track(MarkColor.BLUE, BLACK_BOX, confidence=0.8)]

    result = identify(IMAGE, [], ANIMALS[1:2], [], previous)

    [detection] = result.detections
    assert detection.color is MarkColor.BLUE
    assert detection.confidence == round(0.8 * 0.9, 3)
    assert detection.box == BLACK_BOX


def test_inheritance_does_not_reuse_a_colour_the_frame_already_has():
    previous = [_track(MarkColor.BLUE, BLACK_BOX)]
    marks = [_mark(MarkColor.BLUE, WHITE_BOX)]

    result = identify(IMAGE, marks, ANIMALS[:2], [], previous)

    assert [d.color for d in result.detections] == [MarkColor.BLUE]
    assert result.detections[0].box == WHITE_BOX
    assert result.unidentified == [center(BLACK_BOX)]


def test_a_coat_identifies_its_guinea_pig_when_it_clearly_wins():
    references = [
        Reference(MarkColor.GREEN, "BLACK", [BLACK_COAT]),
        Reference(MarkColor.YELLOW, "WHITE", [WHITE_COAT]),
    ]

    result = identify(IMAGE, [], ANIMALS[1:2], references, [])

    [detection] = result.detections
    assert detection.color is MarkColor.GREEN
    assert detection.confidence > 0.95
    assert result.unidentified == []


def test_two_coats_that_look_alike_identify_nobody():
    references = [
        Reference(MarkColor.GREEN, None, [BLACK_COAT]),
        Reference(MarkColor.YELLOW, None, [BLACK_COAT]),
    ]

    result = identify(IMAGE, [], ANIMALS[1:2], references, [])

    assert result.detections == []
    assert result.unidentified == [center(BLACK_BOX)]


def test_a_coat_that_contradicts_the_registered_one_is_never_matched():
    references = [Reference(MarkColor.GREEN, "WHITE", [BLACK_COAT])]

    result = identify(IMAGE, [], ANIMALS[1:2], references, [])

    assert result.detections == []


def test_a_colour_already_seen_with_a_mark_is_not_given_to_a_coat():
    marks = [_mark(MarkColor.GREEN, BROWN_BOX)]
    references = [Reference(MarkColor.GREEN, None, [BLACK_COAT])]

    result = identify(IMAGE, marks, ANIMALS[1:], references, [])

    assert [d.color for d in result.detections] == [MarkColor.GREEN]
    assert result.detections[0].box == BROWN_BOX
    assert result.unidentified == [center(BLACK_BOX)]


def test_without_references_only_the_marks_come_back():
    marks = [_mark(MarkColor.RED, WHITE_BOX)]

    result = identify(IMAGE, marks, [], [], [])

    assert [d.color for d in result.detections] == [MarkColor.RED]
    assert result.unidentified == []


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def similarity_to(a, b):
    return similarity(a, b) > 0.95
