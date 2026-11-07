import asyncio
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.contracts import MarkColor
from app.vision.classes import DETECTOR_CLASSES, class_id, color_named, color_of
from app.vision.detector import OnnxDetector, UnreadableFrame, letterbox, parse_output
from training.synthetic_dataset import render_frame


class FakeSession:
    """Answers with a fixed output and keeps the tensor it was given."""

    def __init__(self, output: np.ndarray):
        self.output = output
        self.inputs: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def run(self, output_names, input_feed):
        self.inputs.append(input_feed["images"])
        return [self.output]


def to_model(box, frame_bgr, size=640):
    _, lb = letterbox(frame_bgr, size)
    x1, y1, x2, y2 = box
    return [
        x1 * lb.scale + lb.pad_x,
        y1 * lb.scale + lb.pad_y,
        x2 * lb.scale + lb.pad_x,
        y2 * lb.scale + lb.pad_y,
    ]


def jpeg(image):
    return cv2.imencode(".jpg", image)[1].tobytes()


def test_classes_follow_mark_color_and_accept_spanish_names():
    assert DETECTOR_CLASSES == tuple(MarkColor)
    assert class_id(MarkColor.BLUE) == 1 and color_of(1) is MarkColor.BLUE
    assert color_named("Rojo") is MarkColor.RED
    assert color_named("mark_white") is MarkColor.WHITE
    with pytest.raises(ValueError):
        color_named("pink")
    with pytest.raises(ValueError):
        color_of(8)


def test_letterbox_keeps_the_aspect_and_gives_the_model_nchw_rgb():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # red in BGR
    tensor, box = letterbox(frame, 640)

    assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
    assert box.scale == 1 and box.pad_y == 80 and box.pad_x == 0
    assert tensor[0, 0, 320, 320] == pytest.approx(1.0)  # channel 0 is red: RGB
    assert tensor[0, 0, 10, 320] == pytest.approx(114 / 255)  # grey padding


def test_end_to_end_output_comes_back_in_frame_pixels_one_per_colour():
    frame, boxes = render_frame(seed=0)
    rows = []
    for b in boxes:
        rows.append([*to_model((b.x1, b.y1, b.x2, b.y2), frame), 0.9, class_id(b.color)])
    first = boxes[0]
    rows.append(
        [*to_model((first.x1, first.y1, first.x2, first.y2), frame), 0.5, class_id(first.color)]
    )
    rows.append([0, 0, 10, 10, 0.2, 0])  # under the threshold: dropped
    session = FakeSession(np.array([rows], dtype=np.float32))

    found = OnnxDetector(session).detect(jpeg(frame))

    assert {d.color for d in found} == {b.color for b in boxes}
    for detection in found:
        expected = next(b for b in boxes if b.color is detection.color)
        assert detection.confidence == pytest.approx(0.9)
        assert detection.box == pytest.approx(
            (expected.x1, expected.y1, expected.x2, expected.y2), abs=1
        )
    assert session.inputs[0].shape == (1, 3, 640, 640)


def test_raw_output_is_decoded_with_one_box_per_overlapping_group():
    columns = np.zeros((4 + len(DETECTOR_CLASSES), 3), dtype=np.float32)
    columns[:4, 0] = [100, 100, 40, 40]
    columns[4 + class_id(MarkColor.GREEN), 0] = 0.8
    columns[:4, 1] = [102, 101, 40, 40]
    columns[4 + class_id(MarkColor.GREEN), 1] = 0.7
    columns[:4, 2] = [300, 300, 50, 50]
    columns[4 + class_id(MarkColor.BLACK), 2] = 0.6

    found = parse_output(columns[None], confidence=0.35, iou=0.5)

    assert sorted((cls, round(score, 2)) for _, score, cls in found) == [
        (class_id(MarkColor.GREEN), 0.8),
        (class_id(MarkColor.BLACK), 0.6),
    ]
    green = next(box for box, _, cls in found if cls == class_id(MarkColor.GREEN))
    assert list(green) == pytest.approx([80, 80, 120, 120])


def test_a_model_with_other_classes_is_refused():
    with pytest.raises(ValueError, match="classes"):
        parse_output(np.zeros((1, 4 + 3, 5), dtype=np.float32), 0.35, 0.5)


def test_an_unreadable_frame_is_an_error_and_inference_runs_off_the_event_loop():
    detector = OnnxDetector(FakeSession(np.zeros((1, 0, 6), dtype=np.float32)))

    with pytest.raises(UnreadableFrame):
        detector.detect(b"not a jpeg")
    frame, _ = render_frame(seed=1)
    assert asyncio.run(detector.detect_async(jpeg(frame))) == []


class SizedSession(FakeSession):
    """A model exported with another imgsz: its input says (1, 3, 320, 320)."""

    def __init__(self, output, shape):
        super().__init__(output)
        self.shape = shape

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=self.shape)]


def test_the_input_size_comes_from_the_model_it_was_exported_with():
    session = SizedSession(np.zeros((1, 0, 6), np.float32), [1, 3, 320, 320])
    OnnxDetector(session).detect_image(np.zeros((480, 640, 3), np.uint8))

    assert session.inputs[0].shape == (1, 3, 320, 320)


def test_a_model_with_dynamic_sizes_gets_640():
    session = SizedSession(np.zeros((1, 0, 6), np.float32), [1, 3, "height", "width"])
    OnnxDetector(session).detect_image(np.zeros((480, 640, 3), np.uint8))

    assert session.inputs[0].shape == (1, 3, 640, 640)
