from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
import pytest

from app.behavior.classifier import BehaviorClassifier, RuleBehaviorClassifier
from app.behavior.engine import TrainedBehaviorEngine
from app.contracts import MarkColor
from app.vision.annotate import annotate
from app.vision.detector import Detection
from app.vision.zones import NO_ZONES


def _pixel(jpeg: bytes, x: int, y: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image[y, x].astype(int)


def test_identified_guinea_pigs_are_boxed_in_their_mark_colour_and_the_rest_in_grey():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    red = Detection(MarkColor.RED, 0.9, (10.0, 60.0, 100.0, 180.0))
    jpeg = annotate(frame, [red], [red.box, (150.0, 60.0, 250.0, 180.0)])

    # JPEG blurs the edges: the colour is checked on the middle of the 2 px line
    assert _pixel(jpeg, 10, 120)[2] > 150  # red channel of the RED box
    assert abs(int(_pixel(jpeg, 150, 120)[0]) - 160) < 50  # grey of the unidentified animal
    assert _pixel(jpeg, 200, 120).max() < 60  # nothing drawn between the boxes


class _Detector:
    def detect_image(self, image_bgr):
        return [Detection(MarkColor.RED, 0.9, (10.0, 60.0, 100.0, 180.0))]


@pytest.mark.anyio
async def test_the_engine_keeps_the_picture_of_the_latest_analysed_frame():
    engine = TrainedBehaviorEngine(
        _Detector(), BehaviorClassifier(RuleBehaviorClassifier()), lambda _: NO_ZONES, 60
    )
    jpeg = annotate(np.zeros((240, 320, 3), dtype=np.uint8), [], [])  # any JPEG of the right size
    now = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
    assert engine.latest_annotated("cage-1") is None

    await engine.submit_frame("cage-1", jpeg, now)
    await engine.submit_frame("cage-1", jpeg, now - timedelta(seconds=5))  # late: not kept

    captured_at, picture = engine.latest_annotated("cage-1")
    assert captured_at == now
    assert _pixel(picture, 10, 120)[2] > 150  # the red box of the detector
