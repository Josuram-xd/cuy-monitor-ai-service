"""The annotated live picture: each guinea pig boxed in its mark colour, the rest grey with "?".

Drawn only for the frames the engine analyses (the latest of each cage), so it stays cheap. OpenCV's
built-in font is ASCII only, hence the plain caption.
"""

import cv2
import numpy as np

from app.testdata.scene import MARK_BGR
from app.vision.detector import Detection

GREY_BGR = (160, 160, 160)


def annotate(
    image_bgr: np.ndarray,
    detections: list[Detection],
    animal_boxes: list[tuple[float, float, float, float]],
) -> bytes:
    """A JPEG of the frame: the identified guinea pigs, then the animals nobody could name."""
    canvas = image_bgr.copy()
    identified = {d.box for d in detections}
    unknown = [box for box in animal_boxes if box not in identified]
    for detection in detections:
        _box(canvas, detection.box, MARK_BGR[detection.color], detection.color.value)
    for box in unknown:
        _box(canvas, box, GREY_BGR, "?")
    caption = f"IA: {len(detections) + len(unknown)} cuyes - {len(detections)} identificados"
    (width, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(canvas, (0, 0), (width + 12, 26), (0, 0, 0), -1)
    cv2.putText(canvas, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def _box(canvas: np.ndarray, box, color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(canvas, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
