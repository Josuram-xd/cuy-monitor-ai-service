import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from app.contracts import MarkColor
from app.vision.classes import DETECTOR_CLASSES, color_of

logger = logging.getLogger(__name__)

DEFAULT_INPUT_SIZE = 640
DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.5
_PAD_VALUE = 114


class UnreadableFrame(ValueError):
    """The bytes are not an image OpenCV can decode."""


class InferenceSession(Protocol):
    """The part of onnxruntime.InferenceSession the detector uses (a fake one in the tests)."""

    def get_inputs(self) -> list[Any]: ...

    def run(self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]) -> list: ...


@dataclass(frozen=True)
class Detection:
    """One marked guinea pig in a frame. box = (x1, y1, x2, y2) in pixels of the original frame."""

    color: MarkColor
    confidence: float
    box: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass(frozen=True)
class Letterbox:
    """How a frame was scaled and padded to the square model input, to undo it on the boxes."""

    scale: float
    pad_x: float
    pad_y: float
    width: int
    height: int

    def to_frame(self, box: np.ndarray) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        x1 = (x1 - self.pad_x) / self.scale
        x2 = (x2 - self.pad_x) / self.scale
        y1 = (y1 - self.pad_y) / self.scale
        y2 = (y2 - self.pad_y) / self.scale
        return (
            float(np.clip(x1, 0, self.width)),
            float(np.clip(y1, 0, self.height)),
            float(np.clip(x2, 0, self.width)),
            float(np.clip(y2, 0, self.height)),
        )


def decode_jpeg(frame: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise UnreadableFrame("not a readable image")
    return image


def letterbox(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """The frame as the model wants it: square, aspect kept, grey padding, RGB, NCHW float 0-1."""
    height, width = image_bgr.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2
    canvas = np.full((size, size, 3), _PAD_VALUE, dtype=np.uint8)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), Letterbox(scale, left, top, width, height)


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    order = list(np.argsort(-scores))
    kept: list[int] = []
    while order:
        best = order.pop(0)
        kept.append(best)
        if order:
            overlaps = _iou(boxes[best], boxes[order])
            order = [i for i, overlap in zip(order, overlaps, strict=True) if overlap < iou]
    return kept


def parse_output(
    output: np.ndarray, confidence: float, iou: float
) -> list[tuple[np.ndarray, float, int]]:
    """Model output -> [(xyxy box in model pixels, score, class)].

    Two layouts are understood:
    - end-to-end (YOLO26 default export, NMS-free): (1, N, 6) rows of x1, y1, x2, y2, score, class.
    - raw (YOLOv8-style): (1, 4 + classes, N) columns of cx, cy, w, h, then one score per class;
      the best class is taken and overlapping boxes of the same class are removed (NMS).
    """
    rows = np.asarray(output)[0]
    if rows.ndim != 2:
        raise ValueError(f"Unexpected detector output shape {np.shape(output)}")
    if rows.shape[1] == 6:
        keep = rows[:, 4] >= confidence
        return [(row[:4], float(row[4]), int(row[5])) for row in rows[keep]]

    predictions = rows.T
    class_count = predictions.shape[1] - 4
    if class_count != len(DETECTOR_CLASSES):
        raise ValueError(
            f"The model has {class_count} classes, the service expects {len(DETECTOR_CLASSES)}"
        )
    scores = predictions[:, 4:]
    classes = scores.argmax(axis=1)
    best = scores.max(axis=1)
    keep = best >= confidence
    centers, classes, best = predictions[keep, :4], classes[keep], best[keep]
    boxes = np.column_stack(
        [
            centers[:, 0] - centers[:, 2] / 2,
            centers[:, 1] - centers[:, 3] / 2,
            centers[:, 0] + centers[:, 2] / 2,
            centers[:, 1] + centers[:, 3] / 2,
        ]
    )
    found = []
    for class_index in np.unique(classes):
        mine = np.flatnonzero(classes == class_index)
        for i in _nms(boxes[mine], best[mine], iou):
            found.append((boxes[mine[i]], float(best[mine[i]]), int(class_index)))
    return found


def _square_input_size(model_input: Any) -> int | None:
    """640 for an input of shape (1, 3, 640, 640); None when the model has dynamic sizes."""
    shape = list(getattr(model_input, "shape", None) or [])
    if len(shape) == 4 and isinstance(shape[2], int) and shape[2] == shape[3]:
        return shape[2]
    return None


class OnnxDetector:
    """YOLO26n exported to ONNX: one frame in, one Detection per mark colour out.

    A cage has at most one guinea pig of each colour, so when the model sees a colour twice only the
    most confident box is kept. Boxes under the confidence threshold are dropped: a colour is never
    guessed. The session is injected (Dependency Inversion), so tests run without a model file.
    """

    def __init__(
        self,
        session: InferenceSession,
        input_size: int | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
    ):
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        self._session = session
        model_input = session.get_inputs()[0]
        self._input_name = model_input.name
        # the size the model was exported with (imgsz): sending another one fails on every frame
        self._input_size = input_size or _square_input_size(model_input) or DEFAULT_INPUT_SIZE
        self._confidence = confidence
        self._iou = iou

    @classmethod
    def load(cls, path: Path, **options) -> OnnxDetector:
        import onnxruntime

        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return cls(session, **options)

    def detect_image(self, image_bgr: np.ndarray) -> list[Detection]:
        tensor, box = letterbox(image_bgr, self._input_size)
        output = self._session.run(None, {self._input_name: tensor})[0]
        best: dict[MarkColor, Detection] = {}
        for xyxy, score, class_index in parse_output(output, self._confidence, self._iou):
            try:
                color = color_of(class_index)
            except ValueError:
                logger.warning("Ignoring a detection of unknown class %s", class_index)
                continue
            detection = Detection(color, round(score, 3), box.to_frame(xyxy))
            if color not in best or detection.confidence > best[color].confidence:
                best[color] = detection
        return sorted(best.values(), key=lambda d: DETECTOR_CLASSES.index(d.color))

    def detect(self, frame: bytes) -> list[Detection]:
        return self.detect_image(decode_jpeg(frame))

    async def detect_async(self, frame: bytes) -> list[Detection]:
        # inference is CPU bound: never on the event loop (AGENTS rule 6)
        return await asyncio.to_thread(self.detect, frame)
