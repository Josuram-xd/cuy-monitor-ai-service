"""YOLO-World with the fixed prompt "animal": finds guinea pigs that have no mark (Task A2).

The mark detector only sees the coloured marks. This one finds the animal itself, so a guinea pig
whose mark is hidden can still be found and identified by its coat (app/vision/identity.py).
Every class of the model counts as "animal" (the export may use several prompts), so after the
per-class NMS of parse_output one more NMS, without classes, leaves one box per animal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.vision.detector import (
    DEFAULT_INPUT_SIZE,
    InferenceSession,
    _nms,
    _square_input_size,
    letterbox,
    parse_output,
)

# YOLO-World "animal" scores 0.03-0.35 on the farm: a higher threshold would miss most animals
DEFAULT_ANIMAL_CONFIDENCE = 0.05
# fractions of the frame: no crumbs of bedding, and not the whole cage either
MIN_AREA, MAX_AREA = 0.002, 0.4


@dataclass(frozen=True)
class AnimalBox:
    """One animal in a frame. box = (x1, y1, x2, y2) in pixels of the original frame."""

    confidence: float
    box: tuple[float, float, float, float]


class OnnxAnimalDetector:
    def __init__(
        self,
        session: InferenceSession,
        input_size: int | None = None,
        confidence: float = DEFAULT_ANIMAL_CONFIDENCE,
        iou: float = 0.5,
    ):
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        self._session = session
        model_input = session.get_inputs()[0]
        self._input_name = model_input.name
        self._input_size = input_size or _square_input_size(model_input) or DEFAULT_INPUT_SIZE
        self._confidence = confidence
        self._iou = iou

    @classmethod
    def load(cls, path: Path, **options) -> OnnxAnimalDetector:
        import onnxruntime

        # two sessions (marks and animals) share two vCPUs: each one stays on a single thread
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 1
        session = onnxruntime.InferenceSession(
            str(path), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        return cls(session, **options)

    def detect_image(self, image_bgr: np.ndarray) -> list[AnimalBox]:
        tensor, box = letterbox(image_bgr, self._input_size)
        output = self._session.run(None, {self._input_name: tensor})[0]
        found = parse_output(output, self._confidence, self._iou, class_count=None)
        if not found:
            return []
        boxes = np.array([b for b, _, _ in found], dtype=np.float32)
        scores = np.array([s for _, s, _ in found], dtype=np.float32)
        height, width = image_bgr.shape[:2]
        animals = []
        for index in _nms(boxes, scores, self._iou):
            frame_box = box.to_frame(boxes[index])
            area = (frame_box[2] - frame_box[0]) * (frame_box[3] - frame_box[1])
            if MIN_AREA <= area / (width * height) <= MAX_AREA:
                animals.append(AnimalBox(float(scores[index]), frame_box))
        return animals
