import json
from pathlib import Path

import numpy as np
import pytest

from training.export_animal_detector import export, same_boxes


class FakeYolo:
    """Records what the export asked for; writes the files ultralytics would."""

    calls: list = []

    def __init__(self, weights):
        self.weights = weights

    def set_classes(self, prompts):
        FakeYolo.calls.append(("set_classes", prompts))

    def save(self, path):
        FakeYolo.calls.append(("save", path))
        Path(path).write_bytes(b"pt weights")

    def export(self, **options):
        FakeYolo.calls.append(("export", options))
        target = Path(self.weights).with_suffix(".onnx")
        target.write_bytes(b"onnx model")
        return str(target)


def test_the_prompts_are_set_saved_reloaded_and_exported_with_the_fixed_options(tmp_path):
    FakeYolo.calls = []
    out = tmp_path / "models"

    result = export("yolov8s-worldv2.pt", ["animal"], out, model_factory=lambda w: FakeYolo(w))

    names = [name for name, _ in FakeYolo.calls]
    assert names == ["set_classes", "save", "export"]
    assert FakeYolo.calls[0] == ("set_classes", ["animal"])
    exported_options = FakeYolo.calls[2][1]
    assert exported_options == {
        "format": "onnx",
        "imgsz": 640,
        "opset": 17,
        "simplify": True,
        "dynamic": False,
        "nms": False,
    }
    assert result.target == out / "animal_detector.onnx"
    assert result.target.read_bytes() == b"onnx model"
    assert not (out / "animal_world.pt").exists()


def test_the_card_has_the_prompts_and_the_sha256_of_the_model(tmp_path):
    import hashlib

    out = tmp_path / "models"
    export("yolov8s-worldv2.pt", ["animal"], out, model_factory=FakeYolo)

    card = json.loads((out / "animal_detector.json").read_text(encoding="utf-8"))
    assert card["prompts"] == ["animal"]
    assert card["opset"] == 17 and card["imgsz"] == 640
    assert card["sha256"] == hashlib.sha256(b"onnx model").hexdigest()


def test_no_prompt_is_refused(tmp_path):
    with pytest.raises(ValueError, match="prompt"):
        export("w.pt", [], tmp_path, model_factory=FakeYolo)


def test_boxes_agree_within_two_pixels_and_scores_within_a_hundredth():
    box = np.array([10.0, 10.0, 50.0, 60.0])
    ours = [(box, 0.5), (np.array([100.0, 100.0, 120.0, 120.0]), 0.2)]
    close = [(box + 1.0, 0.505), (np.array([100.0, 100.0, 120.0, 121.5]), 0.2)]
    far = [(box + 5.0, 0.5), (np.array([100.0, 100.0, 120.0, 120.0]), 0.2)]

    assert same_boxes(ours, close)
    assert not same_boxes(ours, far)
    assert not same_boxes(ours, close[:1])
