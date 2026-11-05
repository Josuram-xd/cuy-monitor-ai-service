import json

import pytest

from app.vision.classes import DETECTOR_CLASSES
from training.prepare_dataset import prepare, read_class_names, split_of
from training.synthetic_dataset import HEIGHT, WIDTH, render_frame, write_dataset
from training.train_detector import check_data_names, train


def test_synthetic_frames_are_reproducible_and_their_labels_fit_the_frame():
    first, boxes = render_frame(seed=4)
    again, same = render_frame(seed=4)

    assert (first == again).all() and boxes == same
    assert first.shape == (HEIGHT, WIDTH, 3)
    assert len({b.color for b in boxes}) == len(boxes)
    for box in boxes:
        values = [float(v) for v in box.yolo_line().split()[1:]]
        assert all(0 < v < 1 for v in values)


def test_prepare_maps_names_drops_broken_lines_and_splits_the_same_way(tmp_path):
    export = tmp_path / "export"
    write_dataset(export, frames=30)
    (export / "classes.txt").write_text(
        "rojo\nazul\nverde\namarillo\nnaranja\nmorado\nnegro\nblanco\n"
    )
    (export / "labels" / "synthetic_00000.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 1.5 0.5 0.2 0.2\nx\n"
    )

    report = prepare(export, tmp_path / "out")

    assert report.frames["train"] + report.frames["val"] == 30
    assert len(report.dropped) == 2
    assert (
        (tmp_path / "out" / "data.yaml")
        .read_text()
        .endswith("names: [" + ", ".join(c.value for c in DETECTOR_CLASSES) + "]\n")
    )
    split = split_of("synthetic_00000", 20)
    assert (tmp_path / "out" / "images" / split / "synthetic_00000.jpg").exists()
    assert (tmp_path / "out" / "labels" / split / "synthetic_00000.txt").read_text() == (
        "0 0.500000 0.500000 0.200000 0.200000\n"
    )


def test_the_labels_only_export_carries_no_image(tmp_path):
    export = tmp_path / "export"
    write_dataset(export, frames=5)

    prepare(export, tmp_path / "out", labels_only=True)

    assert not (tmp_path / "out" / "images").exists()
    assert not list((tmp_path / "out").rglob("*.jpg"))
    manifest = (tmp_path / "out" / "manifest.csv").read_text().splitlines()
    assert manifest[0] == "frame,image,sha256,split,boxes" and len(manifest) == 6


def test_roboflow_names_are_read_from_data_yaml(tmp_path):
    (tmp_path / "data.yaml").write_text("train: x\nnames: ['red', 'blue']\n")
    assert read_class_names(tmp_path) == ["red", "blue"]
    (tmp_path / "data.yaml").write_text("train: x\nnames:\n  - red\n  - blue\nnc: 2\n")
    assert read_class_names(tmp_path) == ["red", "blue"]
    (tmp_path / "data.yaml").write_text("names:\n  0: red\n  1: blue\nnc: 2\n")
    assert read_class_names(tmp_path) == ["red", "blue"]


def test_training_exports_the_onnx_model_and_its_card(tmp_path):
    export = tmp_path / "export"
    write_dataset(export, frames=5)
    prepare(export, tmp_path / "dataset")
    exported = tmp_path / "runs" / "best.onnx"

    class FakeYolo:
        def __init__(self, weights):
            self.weights = weights

        def train(self, **options):
            self.options = options
            return type("Results", (), {"results_dict": {"metrics/mAP50(B)": 0.91234}})()

        def export(self, **options):
            exported.parent.mkdir(parents=True, exist_ok=True)
            exported.write_bytes(b"onnx")
            return str(exported)

    target = train(
        tmp_path / "dataset" / "data.yaml", tmp_path / "models", epochs=3, model_factory=FakeYolo
    )

    card = json.loads((tmp_path / "models" / "detector.json").read_text())
    assert target.read_bytes() == b"onnx"
    assert card["classes"] == [c.value for c in DETECTOR_CLASSES]
    assert card["metrics"] == {"metrics/mAP50(B)": 0.9123}


def test_training_refuses_a_dataset_with_another_class_order(tmp_path):
    data = tmp_path / "data.yaml"
    data.write_text("names: [BLUE, RED]\n")

    with pytest.raises(ValueError, match="prepare_dataset"):
        check_data_names(data)
