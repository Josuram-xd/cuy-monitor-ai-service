"""Trains the YOLO26n detector and exports it to ONNX for the service (Task 7.1).

Runs in Google Colab with a GPU (Task 7.2), never in the service image: Ultralytics is only
installed there (training/requirements.txt).

  python -m training.train_detector --data /content/cuy-dataset/data.yaml --epochs 100

Output, in --out (models/ by default):
  detector.onnx   what the service loads (DETECTOR_PATH)
  detector.json   classes in order, input size, metrics and sha256, to upload with it to the release
"""

import argparse
import hashlib
import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.vision.classes import DETECTOR_CLASSES


def check_data_names(data_yaml: Path) -> None:
    """The dataset must name the classes in the detector's order, or colours would be swapped."""
    expected = ", ".join(c.value for c in DETECTOR_CLASSES)
    names = next(
        (
            line.split(":", 1)[1].strip()
            for line in data_yaml.read_text().splitlines()
            if line.strip().startswith("names:")
        ),
        "",
    )
    if names.strip("[]").replace("'", "").replace('"', "") != expected:
        raise ValueError(
            f"{data_yaml} names its classes {names}, the detector needs [{expected}]: "
            "make it with training/prepare_dataset.py"
        )


def _default_model_factory(weights: str) -> Any:
    from ultralytics import YOLO

    return YOLO(weights)


def train(
    data: Path,
    out: Path,
    weights: str = "yolo26n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    model_factory: Callable[[str], Any] = _default_model_factory,
) -> Path:
    check_data_names(data)
    model = model_factory(weights)
    results = model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, patience=20)
    exported = Path(model.export(format="onnx", imgsz=imgsz, simplify=True))

    out.mkdir(parents=True, exist_ok=True)
    target = out / "detector.onnx"
    shutil.copy2(exported, target)
    metrics = getattr(results, "results_dict", {}) or {}
    (out / "detector.json").write_text(
        json.dumps(
            {
                "classes": [c.value for c in DETECTOR_CLASSES],
                "imgsz": imgsz,
                "weights": weights,
                "epochs": epochs,
                "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "trainedAt": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="data.yaml of prepare_dataset")
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args(argv)
    target = train(args.data, args.out, args.weights, args.epochs, args.imgsz, args.batch)
    print(f"Detector exported to {target}")


if __name__ == "__main__":
    main()
