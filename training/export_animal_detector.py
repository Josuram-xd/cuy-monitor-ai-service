"""Exports the animal detector: YOLO-World with the fixed prompt "animal", to ONNX (Task A11).

Runs once, offline (Colab or a venv with ultralytics), never in the service image. The prompts are
baked into the model, so the ONNX needs no CLIP text encoder at runtime.

  python -m training.export_animal_detector --weights yolov8s-worldv2.pt --prompts animal \\
         --imgsz 640 --opset 17 --out models [--check-images DIR]

Output, in --out: animal_detector.onnx (what ANIMAL_DETECTOR_PATH loads) and animal_detector.json
(prompts, sizes, opset, sha256 and ultralytics version: publish them with the model). The
intermediate .pt is removed. --check-images compares the ONNX boxes with ultralytics on each image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.vision.animal_detector import OnnxAnimalDetector

CHECK_CONFIDENCE = 0.03  # low, as the service uses it: the comparison covers the faint boxes too
BOX_TOLERANCE_PX = 2.0
SCORE_TOLERANCE = 0.01


@dataclass(frozen=True)
class Export:
    target: Path  # animal_detector.onnx
    reference: Any  # the ultralytics model that was exported, for --check-images


def _default_model_factory(weights: str) -> Any:
    from ultralytics import YOLO

    return YOLO(weights)


def _ultralytics_version() -> str | None:
    try:
        return metadata.version("ultralytics")
    except metadata.PackageNotFoundError:
        return None


def export(
    weights: str,
    prompts: list[str],
    out: Path,
    imgsz: int = 640,
    opset: int = 17,
    model_factory: Callable[[str], Any] = _default_model_factory,
) -> Export:
    """YOLO-World -> set_classes -> save -> reload -> ONNX (the path the ultralytics docs give)."""
    if not prompts:
        raise ValueError("At least one prompt is needed, e.g. --prompts animal")
    model = model_factory(weights)
    model.set_classes(prompts)
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "animal_world.pt"
    model.save(str(saved))
    # the prompts travel inside the saved file: the export runs on the reloaded model
    reloaded = model_factory(str(saved))
    exported = Path(
        reloaded.export(
            format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False, nms=False
        )
    )
    target = out / "animal_detector.onnx"
    shutil.copy2(exported, target)
    saved.unlink(missing_ok=True)
    card = {
        "prompts": prompts,
        "imgsz": imgsz,
        "opset": opset,
        "weights": weights,
        "ultralytics": _ultralytics_version(),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "exportedAt": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "animal_detector.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    return Export(target, reloaded)


def check_onnx(path: Path) -> list[int]:
    """The ONNX checker's verdict (raises if the file is broken); returns the opset versions."""
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return [o.version for o in model.opset_import if o.domain in ("", "ai.onnx")]


def same_boxes(
    ours: list[tuple[np.ndarray, float]], theirs: list[tuple[np.ndarray, float]]
) -> bool:
    """True when both found the same animals: same count, boxes within 2 px, scores within 0.01."""
    if len(ours) != len(theirs):
        return False
    by_score = sorted(range(len(ours)), key=lambda i: -ours[i][1])
    other = sorted(range(len(theirs)), key=lambda i: -theirs[i][1])
    for i, j in zip(by_score, other, strict=True):
        box_a, score_a = ours[i]
        box_b, score_b = theirs[j]
        if np.max(np.abs(np.asarray(box_a) - np.asarray(box_b))) > BOX_TOLERANCE_PX:
            return False
        if abs(score_a - score_b) > SCORE_TOLERANCE:
            return False
    return True


def compare(reference: Any, onnx_path: Path, images: list[Path]) -> int:
    """Prints, per image, whether the ONNX and ultralytics agree. Returns how many disagree."""
    detector = OnnxAnimalDetector.load(onnx_path, confidence=CHECK_CONFIDENCE)
    disagreements = 0
    for image in images:
        ours = [
            (np.asarray(a.box), a.confidence) for a in detector.detect_image(cv2.imread(str(image)))
        ]
        result = reference.predict(str(image), conf=CHECK_CONFIDENCE, verbose=False)[0]
        theirs = [
            (np.asarray(box), float(score))
            for box, score in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), strict=True
            )
        ]
        agree = same_boxes(ours, theirs)
        disagreements += not agree
        print(f"{image.name}: {len(ours)} boxes here, {len(theirs)} in ultralytics, same={agree}")
    return disagreements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default="yolov8s-worldv2.pt")
    parser.add_argument("--prompts", default="animal", help="comma separated, the fixed prompt(s)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--check-images", type=Path, help="folder of real frames to compare with")
    args = parser.parse_args(argv)

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    result = export(args.weights, prompts, args.out, args.imgsz, args.opset)
    print(f"Animal detector exported to {result.target}")
    if args.check_images is None:
        return 0
    print(f"opset {check_onnx(result.target)}")
    images = sorted(p for p in args.check_images.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    disagreements = compare(result.reference, result.target, images[:3])
    return 1 if disagreements else 0


if __name__ == "__main__":
    sys.exit(main())
