"""Turns a YOLO export of Label Studio or Roboflow into the dataset the detector is trained with.

- The class names of the export ("rojo", "red", "mark_red"...) are mapped to the detector's classes,
  in the order of app.vision.classes, whatever order the labeler used.
- Every label line is checked; broken ones are dropped and reported, never trained on.
- Train / validation split by a hash of the frame name: a frame always lands in the same split.
- --labels-only writes the labels, data.yaml and a manifest without any image (Task 6.4: the export
  that is committed; frames stay out of the repo).

  python -m training.prepare_dataset --export ~/label-studio-export --out /tmp/cuy-dataset
"""

import argparse
import csv
import hashlib
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.vision.classes import DETECTOR_CLASSES, class_id, color_named

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
# a name in a block list of data.yaml: "  - red" or "  0: red"
_LIST_ITEM = re.compile(r"^\s+(?:-\s*|\d+\s*:\s*)(\S.*)$")


@dataclass
class Report:
    frames: Counter = field(default_factory=Counter)
    boxes: Counter = field(default_factory=Counter)
    dropped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        per_class = ", ".join(f"{c.value}={self.boxes[c.value]}" for c in DETECTOR_CLASSES)
        return (
            f"train={self.frames['train']} val={self.frames['val']} frames; boxes: {per_class}; "
            f"dropped lines: {len(self.dropped)}"
        )


def read_class_names(export: Path) -> list[str]:
    """classes.txt (Label Studio) or the names of data.yaml (Roboflow), in the export's order."""
    classes = next(export.rglob("classes.txt"), None)
    if classes is not None:
        return [line.strip() for line in classes.read_text().splitlines() if line.strip()]
    data = next(export.rglob("data.yaml"), None)
    if data is None:
        raise ValueError("The export has no classes.txt nor data.yaml")
    lines = data.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("names:"):
            inline = line.split(":", 1)[1].strip()
            if inline.startswith("["):
                return [n.strip().strip("'\"") for n in inline.strip("[]").split(",") if n.strip()]
            names = []
            for item in lines[index + 1 :]:
                entry = _LIST_ITEM.match(item)
                if entry is None:
                    break
                names.append(entry.group(1).strip().strip("'\""))
            return names
    raise ValueError(f"{data} has no names")


def remap_classes(names: list[str]) -> dict[int, int]:
    """Export class index -> detector class index. A name that is not a mark colour is an error."""
    return {index: class_id(color_named(name)) for index, name in enumerate(names)}


def split_of(stem: str, val_percent: int) -> str:
    bucket = int(hashlib.sha256(stem.encode()).hexdigest(), 16) % 100
    return "val" if bucket < val_percent else "train"


def convert_label(text: str, mapping: dict[int, int], where: str, report: Report) -> list[str]:
    kept = []
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        problem = None
        if len(parts) != 5:
            problem = "not 5 values"
        else:
            try:
                source = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
            except ValueError:
                problem = "not numbers"
            else:
                if source not in mapping:
                    problem = f"unknown class {source}"
                elif not all(0 <= v <= 1 for v in (cx, cy, w, h)) or w == 0 or h == 0:
                    problem = "box outside the frame or empty"
        if problem:
            report.dropped.append(f"{where}:{number}: {problem}")
            continue
        target = mapping[source]
        report.boxes[DETECTOR_CLASSES[target].value] += 1
        kept.append(f"{target} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return kept


def _images_by_stem(export: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in export.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
    }


def prepare(export: Path, out: Path, val_percent: int = 20, labels_only: bool = False) -> Report:
    mapping = remap_classes(read_class_names(export))
    images = _images_by_stem(export)
    report = Report()
    manifest = []
    labels = sorted(p for p in export.rglob("*.txt") if p.name != "classes.txt")
    for label in labels:
        image = images.get(label.stem)
        if image is None and not labels_only:
            report.dropped.append(f"{label.name}: no image")
            continue
        split = split_of(label.stem, val_percent)
        lines = convert_label(label.read_text(), mapping, label.name, report)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split / f"{label.stem}.txt").write_text("".join(f"{x}\n" for x in lines))
        digest = hashlib.sha256(image.read_bytes()).hexdigest() if image else ""
        if not labels_only:
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, out / "images" / split / image.name)
        manifest.append((label.stem, image.name if image else "", digest, split, len(lines)))
        report.frames[split] += 1

    (out / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        f"nc: {len(DETECTOR_CLASSES)}\n"
        "names: [" + ", ".join(c.value for c in DETECTOR_CLASSES) + "]\n"
    )
    with (out / "manifest.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "image", "sha256", "split", "boxes"])
        writer.writerows(manifest)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, required=True, help="folder of the YOLO export")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-percent", type=int, default=20)
    parser.add_argument("--labels-only", action="store_true", help="no images (to commit)")
    args = parser.parse_args(argv)
    report = prepare(args.export, args.out, args.val_percent, args.labels_only)
    print(report.summary())
    for line in report.dropped[:20]:
        print("  dropped", line)


if __name__ == "__main__":
    main()
