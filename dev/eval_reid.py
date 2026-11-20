"""Measures how well the guinea pigs are identified on a prepared clip (Task A10).

  python -m dev.eval_reid data/sim_clip --animal-model models/animal_detector.onnx \
      [--mark-model models/detector.onnx] [--min-similarity 0.6] [--min-margin 0.08]

The manifest is the ground truth: wherever a colour has a click, that guinea pig is in the frame.
Each frame goes through identify() and the tracker as the service does, with the clip's crops as
the registered coats. Then, per colour:
- animal found: the click is inside an animal box (the animal detector sees the guinea pig);
- correct: a detection of that colour holds its click;
- wrong colour: the detection of that colour holds another guinea pig's click (worse than below);
- unidentified: no detection of that colour at all.
A detection that holds no click has no ground truth and is not counted. The 60 s window confidence
the service would send is reported too (share of windows under the backend's 0.5 threshold).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from app.testdata.video import load_clip
from app.vision.animal_detector import AnimalBox, OnnxAnimalDetector
from app.vision.detector import Detection, OnnxDetector
from app.vision.identity import MIN_MARGIN, MIN_SIMILARITY, identify
from app.vision.references import ClipReferenceSource
from app.vision.tracker import HungarianTracker, TrackedMark

WINDOW_FRAMES = 60  # one window of a clip at 1 fps
BACKEND_MIN_CONFIDENCE = 0.5  # the backend's min-detection-confidence


@dataclass
class Tally:
    """Counts of one colour over the frames where its guinea pig is visible (clicked)."""

    visible: int = 0
    animal_found: int = 0
    correct: int = 0
    wrong: int = 0
    unidentified: int = 0
    window_confidences: list[float] = field(default_factory=list)


def _inside(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def score_frame(
    clicks: dict[str, tuple[float, float]],
    detections: list[Detection],
    animals: list[AnimalBox],
) -> dict[str, Tally]:
    """What one frame says about each visible guinea pig (clicks: colour -> pixel point)."""
    tallies: dict[str, Tally] = {}
    for color, point in clicks.items():
        tally = tallies.setdefault(color, Tally())
        tally.visible += 1
        if any(_inside(point, animal.box) for animal in animals):
            tally.animal_found += 1
        own = [d for d in detections if d.color.value == color]
        if any(_inside(point, d.box) for d in own):
            tally.correct += 1
        elif any(_inside(c, d.box) for d in own for other, c in clicks.items() if other != color):
            tally.wrong += 1
        elif not own:
            tally.unidentified += 1
    return tallies


def window_confidences(frames: list[list[TrackedMark]]) -> dict[str, list[float]]:
    """The detectionConfidence each colour would get per 60 s window (as BehaviorFeatures does)."""
    values: dict[str, list[float]] = {}
    for start in range(0, len(frames), WINDOW_FRAMES):
        window = frames[start : start + WINDOW_FRAMES]
        for color in {m.color.value for frame in window for m in frame}:
            seen = [m.confidence for frame in window for m in frame if m.color.value == color]
            share = len(seen) / len(window)
            values.setdefault(color, []).append(round(min(1.0, share * statistics.fmean(seen)), 2))
    return values


def evaluate(
    clip_dir: Path,
    animals: OnnxAnimalDetector,
    marks: OnnxDetector | None = None,
    min_similarity: float = MIN_SIMILARITY,
    min_margin: float = MIN_MARGIN,
) -> dict[str, Tally]:
    clip = load_clip(clip_dir)
    if clip is None:
        raise ValueError(f"No prepared clip in {clip_dir}")
    manifest = json.loads((clip_dir / "manifest.json").read_text(encoding="utf-8"))
    references = asyncio.run(ClipReferenceSource(clip.references, animals).references("sim"))

    tallies: dict[str, Tally] = {}
    tracker = HungarianTracker()
    previous: list[TrackedMark] = []
    tracked_frames: list[list[TrackedMark]] = []
    for path, entry in zip(clip.frames, manifest["frames"], strict=True):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        found_marks = marks.detect_image(image) if marks is not None else []
        found_animals = animals.detect_image(image)
        identified = identify(
            image,
            found_marks,
            found_animals,
            references,
            previous,
            min_similarity=min_similarity,
            min_margin=min_margin,
        )
        previous = tracker.update(identified.detections, width, height)
        tracked_frames.append(previous)
        clicks = {color: (float(x), float(y)) for color, (x, y) in entry["marks"].items()}
        for color, tally in score_frame(clicks, identified.detections, found_animals).items():
            total = tallies.setdefault(color, Tally())
            for name in ("visible", "animal_found", "correct", "wrong", "unidentified"):
                setattr(total, name, getattr(total, name) + getattr(tally, name))
    for color, values in window_confidences(tracked_frames).items():
        tallies.setdefault(color, Tally()).window_confidences = values
    return tallies


def format_report(tallies: dict[str, Tally]) -> str:
    lines = [
        f"{'colour':8} {'visible':>7} {'animal':>7} {'correct':>8} {'wrong':>6} {'unident.':>9}"
        f" {'win.conf':>9} {'windows<0.5':>12}"
    ]
    for color in sorted(tallies):
        t = tallies[color]
        mean = statistics.fmean(t.window_confidences) if t.window_confidences else 0.0
        low = sum(c < BACKEND_MIN_CONFIDENCE for c in t.window_confidences)
        lines.append(
            f"{color:8} {t.visible:7} {t.animal_found:7} {t.correct:8} {t.wrong:6}"
            f" {t.unidentified:9} {mean:9.2f} {low:>5}/{len(t.window_confidences):<6}"
        )
    seen = sum(t.correct + t.wrong for t in tallies.values())
    if seen:
        wrong = sum(t.wrong for t in tallies.values())
        lines.append(f"wrong colour among the identified: {wrong / seen:.1%} (aim: under 10%)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("clip", type=Path, help="the prepared clip folder (dev/prepare_clip.py)")
    parser.add_argument("--animal-model", type=Path, required=True)
    parser.add_argument("--mark-model", type=Path)
    parser.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY)
    parser.add_argument("--min-margin", type=float, default=MIN_MARGIN)
    args = parser.parse_args(argv)
    for path in (args.animal_model, args.mark_model):
        if path is not None and not path.is_file():
            parser.error(f"{path} does not exist")

    animals = OnnxAnimalDetector.load(args.animal_model)
    marks = OnnxDetector.load(args.mark_model) if args.mark_model else None
    tallies = evaluate(args.clip, animals, marks, args.min_similarity, args.min_margin)
    print(format_report(tallies))
    return 0


if __name__ == "__main__":
    sys.exit(main())
