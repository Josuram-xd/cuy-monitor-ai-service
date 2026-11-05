"""Synthetic labelled frames: guinea pigs with a coloured mark on bedding, under different lights.

They are not a substitute for real frames (Task 6.1/6.3): they let the whole chain run before the
cage is recorded (prepare_dataset -> train_detector -> detector.onnx -> OnnxDetector) and give the
tests real images with known boxes. Nothing generated here is committed.

  python -m training.synthetic_dataset --out /tmp/synthetic --frames 200
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.contracts import MarkColor
from app.vision.classes import DETECTOR_CLASSES, class_id

WIDTH, HEIGHT = 640, 480

# BGR of each mark, like a paint dot or a pompom
MARK_BGR: dict[MarkColor, tuple[int, int, int]] = {
    MarkColor.RED: (40, 40, 220),
    MarkColor.BLUE: (200, 90, 30),
    MarkColor.GREEN: (60, 170, 50),
    MarkColor.YELLOW: (40, 220, 235),
    MarkColor.ORANGE: (30, 140, 245),
    MarkColor.PURPLE: (150, 50, 130),
    MarkColor.BLACK: (25, 25, 25),
    MarkColor.WHITE: (245, 245, 245),
}
COATS_BGR = ((60, 90, 140), (200, 215, 225), (40, 50, 70), (110, 150, 200))


@dataclass(frozen=True)
class Box:
    color: MarkColor
    x1: float
    y1: float
    x2: float
    y2: float

    def yolo_line(self) -> str:
        cx = (self.x1 + self.x2) / 2 / WIDTH
        cy = (self.y1 + self.y2) / 2 / HEIGHT
        w = (self.x2 - self.x1) / WIDTH
        h = (self.y2 - self.y1) / HEIGHT
        return f"{class_id(self.color)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _bedding(rng: np.random.Generator) -> np.ndarray:
    base = np.array([150, 190, 215], dtype=np.float32)
    noise = rng.normal(0, 28, (HEIGHT, WIDTH, 1)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.6)[..., None]
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _overlaps(box: tuple[int, int, int, int], taken: list[tuple[int, int, int, int]]) -> bool:
    x1, y1, x2, y2 = box
    return any(not (x2 < a1 or x1 > a2 or y2 < b1 or y1 > b2) for a1, b1, a2, b2 in taken)


def render_frame(seed: int, max_guinea_pigs: int = 4) -> tuple[np.ndarray, list[Box]]:
    """One frame and the box of each guinea pig in it. Same seed, same frame."""
    rng = np.random.default_rng(seed)
    pick = random.Random(seed)
    image = _bedding(rng)
    # feeder and water bottle, so the model also learns what is not a guinea pig
    cv2.rectangle(image, (20, 20), (130, 90), (120, 120, 120), -1)
    cv2.rectangle(image, (WIDTH - 60, 30), (WIDTH - 35, 160), (200, 180, 160), -1)

    colors = pick.sample(DETECTOR_CLASSES, pick.randint(1, max_guinea_pigs))
    taken: list[tuple[int, int, int, int]] = []
    boxes: list[Box] = []
    for color in colors:
        for _ in range(30):
            ax, ay = pick.randint(45, 75), pick.randint(28, 45)
            cx, cy = (
                pick.randint(ax + 10, WIDTH - ax - 10),
                pick.randint(ay + 110, HEIGHT - ay - 10),
            )
            box = (cx - ax, cy - ay, cx + ax, cy + ay)
            if not _overlaps(box, taken):
                break
        else:
            continue
        taken.append(box)
        angle = pick.uniform(-30, 30)
        cv2.ellipse(image, (cx, cy), (ax, ay), angle, 0, 360, pick.choice(COATS_BGR), -1)
        mark = max(8, ay // 2)
        cv2.circle(image, (cx, cy), mark, MARK_BGR[color], -1)
        boxes.append(Box(color, *box))

    # a different hour: brighter or darker, warmer or colder
    gain = pick.uniform(0.6, 1.25)
    tint = np.array([pick.uniform(0.9, 1.1), 1.0, pick.uniform(0.9, 1.1)], dtype=np.float32)
    image = np.clip(image.astype(np.float32) * gain * tint, 0, 255).astype(np.uint8)
    return image, boxes


def write_dataset(out: Path, frames: int, seed: int = 0) -> int:
    """Writes images/*.jpg, labels/*.txt and classes.txt the way Label Studio exports YOLO."""
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(c.value.lower() for c in DETECTOR_CLASSES) + "\n")
    for index in range(frames):
        image, boxes = render_frame(seed + index)
        name = f"synthetic_{index:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (out / "labels" / f"{name}.txt").write_text("".join(b.yolo_line() + "\n" for b in boxes))
    return frames


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    count = write_dataset(args.out, args.frames, args.seed)
    print(f"{count} synthetic frames written to {args.out}")


if __name__ == "__main__":
    main()
