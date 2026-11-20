"""Prepares the video of the simulation cage (Task A7): frames at 1 fps, with a coloured mark
painted on the back of each guinea pig, and the zip the build downloads (scripts/fetch_models.py).

The guinea pigs are marked by hand, once: every --every seconds a frame is shown and you click the
back of each colour (right click = that guinea pig is not visible). The marks in between are
interpolated in a straight line. The detector looks for these colours, so the clip is only useful
with marks painted on the animals.

  python -m dev.prepare_clip VIDEO --colors RED,BLUE,GREEN --title "..." --author "..." \\
      --license "CC BY 4.0" --license-url https://creativecommons.org/licenses/by/4.0/ \\
      --source-url https://commons.wikimedia.org/wiki/File:...  [--every 5] [--out data/sim_clip]

Needs ffmpeg on the PATH and tkinter (standard library). Writes OUT/ (frames/, manifest.json) and
OUT.zip next to it, and prints the sha256 of the zip for SIM_CLIP_SHA256.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import tkinter as tk
import zipfile
from pathlib import Path

import cv2
import numpy as np

from app.contracts import MarkColor
from training.synthetic_dataset import MARK_BGR

Point = tuple[float, float]
Clicks = dict[str, dict[int, Point | None]]  # colour -> frame -> the click (None = not visible)
WIDTH = 640


def interpolate_marks(clicks: Clicks, frame_count: int) -> list[dict[str, list[int]]]:
    """The mark of each colour in each frame. Between two clicks of a colour the mark moves in a
    straight line; before the first click or after the last one, and next to a 'not visible'
    click, it is not drawn."""
    frames: list[dict[str, list[int]]] = [{} for _ in range(frame_count)]
    for color, points in clicks.items():
        ordered = sorted(points.items())
        for (start, first), (end, second) in zip(ordered, ordered[1:], strict=False):
            if first is None or second is None:
                continue
            for frame in range(start, min(end, frame_count)):
                t = (frame - start) / (end - start)
                x = first[0] + (second[0] - first[0]) * t
                y = first[1] + (second[1] - first[1]) * t
                frames[frame][color] = [round(x), round(y)]
        for frame, point in ordered:
            if point is not None and frame < frame_count:
                frames[frame][color] = [round(point[0]), round(point[1])]
    return frames


def paint_marks(image: np.ndarray, marks: dict[str, list[int]]) -> None:
    """A filled dot per mark, the colour the detector was trained with (synthetic_dataset)."""
    radius = max(8, round(0.03 * image.shape[1]))
    for color, (x, y) in marks.items():
        cv2.circle(image, (x, y), radius, MARK_BGR[MarkColor(color)], -1)


def zone_fractions(points: list[Point], width: int, height: int) -> list[list[float]]:
    return [[round(x / width, 4), round(y / height, 4)] for x, y in points]


def pack_clip(folder: Path, archive: Path) -> str:
    """Zips manifest.json and frames/ (paths relative to the clip) and returns the zip's sha256."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(folder / "manifest.json", "manifest.json")
        for frame in sorted((folder / "frames").glob("*.jpg")):
            bundle.write(frame, f"frames/{frame.name}")
    return hashlib.sha256(archive.read_bytes()).hexdigest()


class _Picker:
    """One window: shows a frame and returns the points clicked on it (blocks until done)."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("prepare_clip")
        self.label = tk.Label(self.root, font=("Arial", 12), wraplength=WIDTH)
        self.label.pack()
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._left)
        self.canvas.bind("<Button-3>", self._right)
        self.root.bind("<Return>", self._enter)
        self._done = tk.BooleanVar(self.root, value=False)
        self._points: list[Point] = []
        self._needed = 1
        self._photo: tk.PhotoImage | None = None

    def ask(self, image: np.ndarray, prompt: str, needed: int) -> list[Point]:
        """Left click adds a point; after `needed` points it returns. Right click returns [] and
        Enter returns the points so far."""
        ok, png = cv2.imencode(".png", image)
        assert ok
        self._photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))
        self.canvas.config(width=image.shape[1], height=image.shape[0])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.label.config(text=prompt)
        self._points, self._needed = [], needed
        self._done.set(False)
        self.root.wait_variable(self._done)
        return list(self._points)

    def _left(self, event: tk.Event) -> None:
        self._points.append((float(event.x), float(event.y)))
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="yellow")
        if len(self._points) >= self._needed:
            self._done.set(True)

    def _right(self, _event: tk.Event) -> None:
        self._points = []
        self._done.set(True)

    def _enter(self, _event: tk.Event) -> None:
        self._done.set(True)

    def close(self) -> None:
        self.root.destroy()


def _extract_frames(video: Path, frames: Path) -> list[Path]:
    frames.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video)]
    command += ["-vf", f"fps=1,scale={WIDTH}:-2", str(frames / "%04d.jpg")]
    subprocess.run(command, check=True)
    return sorted(frames.glob("*.jpg"))


def _collect(
    picker: _Picker, paths: list[Path], colors: list[str], every: int
) -> tuple[Clicks, dict]:
    first = cv2.imread(str(paths[0]))
    height, width = first.shape[:2]
    zones: dict[str, list[list[float]] | None] = {}
    for name in ("feeder", "waterer"):
        points = picker.ask(first, f"{name}: click its corners (Enter = none)", 4)
        zones[name] = zone_fractions(points, width, height) if len(points) >= 3 else None
    clicks: Clicks = {color: {} for color in colors}
    for index in range(0, len(paths), every):
        image = cv2.imread(str(paths[index]))
        for color in colors:
            points = picker.ask(image, f"{color}: click its back (right click = not visible)", 1)
            clicks[color][index] = points[0] if points else None
    return clicks, zones


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/sim_clip"))
    parser.add_argument("--colors", required=True, help="RED,BLUE,... in the order of the cage")
    parser.add_argument("--every", type=int, default=5, help="click every N seconds")
    parser.add_argument("--base-weight-grams", type=float, default=2700)
    parser.add_argument("--title", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--license-url", required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not on the PATH", file=sys.stderr)
        return 1
    colors = [MarkColor(name.strip().upper()).value for name in args.colors.split(",")]
    paths = _extract_frames(args.video, args.out / "frames")
    if not paths:
        print("ffmpeg made no frames", file=sys.stderr)
        return 1

    picker = _Picker()
    try:
        clicks, zones = _collect(picker, paths, colors, args.every)
    finally:
        picker.close()

    marks = interpolate_marks(clicks, len(paths))
    for path, frame_marks in zip(paths, marks, strict=True):
        image = cv2.imread(str(path))
        paint_marks(image, frame_marks)
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    manifest = {
        "title": args.title,
        "author": args.author,
        "license": args.license,
        "licenseUrl": args.license_url,
        "sourceUrl": args.source_url,
        "fps": 1,
        "colors": colors,
        "baseWeightGrams": args.base_weight_grams,
        "zones": zones,
        "frames": [
            {"file": f"frames/{path.name}", "marks": frame_marks}
            for path, frame_marks in zip(paths, marks, strict=True)
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    archive = args.out.parent / f"{args.out.name}.zip"
    print(f"{archive}\nsha256 {pack_clip(args.out, archive)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
