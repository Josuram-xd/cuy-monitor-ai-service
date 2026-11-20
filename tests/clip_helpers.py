"""A tiny prepared clip folder (two frames, manifest.json) for the video simulation tests."""

import json

import cv2
import numpy as np


def write_clip(directory, colors=("RED", "BLUE"), references=False):
    (directory / "frames").mkdir(parents=True)
    for number in (1, 2):
        image = np.zeros((48, 64, 3), np.uint8)
        ok, jpeg = cv2.imencode(".jpg", image)
        assert ok
        (directory / f"frames/000{number}.jpg").write_bytes(jpeg.tobytes())
    manifest = {
        "title": "Guinea pigs",
        "author": "A. Author",
        "license": "CC BY 4.0",
        "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
        "sourceUrl": "https://commons.wikimedia.org/wiki/File:x.webm",
        "fps": 1,
        "colors": list(colors),
        "baseWeightGrams": 2700,
        "zones": {"feeder": [[0.05, 0.6], [0.3, 0.6], [0.3, 0.95]], "waterer": None},
        "frames": [
            {"file": "frames/0001.jpg", "marks": {}},
            {"file": "frames/0002.jpg", "marks": {}},
        ],
    }
    if references:
        (directory / "refs").mkdir()
        for color in colors:
            cv2.imwrite(str(directory / "refs" / f"{color}.jpg"), np.zeros((48, 48, 3), np.uint8))
        manifest["references"] = {color: f"refs/{color}.jpg" for color in colors}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
