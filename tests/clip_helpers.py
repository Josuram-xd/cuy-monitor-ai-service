"""A tiny prepared clip folder (two frames, manifest.json) for the video simulation tests."""

import json

import cv2
import numpy as np


def write_clip(directory, colors=("RED", "BLUE")):
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
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
