"""A guinea pig's coat as a small colour histogram, to recognise it when its mark is hidden (A4).

The descriptor is the colour of the animal, not its shape: a histogram of hue and saturation for
the coloured pixels, and of brightness for the rest (white, cream, grey, black), taken from the
middle of the box so the bedding in the corners does not count. Two photos of one guinea pig give
similar histograms; no model is loaded (AGENTS: no runtime ML beyond onnxruntime).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.vision.animal_detector import OnnxAnimalDetector

# a reference photo smaller than this (the 32x32 placeholder of the simulation cage) is not a photo
MIN_REFERENCE_SIDE = 64
# the coloured pixels: saturated and not too dark (below this the hue is noise)
_MIN_SATURATION = 40
_MIN_VALUE = 40
_HUE_BINS, _SATURATION_BINS = 12, 4
_VALUE_BINS = 8
_CHROMA_SIZE = _HUE_BINS * _SATURATION_BINS
_DARK_BINS, _LIGHT_BINS = 3, 3  # the first 3 of 8 value bins are dark, the last 3 are light
_SHARE = 0.6  # a tone wins when it holds this share of the achromatic pixels
# a light coat is contradicted by a dark picture, and the other way round
_LIGHT_COATS = ("WHITE", "CREAM")
_DARK_COATS = ("BLACK",)


def describe(image_bgr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """The coat histogram of the box (pixels of image_bgr). Sums to 1; float32."""
    x1, y1, x2, y2 = (round(v) for v in box)
    crop = image_bgr[max(y1, 0) : y2, max(x1, 0) : x2]
    if crop.size == 0:
        raise ValueError("The box is empty")
    height, width = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    rows, cols = np.mgrid[0:height, 0:width]
    # an ellipse covering 90 % of the box: the corners, where the bedding is, are left out
    inside = (
        ((cols - (width - 1) / 2) / max(width * 0.45, 1)) ** 2
        + ((rows - (height - 1) / 2) / max(height * 0.45, 1)) ** 2
    ) <= 1
    hue, saturation, value = (hsv[..., i][inside] for i in range(3))

    coloured = (saturation > _MIN_SATURATION) & (value > _MIN_VALUE)
    chroma_share = coloured.mean()
    chroma, _, _ = np.histogram2d(
        hue[coloured],
        saturation[coloured],
        bins=[_HUE_BINS, _SATURATION_BINS],
        range=[[0, 180], [0, 256]],
    )
    achromatic, _ = np.histogram(value[~coloured], bins=_VALUE_BINS, range=(0, 256))
    chroma = chroma.ravel() / max(chroma.sum(), 1)
    achromatic = achromatic / max(achromatic.sum(), 1)
    return np.concatenate([chroma * chroma_share, achromatic * (1 - chroma_share)]).astype(
        np.float32
    )


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 for the same coat, 0 for coats with nothing in common (1 - Bhattacharyya distance)."""
    overlap = float(np.sqrt(a * b).sum())
    return max(0.0, 1.0 - float(np.sqrt(max(0.0, 1.0 - overlap))))


def dominant_tone(descriptor: np.ndarray) -> Literal["dark", "light", "colour", "mixed"]:
    chroma_share = float(descriptor[:_CHROMA_SIZE].sum())
    if chroma_share >= 0.5:
        return "colour"
    achromatic = descriptor[_CHROMA_SIZE:]
    total = float(achromatic.sum())
    if total <= 0:
        return "mixed"
    if achromatic[:_DARK_BINS].sum() / total >= _SHARE:
        return "dark"
    if achromatic[-_LIGHT_BINS:].sum() / total >= _SHARE:
        return "light"
    return "mixed"


def contradicts(coat_color: str | None, descriptor: np.ndarray) -> bool:
    """True when the registered coat is plainly not what the picture shows (a veto only)."""
    tone = dominant_tone(descriptor)
    if coat_color in _LIGHT_COATS:
        return tone == "dark"
    if coat_color in _DARK_COATS:
        return tone == "light"
    return False


def describe_reference(
    image_bgr: np.ndarray, animals: OnnxAnimalDetector | None
) -> np.ndarray | None:
    """The descriptor of a registered photo, or None when it is too small to say anything.

    The animal is the most confident animal box of the photo when the animal detector finds one;
    otherwise the central 80 % of the photo.
    """
    height, width = image_bgr.shape[:2]
    if min(height, width) < MIN_REFERENCE_SIDE:
        return None
    box = None
    if animals is not None:
        found = animals.detect_image(image_bgr)
        if found:
            box = max(found, key=lambda animal: animal.confidence).box
    if box is None:
        box = (0.1 * width, 0.1 * height, 0.9 * width, 0.9 * height)
    return describe(image_bgr, box)
