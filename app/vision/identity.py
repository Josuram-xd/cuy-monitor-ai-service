"""Who is who in one frame, for the guinea pigs whose mark is not visible (Task A5).

The mark detector gives a colour to each marked guinea pig. The animal detector finds every guinea
pig, marked or not. identify() joins the two, in this order, and never guesses a colour:

1. a mark inside an animal box: the animal is that colour (its body box is the better one);
2. an animal that overlaps, from the previous frame, a guinea pig of a colour not seen yet: it keeps
   that colour, with a confidence that decays (a long run without a mark goes below the threshold);
3. an animal whose coat matches the registered photo (or the gallery) of a colour not seen yet, when
   it wins clearly: the best match, with a margin over the second one, and above MIN_SIMILARITY;
4. anything else is left out, and only its centre is reported (unidentified).

The result is ordinary Detections, so the tracker and the behavior features do not change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.contracts import MarkColor
from app.vision.animal_detector import AnimalBox
from app.vision.appearance import contradicts, describe, similarity
from app.vision.detector import Detection
from app.vision.tracker import TrackedMark

logger = logging.getLogger(__name__)

# calibrated with the video clip (dev/eval_reid.py): below this a coat is not the same guinea pig
MIN_SIMILARITY = 0.6
# the second best guinea pig must be this far behind: similar coats identify nobody
MIN_MARGIN = 0.08
INHERIT_IOU = 0.3
INHERIT_DECAY = 0.9
# a mark this confident inside an animal box teaches the gallery of that guinea pig
ENROL_CONFIDENCE = 0.6


@dataclass
class Reference:
    """A guinea pig of the cage: its colour, its registered coat and the coat photos it has."""

    color: MarkColor
    coat_color: str | None
    descriptors: list[np.ndarray]


@dataclass(frozen=True)
class Identified:
    detections: list[Detection]
    # coats seen with a confident mark, for the gallery: (colour, descriptor)
    enrol: list[tuple[MarkColor, np.ndarray]]
    # centres (pixels) of the animals nobody could be identified as
    unidentified: list[tuple[float, float]]


def identify(
    image_bgr: np.ndarray,
    marks: list[Detection],
    animals: list[AnimalBox],
    references: list[Reference],
    previous: list[TrackedMark],
    min_similarity: float = MIN_SIMILARITY,
    min_margin: float = MIN_MARGIN,
) -> Identified:
    detections: list[Detection] = []
    enrol: list[tuple[MarkColor, np.ndarray]] = []
    taken = {mark.color for mark in marks}  # colours already known in this frame
    used: set[int] = set()  # animals already given a colour

    for mark in marks:
        index = _animal_under(mark, animals)
        if index is None:
            detections.append(mark)
            continue
        used.add(index)
        box = animals[index].box
        detections.append(Detection(mark.color, mark.confidence, box))
        if mark.confidence >= ENROL_CONFIDENCE:
            enrol.append((mark.color, describe(image_bgr, box)))

    for index in sorted(_free(animals, used), key=lambda i: -animals[i].confidence):
        best, best_overlap = None, INHERIT_IOU
        for track in previous:
            overlap = _iou(animals[index].box, track.box)
            if track.color not in taken and overlap >= best_overlap:
                best, best_overlap = track, overlap
        if best is not None:
            taken.add(best.color)
            used.add(index)
            confidence = round(best.confidence * INHERIT_DECAY, 3)
            detections.append(Detection(best.color, confidence, animals[index].box))

    free = _free(animals, used)
    candidates = [r for r in references if r.color not in taken and r.descriptors]
    if free and candidates:
        matches = _match_by_coat(image_bgr, animals, free, candidates, min_similarity, min_margin)
        for index, color, score in matches:
            taken.add(color)
            used.add(index)
            detections.append(Detection(color, score, animals[index].box))

    unidentified = [_centre(animals[i].box) for i in _free(animals, used)]
    if unidentified:
        logger.debug("%s animals without an identity in this frame", len(unidentified))
    return Identified(detections, enrol, unidentified)


def _match_by_coat(
    image_bgr: np.ndarray,
    animals: list[AnimalBox],
    free: list[int],
    candidates: list[Reference],
    min_similarity: float,
    min_margin: float,
) -> list[tuple[int, MarkColor, float]]:
    """Free animals to the registered coats: one-to-one, best total, clear winners only."""
    coats = {index: describe(image_bgr, animals[index].box) for index in free}
    scores = np.zeros((len(free), len(candidates)))
    for row, index in enumerate(free):
        coat = coats[index]
        for col, reference in enumerate(candidates):
            if not contradicts(reference.coat_color, coat):
                scores[row, col] = max(similarity(coat, d) for d in reference.descriptors)
    rows, cols = linear_sum_assignment(1.0 - scores)
    matches = []
    for row, col in zip(rows, cols, strict=True):
        score = float(scores[row, col])
        rival = max((scores[row, c] for c in range(len(candidates)) if c != col), default=0.0)
        if score >= min_similarity and score - rival >= min_margin:
            matches.append((free[row], candidates[col].color, round(score, 3)))
    return matches


def _animal_under(mark: Detection, animals: list[AnimalBox]) -> int | None:
    """The animal whose box holds the mark's centre; the one covering most of the mark wins."""
    cx, cy = mark.center
    best, best_key = None, None
    for index, animal in enumerate(animals):
        x1, y1, x2, y2 = animal.box
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        key = (_intersection(mark.box, animal.box), animal.confidence)
        if best_key is None or key > best_key:
            best, best_key = index, key
    return best


def _free(animals: list[AnimalBox], used: set[int]) -> list[int]:
    return [i for i in range(len(animals)) if i not in used]


def _centre(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _intersection(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(width, 0.0) * max(height, 0.0)


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    inter = _intersection(a, b)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
