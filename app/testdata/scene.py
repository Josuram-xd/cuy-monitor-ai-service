"""A drawn cage, for demos: the first guinea pig is sick, the others are healthy.

The sick one stays curled up in a corner, away from the group, and never goes to the feeder. The
healthy ones go around the cage and take turns at the feeder and the water. Nothing in the picture
says who is sick: the model reading the frames has to see it.
"""

import math

import cv2
import numpy as np

from app.contracts import MarkColor

WIDTH, HEIGHT = 640, 480
FEEDER, WATER = (110, 400), (560, 120)
SICK_SPOT = (585, 430)  # bottom right corner, far from the feeder and the others
MARK_BGR: dict[MarkColor, tuple[int, int, int]] = {
    MarkColor.RED: (0, 0, 230),
    MarkColor.BLUE: (220, 60, 0),
    MarkColor.GREEN: (0, 170, 0),
    MarkColor.YELLOW: (0, 220, 240),
    MarkColor.ORANGE: (0, 140, 255),
    MarkColor.PURPLE: (160, 30, 130),
    MarkColor.BLACK: (20, 20, 20),
    MarkColor.WHITE: (250, 250, 250),
}


def _healthy_position(index: int, count: int, step: int) -> tuple[int, int]:
    # every few frames one of them is at the feeder or at the water, the rest go round the cage
    if step % 4 == 0 and index == (step // 4) % count:
        return FEEDER[0] + 45, FEEDER[1] - 35
    if step % 4 == 2 and index == (step // 4) % count:
        return WATER[0] - 40, WATER[1] + 60
    phase = step * 0.9 + 2 * math.pi * index / count
    return int(300 + 170 * math.cos(phase)), int(240 + 110 * math.sin(phase))


def draw_scene(step: int, colors: list[MarkColor]) -> bytes:
    """Frame number `step` as a JPEG. colors[0] is the sick guinea pig."""
    img = np.full((HEIGHT, WIDTH, 3), (170, 200, 215), np.uint8)  # wood shavings
    cv2.rectangle(img, (10, 10), (WIDTH - 10, HEIGHT - 10), (90, 90, 90), 8)  # cage wall
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.ellipse(img, FEEDER, (55, 30), 0, 0, 360, (40, 90, 160), -1)
    cv2.putText(img, "FOOD", (FEEDER[0] - 30, FEEDER[1] + 6), font, 0.6, (255, 255, 255), 2)
    bottle = ((WATER[0] - 15, WATER[1] - 80), (WATER[0] + 15, WATER[1] + 10))
    cv2.rectangle(img, *bottle, (230, 200, 160), -1)
    cv2.putText(img, "WATER", (WATER[0] - 40, WATER[1] - 90), font, 0.6, (120, 60, 0), 2)
    healthy = colors[1:]
    for index, color in enumerate(colors):
        sick = index == 0
        if sick:
            # a pixel of jitter: alive, but not moving
            x, y = SICK_SPOT[0] + step % 2, SICK_SPOT[1]
        else:
            x, y = _healthy_position(index - 1, len(healthy), step)
        # the sick one is curled up: rounder and smaller, head tucked in
        cv2.ellipse(img, (x, y), (40, 34) if sick else (60, 38), 0, 0, 360, (60, 110, 160), -1)
        head = (x + 30, y - 4) if sick else (x + 50, y - 10)
        cv2.circle(img, head, 12 if sick else 14, (40, 80, 120), -1)
        cv2.circle(img, (x - 5, y), 15, MARK_BGR[color], -1)  # the coloured mark on the back
    noise = np.random.default_rng(step).integers(-12, 12, img.shape)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
