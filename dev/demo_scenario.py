"""Sends a drawn cage scene to the AI service as a live camera would, so a demo shows the real
chain working: the frames go through /ai/frames, the Bedrock model reads them, the backend decides.

The scene has three marked guinea pigs. RED is sick: it stays hunched in a corner, away from the
others, and never goes to the feeder. BLUE and GREEN wander, eat and drink. No verdict is sent: the
model has to see it, and the backend raises the alert only after the anomaly lasts (30 min default).

Run it inside the ai-service container (it reads API_KEY there), for a cage the service accepts:

  python -m dev.demo_scenario --cage-id cage-0a1b2c3d --minutes 45
"""

import argparse
import math
import os
import time
from datetime import UTC, datetime

import cv2
import httpx
import numpy as np

WIDTH, HEIGHT = 640, 480
FEEDER, WATER = (110, 400), (560, 120)
MARKS = {"RED": (0, 0, 230), "BLUE": (220, 60, 0), "GREEN": (0, 170, 0)}  # BGR
SICK_SPOT = (585, 430)  # bottom right corner, far from the feeder and the others


def _position(color: str, step: int) -> tuple[int, int]:
    if color == "RED":
        # a pixel or two of jitter: alive, but not moving
        return SICK_SPOT[0] + step % 2, SICK_SPOT[1]
    # the healthy ones go around the cage and pass by the feeder (BLUE) or the water (GREEN)
    phase = step * 0.9 + (0 if color == "BLUE" else math.pi)
    if color == "BLUE" and step % 4 == 0:
        return FEEDER[0] + 45, FEEDER[1] - 35
    if color == "GREEN" and step % 4 == 2:
        return WATER[0] - 40, WATER[1] + 60
    return int(300 + 170 * math.cos(phase)), int(240 + 110 * math.sin(phase))


def draw_frame(step: int) -> bytes:
    img = np.full((HEIGHT, WIDTH, 3), (170, 200, 215), np.uint8)  # wood shavings
    cv2.rectangle(img, (10, 10), (WIDTH - 10, HEIGHT - 10), (90, 90, 90), 8)  # cage wall
    cv2.ellipse(img, FEEDER, (55, 30), 0, 0, 360, (40, 90, 160), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "FOOD", (FEEDER[0] - 30, FEEDER[1] + 6), font, 0.6, (255, 255, 255), 2)
    bottle = ((WATER[0] - 15, WATER[1] - 80), (WATER[0] + 15, WATER[1] + 10))
    cv2.rectangle(img, *bottle, (230, 200, 160), -1)
    cv2.putText(img, "WATER", (WATER[0] - 40, WATER[1] - 90), font, 0.6, (120, 60, 0), 2)
    for color, mark in MARKS.items():
        x, y = _position(color, step)
        sick = color == "RED"
        # the sick one is curled up: rounder and smaller, head tucked in
        axes = (40, 34) if sick else (60, 38)
        cv2.ellipse(img, (x, y), axes, 0, 0, 360, (60, 110, 160), -1)
        head = (x + 30, y - 4) if sick else (x + 50, y - 10)
        cv2.circle(img, head, 12 if sick else 14, (40, 80, 120), -1)
        cv2.circle(img, (x - 5, y), 15, mark, -1)  # coloured mark on the back
    noise = np.random.default_rng(step).integers(-12, 12, img.shape)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cage-id", required=True)
    parser.add_argument("--minutes", type=float, default=45)
    parser.add_argument("--interval", type=float, default=10, help="seconds between frames")
    parser.add_argument("--ai-url", default=os.getenv("AI_URL", "http://localhost:8000"))
    args = parser.parse_args()

    headers = {"X-API-Key": os.environ["API_KEY"]}
    deadline = time.monotonic() + args.minutes * 60
    step = 0
    with httpx.Client(base_url=args.ai_url, headers=headers, timeout=10) as client:
        while time.monotonic() < deadline:
            response = client.post(
                "/ai/frames",
                files={"file": ("frame.jpg", draw_frame(step), "image/jpeg")},
                data={"capturedAt": datetime.now(UTC).isoformat(), "cageId": args.cage_id},
            )
            # a wrong cage or key stops the demo instead of sending into the void
            response.raise_for_status()
            step += 1
            if step % 30 == 0:
                print(f"{step} frames sent", flush=True)
            time.sleep(args.interval)
    print(f"done: {step} frames", flush=True)


if __name__ == "__main__":
    main()
