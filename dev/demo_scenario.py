"""Sends the drawn demo cage (app/testdata/scene.py) to the AI service over HTTP, as a camera would.

The dashboard's "Simulación" button does the same from inside the service; this script is for trying
the HTTP path from outside, for a cage the service accepts (CAGE_ID / CAGE_IDS). The first colour is
the sick guinea pig.

  python -m dev.demo_scenario --cage-id cage-1 --colors RED,BLUE,GREEN --minutes 35
"""

import argparse
import os
import time
from datetime import UTC, datetime

import httpx

from app.testdata.scenarios import parse_colors
from app.testdata.scene import draw_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cage-id", required=True)
    parser.add_argument("--colors", default="RED,BLUE,GREEN", help="the first one is drawn sick")
    parser.add_argument("--minutes", type=float, default=35)
    parser.add_argument("--interval", type=float, default=10, help="seconds between frames")
    parser.add_argument("--ai-url", default=os.getenv("AI_URL", "http://localhost:8000"))
    args = parser.parse_args()
    colors = list(parse_colors(args.colors))

    headers = {"X-API-Key": os.environ["API_KEY"]}
    frames = int(args.minutes * 60 / args.interval)
    with httpx.Client(base_url=args.ai_url, headers=headers, timeout=10) as client:
        for step in range(frames):
            response = client.post(
                "/ai/v1/frames",
                files={"file": ("frame.jpg", draw_scene(step, colors), "image/jpeg")},
                data={"capturedAt": datetime.now(UTC).isoformat(), "cageId": args.cage_id},
            )
            # a wrong cage or key stops the demo instead of sending into the void
            response.raise_for_status()
            if (step + 1) % 30 == 0:
                print(f"{step + 1} frames sent", flush=True)
            time.sleep(args.interval)
    print(f"done: {frames} frames", flush=True)


if __name__ == "__main__":
    main()
