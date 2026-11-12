"""Inject test data to try the models end to end.

  behavior  sends the windows of a scenario straight to the backend, as the AI service would
            (healthy, lethargic, distress, recovery), so the anomaly chain, the states and the
            alerts can be watched in the dashboard.
  audio     posts synthetic clips to the AI service (real mode), so the audio classifier runs
            through its API and its verdict reaches the backend.

Examples:
  python -m dev.inject_test_data behavior --scenario lethargic --windows 12 --interval 5
  python -m dev.inject_test_data audio --kind squeal --count 3
"""

import argparse
import asyncio
import logging
import os
from datetime import UTC, datetime

import httpx
from pydantic import SecretStr

from app.config import Settings
from app.messaging.backend_client import BackendEventClient
from app.mock.producer import scenario_events
from app.testdata.audio import EXPECTED_LABEL, ClipKind, synthesize_clip
from app.testdata.scenarios import SCENARIOS, build_scenario, parse_colors

logger = logging.getLogger("cuy-monitor-inject")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("can't be negative")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inject test data for the Cuy Monitor models.")
    parser.add_argument(
        "--api-key", default=os.getenv("API_KEY"), help="ingestion API key (API_KEY)"
    )
    parser.add_argument("--cage-id", default=os.getenv("CAGE_ID", "cage-1"))
    commands = parser.add_subparsers(dest="command", required=True)

    behavior = commands.add_parser("behavior", help="send a scenario to the backend")
    behavior.add_argument("--scenario", choices=sorted(SCENARIOS), default="lethargic")
    behavior.add_argument(
        "--colors", default="RED,BLUE,GREEN,YELLOW", help="mark colours in the cage"
    )
    behavior.add_argument("--windows", type=_positive_int, default=10)
    behavior.add_argument(
        "--interval",
        type=_non_negative_float,
        default=60.0,
        help="seconds between windows; the backend drops events older than 10 minutes",
    )
    behavior.add_argument(
        "--backend-url", default=os.getenv("BACKEND_URL", "http://localhost:8080")
    )

    audio = commands.add_parser("audio", help="post synthetic clips to the AI service")
    audio.add_argument("--kind", choices=[kind.value for kind in ClipKind], default=ClipKind.SQUEAL)
    audio.add_argument("--count", type=_positive_int, default=1)
    audio.add_argument("--ai-url", default=os.getenv("AI_URL", "http://localhost:8000"))
    return parser


async def inject_behavior(args: argparse.Namespace) -> int:
    scenario = build_scenario(args.scenario, parse_colors(args.colors))
    settings = Settings(
        api_key=SecretStr(args.api_key),
        backend_url=args.backend_url,
        cage_id=args.cage_id,
        _env_file=None,
    )
    client = BackendEventClient(settings)
    sent = 0
    logger.info("%s: %s", scenario.name, scenario.description)
    try:
        for window in range(args.windows):
            for event in scenario_events(scenario, window, args.cage_id):
                await client.enqueue_event(event)
                sent += 1
            while client.pending_count:
                if not await client.flush_pending_once():
                    await asyncio.sleep(1)
            logger.info("window %s/%s sent", window + 1, args.windows)
            if window + 1 < args.windows:
                await asyncio.sleep(args.interval)
    finally:
        await client.close()
    return sent


def inject_audio(args: argparse.Namespace) -> list[int]:
    kind = ClipKind(args.kind)
    statuses = []
    with httpx.Client(
        base_url=args.ai_url, headers={"X-API-Key": args.api_key}, timeout=10
    ) as http:
        for index in range(args.count):
            response = http.post(
                "/ai/v1/audio",
                files={"file": (f"{kind}.wav", synthesize_clip(kind, seed=index), "audio/wav")},
                data={"capturedAt": datetime.now(UTC).isoformat(), "cageId": args.cage_id},
            )
            statuses.append(response.status_code)
            logger.info(
                "%s clip %s -> %s (expected %s)",
                kind,
                index + 1,
                response.status_code,
                EXPECTED_LABEL[kind],
            )
    return statuses


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_parser().parse_args(argv)
    if not args.api_key:
        raise SystemExit("An API key is needed: --api-key or API_KEY")
    if args.command == "behavior":
        sent = asyncio.run(inject_behavior(args))
        logger.info("%s events sent to %s", sent, args.backend_url)
    else:
        inject_audio(args)


if __name__ == "__main__":
    main()
