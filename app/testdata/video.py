"""The video simulation: a prepared guinea pig clip replayed through the real frame chain.

A simulated cage has a code starting with SIM_CAGE_PREFIX (the backend generates it). Its events
carry source "simulation", so the dashboard and the logs can tell them from a real cage's.

The clip is a folder made by dev/prepare_clip.py: manifest.json plus one JPEG per second, with
the coloured marks already painted on the guinea pigs' backs. The service only reads JPEGs.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from app.contracts import MarkColor
from app.testdata.audio import ClipKind
from app.testdata.simulation import SubmitFrame
from app.vision.zones import NO_ZONES, CageZones, parse_cage_zones

logger = logging.getLogger(__name__)

SIM_CAGE_PREFIX = "sim-"
CREDIT_KEYS = ("title", "author", "license", "licenseUrl", "sourceUrl")

VIDEO_MINUTES = 60
MAX_VIDEO_SIMULATIONS = 2  # the EC2 has two vCPUs: more would slow the real cages down
WINDOW_SECONDS = 60
# the scripted sensors: about four minutes in, the weight starts to fall and the distress calls
# start, so a cage alert shows up during the demo (the frames themselves are not scripted)
SCRIPT_START_WINDOW = 4
SCRIPT_DROP_GRAMS = 25.0
SCRIPT_MAX_DROP_STEPS = 3  # the weight stops falling after 75 g
SCRIPT_SQUEAL_WINDOWS = 2  # distress calls in the first two windows of the drop
CALM_CLIPS = (ClipKind.SILENCE, ClipKind.RUSTLE, ClipKind.CHIRP)

Sleep = Callable[[float], Awaitable[None]]
ReportWindow = Callable[[], Awaitable[None]]
Progress = Callable[[int], None]


def is_simulated(cage_id: str) -> bool:
    return cage_id.startswith(SIM_CAGE_PREFIX)


def source_for(cage_id: str) -> str:
    return "simulation" if is_simulated(cage_id) else "ai-service"


@dataclass(frozen=True)
class VideoClip:
    frames: tuple[Path, ...]
    colors: tuple[MarkColor, ...]
    zones: CageZones
    credit: dict[str, str]
    base_weight_grams: float


def load_clip(directory: Path) -> VideoClip | None:
    """The clip in directory, or None (logged) when it is missing or broken: never a fatal error."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        logger.info("No simulation clip in %s: the video simulation is off", directory)
        return None
    try:
        return _parse_clip(directory, json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exception:
        logger.warning("The simulation clip in %s is broken: %s", directory, exception)
        return None


def _parse_clip(directory: Path, manifest: dict) -> VideoClip:
    colors = tuple(MarkColor(name) for name in manifest["colors"])
    frames = tuple(directory / entry["file"] for entry in manifest["frames"])
    if not colors or not frames:
        raise ValueError("the clip needs colors and frames")
    missing = [str(frame) for frame in frames if not frame.is_file()]
    if missing:
        raise ValueError(f"missing frames, e.g. {missing[0]}")
    zones = parse_cage_zones(json.dumps({"sim": manifest.get("zones") or {}})).get("sim", NO_ZONES)
    base = float(manifest["baseWeightGrams"])
    if base <= 0:
        raise ValueError("baseWeightGrams must be positive")
    return VideoClip(
        frames=frames,
        colors=colors,
        zones=zones,
        credit={key: str(manifest[key]) for key in CREDIT_KEYS},
        base_weight_grams=base,
    )


def scripted_window(window: int, base_grams: float, rng: random.Random) -> tuple[float, ClipKind]:
    """The weight and the audio clip of window number `window` (0 is the first one sent)."""
    if window < SCRIPT_START_WINDOW:
        return round(base_grams + rng.uniform(-3, 3), 1), rng.choice(CALM_CLIPS)
    steps = min(window - SCRIPT_START_WINDOW + 1, SCRIPT_MAX_DROP_STEPS)
    grams = round(base_grams + rng.uniform(-3, 3) - steps * SCRIPT_DROP_GRAMS, 1)
    squeal = window < SCRIPT_START_WINDOW + SCRIPT_SQUEAL_WINDOWS
    return max(grams, 0.0), ClipKind.SQUEAL if squeal else rng.choice(CALM_CLIPS)


async def run_video_simulation(
    cage_id: str,
    clip: VideoClip,
    submit: SubmitFrame,
    report_window: ReportWindow,
    progress: Progress,
    minutes: float = VIDEO_MINUTES,
    interval: float = 1.0,
    sleep: Sleep = asyncio.sleep,
) -> None:
    """Replays the clip's frames in a loop, one per interval, for `minutes`. Every window of frames
    closes with report_window (the scripted weight and sound, see scripted_window)."""
    frames_per_window = round(WINDOW_SECONDS / interval)
    total = int(minutes * 60 / interval)
    logger.info("Video simulation of %s started: %s frames", cage_id, total)
    for sent, path in enumerate(itertools.islice(itertools.cycle(clip.frames), total), start=1):
        await submit(path.read_bytes())
        progress(sent)
        if sent % frames_per_window == 0:
            await report_window()
        await sleep(interval)
    logger.info("Video simulation of %s finished", cage_id)
