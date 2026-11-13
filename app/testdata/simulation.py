"""The demo behind the dashboard's "Simulación" button: a drawn cage fed to the real frame chain.

The frames enter exactly where a camera's frames do, so the same model analyses them and the
backend applies the same rules. The backend raises a guinea pig alert only after the anomaly lasts
30 minutes (observed-to-alert), which is why the demo runs a little longer than that.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.contracts import MarkColor
from app.testdata.scene import draw_scene

logger = logging.getLogger(__name__)

SIMULATION_MINUTES = 35
FRAME_INTERVAL_SECONDS = 10

SubmitFrame = Callable[[bytes], Awaitable[object]]
Sleep = Callable[[float], Awaitable[None]]


async def run_simulation(
    cage_id: str,
    colors: list[MarkColor],
    submit: SubmitFrame,
    minutes: float = SIMULATION_MINUTES,
    interval: float = FRAME_INTERVAL_SECONDS,
    sleep: Sleep = asyncio.sleep,
) -> None:
    frames = int(minutes * 60 / interval)
    logger.info("Simulation of %s started: %s frames, %s sick", cage_id, frames, colors[0])
    for step in range(frames):
        await submit(draw_scene(step, colors))
        await sleep(interval)
    logger.info("Simulation of %s finished", cage_id)
