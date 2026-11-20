"""python -m edge_agent: reads edge_agent/.env and runs until stopped (the service manager restarts
it)."""

import asyncio
import logging
import sys

from pydantic import ValidationError

from edge_agent.agent import run
from edge_agent.config import EdgeSettings
from edge_agent.rtsp import FfmpegMissing


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        settings = EdgeSettings()
    except ValidationError as exception:
        # never print the values: one of them is the API key
        # a model-level error (no camera at all) has no field name: its message names the fields
        fields = ", ".join(
            str(error["loc"][0]).upper() if error["loc"] else error["msg"]
            for error in exception.errors()
        )
        print(f"edge_agent/.env is missing or has invalid values: {fields}", file=sys.stderr)
        return 2
    try:
        asyncio.run(run(settings))
    except FfmpegMissing as exception:
        print(exception, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
