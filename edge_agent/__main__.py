"""python -m edge_agent: reads edge_agent/.env and runs until stopped (the service manager restarts
it)."""

import asyncio
import logging
import sys

from pydantic import ValidationError

from edge_agent.agent import run
from edge_agent.config import EdgeSettings


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        settings = EdgeSettings()
    except ValidationError as exception:
        # never print the values: one of them is the API key
        fields = ", ".join(str(error["loc"][0]).upper() for error in exception.errors())
        print(f"edge_agent/.env is missing or has invalid values: {fields}", file=sys.stderr)
        return 2
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
