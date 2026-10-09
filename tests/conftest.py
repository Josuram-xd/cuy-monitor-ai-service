import pytest


@pytest.fixture
def anyio_backend() -> str:
    # anyio ships with FastAPI; its pytest plugin runs the async tests on asyncio
    return "asyncio"
