import asyncio

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.contracts import MarkColor
from app.contracts.events import AudioLabel
from app.main import app
from app.testdata.scene import SICK_SPOT, draw_scene
from app.testdata.simulation import run_simulation

HEADERS = {"X-API-Key": "test-secret"}


class _Client:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("CAGE_ID", "cage-1")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        app.state.backend_event_client = _Client()
        yield test_client
    get_settings.cache_clear()


def _decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_the_sick_guinea_pig_stays_in_its_corner_and_the_others_move():
    colors = [MarkColor.RED, MarkColor.BLUE, MarkColor.GREEN]
    first, later = _decode(draw_scene(0, colors)), _decode(draw_scene(5, colors))

    x, y = SICK_SPOT
    corner = (slice(y - 30, y + 30), slice(x - 50, x + 10))
    # the corner barely changes (noise only) while the rest of the cage does
    assert np.abs(first[corner].astype(int) - later[corner].astype(int)).mean() < 15
    assert np.abs(first.astype(int) - later.astype(int)).mean() > 5


def test_the_simulation_feeds_one_frame_per_interval():
    frames, pauses = [], []

    async def submit(frame):
        frames.append(frame)

    async def sleep(seconds):
        pauses.append(seconds)

    demo = run_simulation("cage-1", [MarkColor.RED], submit, minutes=1, interval=10, sleep=sleep)
    asyncio.run(demo)

    assert len(frames) == 6
    assert pauses == [10] * 6


def test_the_button_starts_the_demo_for_any_cage_and_reports_a_distress_clip(client):
    response = client.post(
        "/ai/v1/simulation",
        headers=HEADERS,
        json={"cageId": "cage-0a1b2c3d", "colors": ["RED", "BLUE"]},
    )

    assert response.status_code == 202
    # not in CAGE_IDS: the backend already checked the owner
    assert "cage-0a1b2c3d" in app.state.simulations
    events = app.state.backend_event_client.events
    assert [e.payload.label for e in events] == [AudioLabel.DISTRESS]
    assert events[0].cage_id == "cage-0a1b2c3d"


def test_pressing_the_button_twice_does_not_start_a_second_demo(client):
    body = {"cageId": "cage-1", "colors": ["RED"]}
    client.post("/ai/v1/simulation", headers=HEADERS, json=body)
    first = app.state.simulations["cage-1"]

    client.post("/ai/v1/simulation", headers=HEADERS, json=body)

    assert app.state.simulations["cage-1"] is first


def test_the_demo_needs_the_api_key_and_a_known_colour(client):
    body = {"cageId": "cage-1", "colors": ["RED"]}
    assert client.post("/ai/v1/simulation", json=body).status_code == 401
    bad = {"cageId": "cage-1", "colors": ["PINK"]}
    assert client.post("/ai/v1/simulation", headers=HEADERS, json=bad).status_code == 422
