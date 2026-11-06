"""Task 10.3: frames -> detector -> tracker -> window -> Random Forest -> BEHAVIOR events."""

import asyncio
import csv
import threading
from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.behavior.classifier import BehaviorClassifier
from app.behavior.engine import MAX_FRAMES_IN_FLIGHT, TrainedBehaviorEngine
from app.config import Settings, get_settings
from app.contracts import BehaviorEventPayload, EventType, MarkColor
from app.vision.detector import Detection
from app.vision.pipeline import close_trained_window_once, run_trained_behavior_pipeline
from app.vision.zones import NO_ZONES, CageZones, Zone
from training.train_behavior import main as train_main

START = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
W, H = 320, 240


def _jpeg() -> bytes:
    ok, data = cv2.imencode(".jpg", np.zeros((H, W, 3), dtype=np.uint8))
    assert ok
    return data.tobytes()


JPEG = _jpeg()


def _box(x: float, y: float) -> tuple[float, float, float, float]:
    return (x - 10, y - 10, x + 10, y + 10)


class _ScriptedDetector:
    """Returns the next list of detections each frame (then nothing)."""

    def __init__(self, frames: list[list[Detection]]):
        self.frames = list(frames)

    def detect_image(self, image_bgr):
        assert image_bgr.shape[:2] == (H, W)
        return self.frames.pop(0) if self.frames else []


class _FakeModel:
    classes_ = [0, 1]

    def predict_proba(self, rows):
        # anomaly = how still it was: enough to check the wiring
        return [[1 - r[0], r[0]] for r in rows]


class _Client:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


def _engine(detector, zones=NO_ZONES, log=None) -> TrainedBehaviorEngine:
    return TrainedBehaviorEngine(
        detector, BehaviorClassifier(_FakeModel()), lambda _: zones, 60, log
    )


@pytest.mark.anyio
async def test_frames_become_one_behavior_payload_per_guinea_pig():
    still_red = [Detection(MarkColor.RED, 0.9, _box(160, 120))]
    moving_blue = [
        Detection(MarkColor.BLUE, 0.8, _box(40 if i % 2 else 120, 200)) for i in range(10)
    ]
    detector = _ScriptedDetector([still_red + [blue] for blue in moving_blue])
    engine = _engine(detector)

    for i in range(10):
        assert await engine.submit_frame("cage-1", JPEG, START + timedelta(seconds=i))
    red, blue = await engine.close_window("cage-1")

    assert isinstance(red, BehaviorEventPayload)
    assert red.color is MarkColor.RED and red.still_seconds == 60
    assert red.prob_anomaly == 1.0
    assert blue.color is MarkColor.BLUE and blue.still_seconds == 0
    assert blue.prob_anomaly == 0.0
    assert red.detection_confidence == 0.9 and blue.detection_confidence == 0.8
    # the window starts empty again
    assert await engine.close_window("cage-1") == []


@pytest.mark.anyio
async def test_cage_zones_give_the_feeder_visits():
    feeder = Zone(((0.0, 0.0), (0.3, 0.0), (0.3, 0.3), (0.0, 0.3)))
    detector = _ScriptedDetector(
        [
            [Detection(MarkColor.GREEN, 0.9, _box(x, y))]
            for x, y in [(200, 200), (30, 30), (200, 200)]
        ]
    )
    engine = _engine(detector, zones=CageZones(feeder=feeder))

    for i in range(3):
        await engine.submit_frame("cage-1", JPEG, START + timedelta(seconds=i))
    [green] = await engine.close_window("cage-1")

    assert green.feeder_visits == 1


@pytest.mark.anyio
async def test_each_cage_has_its_own_tracker_and_window():
    detector = _ScriptedDetector([[Detection(MarkColor.RED, 0.9, _box(50, 50))], []])
    engine = _engine(detector)

    await engine.submit_frame("cage-1", JPEG, START)
    await engine.submit_frame("cage-2", JPEG, START)

    assert sorted(engine.cage_ids()) == ["cage-1", "cage-2"]
    assert [p.color for p in await engine.close_window("cage-1")] == [MarkColor.RED]
    assert await engine.close_window("cage-2") == []


@pytest.mark.anyio
async def test_an_unreadable_frame_or_a_failing_detector_loses_only_that_frame():
    class _Broken:
        def detect_image(self, image_bgr):
            raise RuntimeError("onnx exploded")

    assert (
        await _engine(_ScriptedDetector([])).submit_frame("cage-1", b"not a jpeg", START) is False
    )
    assert await _engine(_Broken()).submit_frame("cage-1", JPEG, START) is False


@pytest.mark.anyio
async def test_frames_are_dropped_not_queued_when_the_detector_falls_behind():
    release = threading.Event()

    class _Slow:
        def detect_image(self, image_bgr):
            release.wait(5)
            return []

    engine = _engine(_Slow())
    busy = [
        asyncio.create_task(engine.submit_frame("cage-1", JPEG, START))
        for _ in range(MAX_FRAMES_IN_FLIGHT)
    ]
    await asyncio.sleep(0.05)

    assert await engine.submit_frame("cage-1", JPEG, START) is False
    release.set()
    assert all(await asyncio.gather(*busy))


@pytest.mark.anyio
async def test_a_naive_captured_at_is_taken_as_utc():
    detector = _ScriptedDetector([[Detection(MarkColor.RED, 0.9, _box(50, 50))]])
    engine = _engine(detector)

    assert await engine.submit_frame("cage-1", JPEG, START.replace(tzinfo=None))
    assert len(await engine.close_window("cage-1")) == 1


@pytest.mark.anyio
async def test_the_features_log_gets_one_row_per_guinea_pig_and_window(tmp_path):
    log = tmp_path / "logs" / "windows.csv"
    detector = _ScriptedDetector([[Detection(MarkColor.RED, 0.9, _box(50, 50))]] * 2)
    engine = _engine(detector, log=log)

    for window in range(2):
        await engine.submit_frame("cage-1", JPEG, START + timedelta(minutes=window))
        await engine.close_window("cage-1")

    rows = list(csv.DictReader(log.open()))
    assert len(rows) == 2
    assert rows[0]["color"] == "RED" and rows[0]["cageId"] == "cage-1"
    assert rows[0]["label"] == ""  # to be filled by a person before retraining


@pytest.mark.anyio
async def test_a_closed_window_is_sent_as_behavior_events_for_its_cage():
    detector = _ScriptedDetector([[Detection(MarkColor.RED, 0.9, _box(50, 50))]])
    engine = _engine(detector)
    client = _Client()
    await engine.submit_frame("cage-7", JPEG, START)

    assert await close_trained_window_once(client, engine, "cage-7") == 1

    [event] = client.events
    assert event.type is EventType.BEHAVIOR
    assert event.cage_id == "cage-7"
    assert event.source == "ai-service"


@pytest.mark.anyio
async def test_a_failing_window_is_skipped_and_the_loop_goes_on():
    class _Engine:
        def __init__(self):
            self.closed = []

        def cage_ids(self):
            return ["cage-1", "cage-2"]

        async def close_window(self, cage_id):
            self.closed.append(cage_id)
            if cage_id == "cage-1":
                raise RuntimeError("boom")
            return []

    engine, client, calls = _Engine(), _Client(), []

    async def sleep(seconds):
        calls.append(seconds)
        if len(calls) > 1:
            raise asyncio.CancelledError

    settings = Settings(_env_file=None, api_key="k", window_seconds=60, mock_mode=False)
    with pytest.raises(asyncio.CancelledError):
        await run_trained_behavior_pipeline(client, engine, settings, sleep)

    assert engine.closed == ["cage-1", "cage-2"]
    assert calls == [60, 60]


@pytest.fixture
def trained_app(monkeypatch, tmp_path):
    model = tmp_path / "behavior_rf.joblib"
    assert train_main(["--synthetic", "200", "--out", str(model)]) == 0
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BEHAVIOR_MODEL_PATH", str(model))
    detector = _ScriptedDetector([[Detection(MarkColor.RED, 0.9, _box(50, 50))]])
    monkeypatch.setattr(main_module, "load_detector", lambda *_: detector)
    # the app object outlives this test: what its startup stores must not leak into the next ones
    for name in ("detector", "behavior_classifier", "behavior_engine"):
        monkeypatch.setattr(main_module.app.state, name, None, raising=False)
    get_settings.cache_clear()
    with TestClient(main_module.app) as test_client:
        main_module.app.state.backend_event_client = _Client()
        yield test_client
    get_settings.cache_clear()


def test_with_both_models_the_frames_go_through_the_trained_chain(trained_app):
    state = main_module.app.state
    assert isinstance(state.behavior_engine, TrainedBehaviorEngine)

    response = trained_app.post(
        "/ai/frames",
        headers={"X-API-Key": "test-secret"},
        data={"capturedAt": "2026-11-05T12:00:00Z", "cageId": "cage-1"},
        files={"file": ("f.jpg", JPEG, "image/jpeg")},
    )

    assert response.status_code == 202
    # not kept for Bedrock: the background task already tracked it
    assert len(state.frame_windows) == 0
    assert state.behavior_engine.cage_ids() == ["cage-1"]
    assert trained_app.get("/ai/health").json()["models"]["behavior"] is True


def test_without_the_classifier_bedrock_stays_the_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BEHAVIOR_MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(main_module, "load_detector", lambda *_: _ScriptedDetector([]))
    get_settings.cache_clear()
    with TestClient(main_module.app):
        assert main_module.app.state.behavior_engine is None
    get_settings.cache_clear()
