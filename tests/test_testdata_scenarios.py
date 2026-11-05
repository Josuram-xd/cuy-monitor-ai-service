import asyncio

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.contracts import EventType, MarkColor
from app.contracts.events import AudioLabel
from app.mock.producer import run_scenario_producer, scenario_events
from app.testdata.scenarios import (
    SCENARIOS,
    Healthy,
    Lethargic,
    Recovery,
    build_scenario,
    parse_colors,
)

# the backend's default profile and thresholds (app.health): what the scenarios are written against
DEFAULT_STILL_SECONDS = 30
STILL_FACTOR = 1.5
CLASSIFIER_THRESHOLD = 0.7


def looks_anomalous(payload) -> bool:
    return payload.prob_anomaly >= CLASSIFIER_THRESHOLD or (
        payload.still_seconds > DEFAULT_STILL_SECONDS * STILL_FACTOR and payload.feeder_visits == 0
    )


def test_a_healthy_cage_never_looks_anomalous():
    scenario = Healthy()

    windows = [payload for window in range(50) for payload in scenario.behavior(window)]

    assert len(windows) == 200
    assert not any(looks_anomalous(payload) for payload in windows)
    assert all(payload.detection_confidence >= 0.5 for payload in windows)


def test_only_the_sick_guinea_pig_turns_anomalous_and_from_its_window_on():
    scenario = Lethargic(sick=MarkColor.BLUE, starts_at=3)

    before = {p.color: looks_anomalous(p) for p in scenario.behavior(2)}
    after = {p.color: looks_anomalous(p) for p in scenario.behavior(3)}

    assert not any(before.values())
    assert after == {
        MarkColor.RED: False,
        MarkColor.BLUE: True,
        MarkColor.GREEN: False,
        MarkColor.YELLOW: False,
    }


def test_recovery_is_lethargic_first_and_healthy_afterwards():
    scenario = Recovery(sick_windows=4)

    sick = [looks_anomalous(scenario.behavior(w)[0]) for w in range(4)]
    well = [looks_anomalous(scenario.behavior(w)[0]) for w in range(4, 12)]

    assert all(sick)
    assert not any(well)


def test_distress_reports_distress_calls_every_window():
    scenario = build_scenario("distress")

    assert all(scenario.audio(w).label is AudioLabel.DISTRESS for w in range(5))
    assert not any(looks_anomalous(p) for p in scenario.behavior(0))


def test_a_scenario_replays_the_same_numbers():
    assert Lethargic().behavior(5) == Lethargic().behavior(5)


def test_scenarios_are_built_by_name_for_the_colours_of_the_cage():
    scenario = build_scenario("lethargic", parse_colors("red, purple"), window_seconds=30)

    assert sorted(SCENARIOS) == ["distress", "healthy", "lethargic", "recovery"]
    assert [p.color for p in scenario.behavior(0)] == [MarkColor.RED, MarkColor.PURPLE]
    assert all(p.window_seconds == 30 for p in scenario.behavior(9))
    with pytest.raises(ValueError, match="Unknown scenario"):
        build_scenario("sleepy")
    with pytest.raises(ValueError, match="RED"):
        build_scenario("lethargic", parse_colors("BLUE"))


def test_the_events_of_a_window_follow_the_contract():
    events = scenario_events(build_scenario("distress"), 0, "cage-7")

    assert [e.type for e in events] == [EventType.BEHAVIOR] * 4 + [EventType.AUDIO]
    assert {e.cage_id for e in events} == {"cage-7"}
    assert len({e.event_id for e in events}) == len(events)
    body = events[0].model_dump(mode="json", by_alias=True)
    assert {"eventId", "cageId", "schemaVersion", "payload"} <= body.keys()
    assert "stillSeconds" in body["payload"]


def test_the_scenario_producer_sends_one_window_per_period_to_every_cage():
    async def run_test():
        settings = Settings(
            api_key=SecretStr("test-secret"),
            cage_id="cage-a",
            cage_ids="cage-a,cage-b",
            window_seconds=60,
            _env_file=None,
        )

        class RecordingClient:
            def __init__(self):
                self.events = []

            async def enqueue_event(self, event):
                self.events.append(event)

        client = RecordingClient()
        sleeps = 0

        async def stop_after_two_windows(delay: float) -> None:
            nonlocal sleeps
            sleeps += 1
            if sleeps == 3:
                raise StopAsyncIteration

        scenario = build_scenario("lethargic", parse_colors("RED,BLUE"))
        with pytest.raises(StopAsyncIteration):
            await run_scenario_producer(client, settings, scenario, sleep=stop_after_two_windows)

        return client.events

    events = asyncio.run(run_test())

    # 2 windows x 2 cages x (2 behavior + 1 audio)
    assert len(events) == 12
    assert {e.cage_id for e in events} == {"cage-a", "cage-b"}
