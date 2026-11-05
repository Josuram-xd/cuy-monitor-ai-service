import asyncio

import pytest

from app.contracts import MarkColor
from app.testdata.model_answers import (
    HEALTHY_GROUP,
    IMPOSSIBLE_NUMBERS,
    LETHARGIC_RED,
    NOBODY,
    NOT_JSON,
    ScriptedRuntime,
)
from app.vision.analysis import BedrockBehaviorAnalyzer

FRAMES = [b"frame"] * 6


def analyze(answer: str):
    runtime = ScriptedRuntime([answer])
    analyzer = BedrockBehaviorAnalyzer(runtime, "test-model", window_seconds=60)
    return asyncio.run(analyzer.analyze(FRAMES)), runtime


def test_a_healthy_group_gives_one_calm_window_per_guinea_pig():
    payloads, runtime = analyze(HEALTHY_GROUP)

    assert [p.color for p in payloads] == [MarkColor.RED, MarkColor.BLUE]
    assert all(p.prob_anomaly < 0.7 for p in payloads)
    assert runtime.calls == [
        {"model_id": "test-model", "text": runtime.calls[0]["text"], "images": 6}
    ]


def test_a_lethargic_guinea_pig_reaches_the_backend_as_still_and_anomalous():
    payloads, _ = analyze(LETHARGIC_RED)
    red = next(p for p in payloads if p.color is MarkColor.RED)

    assert red.still_seconds == 60
    assert red.feeder_visits == 0
    assert red.prob_anomaly == pytest.approx(0.88)
    assert red.detection_confidence == pytest.approx(0.91)


def test_impossible_or_unknown_observations_are_dropped_not_trusted():
    payloads, _ = analyze(IMPOSSIBLE_NUMBERS)

    assert payloads == []


@pytest.mark.parametrize("answer", [NOT_JSON, NOBODY])
def test_an_answer_without_guinea_pigs_sends_nothing(answer):
    payloads, _ = analyze(answer)

    assert payloads == []
