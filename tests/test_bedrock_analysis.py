import json

import pytest
from botocore.exceptions import ClientError

from app.bedrock.runtime import BedrockError, BedrockRuntime
from app.contracts import BehaviorEventPayload, MarkColor
from app.vision.analysis import (
    SYSTEM_PROMPT,
    BedrockBehaviorAnalyzer,
    Observation,
    parse_observations,
    to_payload,
)


def _item(**overrides):
    item = {
        "color": "RED",
        "framesSeen": 5,
        "framesStill": 3,
        "feederVisits": 1,
        "waterVisits": 0,
        "avgGroupDistance": 0.4,
        "anomaly": 0.1,
        "confidence": 0.9,
    }
    return item | overrides


def _answer(*items) -> str:
    return json.dumps({"guineaPigs": list(items)})


def test_a_valid_answer_becomes_observations():
    observations = parse_observations(_answer(_item(), _item(color="BLUE")), frame_count=6)

    assert [o.color for o in observations] == [MarkColor.RED, MarkColor.BLUE]
    assert observations[0].frames_still == 3


def test_json_wrapped_in_text_or_a_code_fence_is_found():
    text = "Here you go:\n```json\n" + _answer(_item()) + "\n```"

    assert len(parse_observations(text, frame_count=6)) == 1


@pytest.mark.parametrize("answer", ["", "no json here", "{not json}", '{"guineaPigs": "x"}', "[]"])
def test_a_bad_answer_gives_nothing_instead_of_failing(answer):
    assert parse_observations(answer, frame_count=6) == []


@pytest.mark.parametrize(
    "bad_item",
    [
        _item(color="PINK"),
        _item(color=None),
        _item(anomaly=1.5),
        _item(confidence=-0.1),
        _item(framesSeen=0),
        _item(framesSeen=9),  # more frames than were sent
        _item(framesStill=6),  # still in more frames than it was seen
        _item(feederVisits=-1),
        "RED",
    ],
)
def test_impossible_or_invalid_items_are_dropped_one_by_one(bad_item):
    observations = parse_observations(_answer(bad_item, _item(color="GREEN")), frame_count=6)

    assert [o.color for o in observations] == [MarkColor.GREEN]


def test_a_colour_repeated_by_the_model_keeps_the_better_supported_one():
    observations = parse_observations(
        _answer(_item(framesSeen=2, framesStill=1), _item(framesSeen=6, framesStill=4)),
        frame_count=6,
    )

    assert len(observations) == 1
    assert observations[0].frames_seen == 6


def test_the_payload_follows_the_event_contract():
    observation = Observation.model_validate(_item(framesSeen=5, framesStill=3, confidence=0.9))

    payload = to_payload(observation, frame_count=6, window_seconds=60)

    assert isinstance(payload, BehaviorEventPayload)
    assert payload.still_seconds == 36.0  # 3 of 5 frames still, over 60 s
    assert payload.feeder_visits == 1
    assert payload.waterer_visits == 0
    assert payload.prob_anomaly == 0.1
    # seen in 5 of 6 frames and 90% sure
    assert payload.detection_confidence == 0.75
    assert payload.window_seconds == 60


def test_a_guinea_pig_seen_in_one_frame_gets_a_low_confidence():
    observation = Observation.model_validate(_item(framesSeen=1, framesStill=0, confidence=1))

    assert to_payload(observation, frame_count=6, window_seconds=60).detection_confidence == 0.17


class _FakeRuntime:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls: list[dict] = []

    async def converse(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.answer


@pytest.mark.anyio
async def test_the_analyzer_sends_the_frames_and_returns_contract_payloads():
    runtime = _FakeRuntime(_answer(_item(), _item(color="ORANGE", framesStill=0)))
    analyzer = BedrockBehaviorAnalyzer(runtime, "model-x", window_seconds=60)
    frames = [b"a", b"b", b"c", b"d", b"e", b"f"]

    payloads = await analyzer.analyze(frames)

    assert [p.color for p in payloads] == [MarkColor.RED, MarkColor.ORANGE]
    call = runtime.calls[0]
    assert call["model_id"] == "model-x"
    assert call["images"] == frames
    assert call["system"] == SYSTEM_PROMPT
    assert "6 frames" in call["text"]


class _FakeBoto:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs: dict = {}

    def converse(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


@pytest.mark.anyio
async def test_the_runtime_builds_a_converse_request_with_jpeg_images():
    boto = _FakeBoto({"output": {"message": {"content": [{"text": "he"}, {"text": "llo"}]}}})
    runtime = BedrockRuntime("us-east-1", client=boto)

    answer = await runtime.converse(model_id="m", system="sys", text="look", images=[b"\xff\xd8"])

    assert answer == "hello"
    assert boto.kwargs["modelId"] == "m"
    assert boto.kwargs["system"] == [{"text": "sys"}]
    content = boto.kwargs["messages"][0]["content"]
    assert content[0] == {"image": {"format": "jpeg", "source": {"bytes": b"\xff\xd8"}}}
    assert content[-1] == {"text": "look"}
    assert boto.kwargs["inferenceConfig"]["temperature"] == 0


@pytest.mark.anyio
async def test_an_aws_error_becomes_a_bedrock_error():
    error = ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow"}}, "Converse")
    runtime = BedrockRuntime("us-east-1", client=_FakeBoto(error=error))

    with pytest.raises(BedrockError):
        await runtime.converse(model_id="m", system="s", text="t", images=[])


@pytest.mark.anyio
async def test_an_answer_without_text_is_an_error():
    runtime = BedrockRuntime(
        "us-east-1", client=_FakeBoto({"output": {"message": {"content": []}}})
    )

    with pytest.raises(BedrockError):
        await runtime.converse(model_id="m", system="s", text="t", images=[])


def test_building_the_runtime_needs_no_credentials_or_network():
    BedrockRuntime("us-east-1")  # the boto3 client is only created on the first call
