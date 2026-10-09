from pathlib import Path

import pytest

from app.contracts import (
    AudioEventPayload,
    BehaviorEventPayload,
    EventEnvelope,
    EventType,
    WeightEventPayload,
)

EXAMPLES = Path(__file__).parent / "fixtures" / "backend-events"
PAYLOAD_TYPES = {
    EventType.BEHAVIOR: BehaviorEventPayload,
    EventType.AUDIO: AudioEventPayload,
    EventType.WEIGHT: WeightEventPayload,
}


@pytest.mark.parametrize("example_name", ["behavior.json", "audio.json", "weight.json"])
def test_backend_event_examples_match_contract_models(example_name):
    example_path = EXAMPLES / example_name
    event = EventEnvelope.model_validate_json(example_path.read_text(encoding="utf-8"))

    assert isinstance(event.payload, PAYLOAD_TYPES[event.type])
