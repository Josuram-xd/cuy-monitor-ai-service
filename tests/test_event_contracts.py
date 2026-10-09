from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.contracts import (
    AudioEventPayload,
    BehaviorEventPayload,
    EventEnvelope,
    EventType,
    MarkColor,
    WeightEventPayload,
)


def test_event_envelope_validates_payload_and_serializes_camel_case_aliases():
    event = EventEnvelope(
        eventId="3f1c2a5e-8f0b-4a53-9d0e-6b1c0a7e9a11",
        type=EventType.BEHAVIOR,
        cageId="cage-1",
        timestamp="2026-10-05T14:32:00Z",
        source="ai-service",
        schemaVersion=1,
        payload={
            "color": MarkColor.RED,
            "windowSeconds": 60,
            "stillSeconds": 48,
            "feederVisits": 0,
            "watererVisits": 1,
            "avgGroupDistance": 0.72,
            "probAnomaly": 0.81,
            "detectionConfidence": 0.93,
        },
    )

    serialized = event.model_dump(mode="json", by_alias=True)

    assert serialized["eventId"] == "3f1c2a5e-8f0b-4a53-9d0e-6b1c0a7e9a11"
    assert serialized["schemaVersion"] == 1
    assert serialized["payload"]["windowSeconds"] == 60
    assert isinstance(event.payload, BehaviorEventPayload)
    assert event.timestamp == datetime(2026, 10, 5, 14, 32, tzinfo=UTC)


@pytest.mark.parametrize(
    ("event_type", "payload", "payload_type"),
    [
        (
            EventType.BEHAVIOR,
            {
                "color": "BLUE",
                "windowSeconds": 60,
                "stillSeconds": 12,
                "feederVisits": 1,
                "watererVisits": 0,
                "avgGroupDistance": 0.4,
                "probAnomaly": 0.2,
                "detectionConfidence": 0.9,
            },
            BehaviorEventPayload,
        ),
        (
            EventType.AUDIO,
            {"label": "NORMAL", "probability": 0.8, "durationMs": 960},
            AudioEventPayload,
        ),
        (EventType.WEIGHT, {"grams": 812.4, "stable": True}, WeightEventPayload),
    ],
)
def test_envelope_selects_payload_model(event_type, payload, payload_type):
    event = EventEnvelope.model_validate(
        {
            "eventId": "3f1c2a5e-8f0b-4a53-9d0e-6b1c0a7e9a11",
            "type": event_type,
            "cageId": "cage-1",
            "timestamp": "2026-10-05T14:32:00Z",
            "source": "ai-service",
            "schemaVersion": 1,
            "payload": payload,
            "extraEnvelopeField": "ignored",
        }
    )

    assert isinstance(event.payload, payload_type)


def test_envelope_ignores_unknown_payload_fields():
    payload = WeightEventPayload.model_validate(
        {"grams": 812.4, "stable": True, "producerMetadata": "ignored"}
    )

    assert payload.model_dump() == {"grams": 812.4, "stable": True}


@pytest.mark.parametrize(
    "event_data",
    [
        {
            "eventId": "3f1c2a5e-8f0b-4a53-9d0e-6b1c0a7e9a11",
            "type": "AUDIO",
            "cageId": "cage-1",
            "timestamp": "2026-10-05T14:32:00Z",
            "source": "ai-service",
            "schemaVersion": 1,
            "payload": {"grams": 812.4, "stable": True},
        },
        {
            "eventId": "3f1c2a5e-8f0b-4a53-9d0e-6b1c0a7e9a11",
            "type": "WEIGHT",
            "cageId": "cage-1",
            "timestamp": "2026-10-05T14:32:00+00:00",
            "source": "ai-service",
            "schemaVersion": 1,
            "payload": {"grams": 812.4, "stable": True},
        },
    ],
)
def test_envelope_rejects_mismatched_payload_or_non_z_timestamp(event_data):
    with pytest.raises(ValidationError):
        EventEnvelope.model_validate(event_data)


def test_payload_validation_rejects_out_of_range_values():
    with pytest.raises(ValidationError):
        BehaviorEventPayload(
            color=MarkColor.RED,
            windowSeconds=60,
            stillSeconds=61,
            feederVisits=0,
            watererVisits=0,
            avgGroupDistance=0.2,
            probAnomaly=0.1,
            detectionConfidence=0.9,
        )
