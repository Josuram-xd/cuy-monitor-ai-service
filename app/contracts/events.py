from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

from app.contracts.enums import EventType, MarkColor


class ContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )


class AudioLabel(StrEnum):
    DISTRESS = "DISTRESS"
    NORMAL = "NORMAL"


class BehaviorEventPayload(ContractModel):
    color: MarkColor
    window_seconds: int = Field(gt=0)
    still_seconds: float = Field(ge=0)
    feeder_visits: int = Field(ge=0)
    waterer_visits: int = Field(ge=0)
    avg_group_distance: float = Field(ge=0, le=1)
    prob_anomaly: float = Field(ge=0, le=1)
    detection_confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def still_time_fits_window(self) -> Self:
        if self.still_seconds > self.window_seconds:
            raise ValueError("stillSeconds cannot exceed windowSeconds")
        return self


class AudioEventPayload(ContractModel):
    label: AudioLabel
    probability: float = Field(ge=0, le=1)
    duration_ms: int = Field(gt=0)


class WeightEventPayload(ContractModel):
    grams: float = Field(ge=0)
    stable: bool


EventPayload = BehaviorEventPayload | AudioEventPayload | WeightEventPayload


class EventEnvelope(ContractModel):
    event_id: UUID
    type: EventType
    cage_id: str = Field(min_length=1, max_length=50)
    timestamp: datetime
    source: str = Field(min_length=1, max_length=30)
    schema_version: Literal[1] = Field(alias="schemaVersion")
    payload: EventPayload

    @field_validator("timestamp", mode="before")
    @classmethod
    def require_utc_timestamp(cls, value: object) -> object:
        if isinstance(value, str) and not value.endswith("Z"):
            raise ValueError("timestamp must be an ISO-8601 UTC value ending in Z")
        if isinstance(value, datetime) and (
            value.tzinfo is None or value.utcoffset() != timedelta(0)
        ):
            raise ValueError("timestamp must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def event_type_matches_payload(self) -> Self:
        expected_payload = {
            EventType.BEHAVIOR: BehaviorEventPayload,
            EventType.AUDIO: AudioEventPayload,
            EventType.WEIGHT: WeightEventPayload,
        }[self.type]
        if not isinstance(self.payload, expected_payload):
            raise ValueError(f"{self.type.value} events require {expected_payload.__name__}")
        return self
