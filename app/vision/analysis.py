import json
import logging
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel

from app.bedrock.runtime import ModelRuntime
from app.contracts import BehaviorEventPayload, MarkColor

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You analyse frames of a pet guinea pig cage for a health monitor.
Each guinea pig wears a coloured mark on its back. The possible colours are:
RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE, BLACK, WHITE.

You get several frames taken evenly over one observation window. For every guinea pig whose mark
colour you can identify with confidence, report:
- framesSeen: in how many of the frames you could see it.
- framesStill: in how many of those frames it stayed in the same place and posture as in the
  previous frame it appears in (it is not moving).
- feederVisits / waterVisits: in how many frames it is at the feeder / at the water bottle.
- avgGroupDistance: how far it is from the other guinea pigs, from 0 (touching) to 1 (the far
  side of the cage). Use 0.5 if it is alone in view.
- anomaly: 0 to 1, how unusual it looks (lethargic, hunched, isolated, trembling, not eating).
- confidence: 0 to 1, how sure you are about its colour and your observations.

Never guess a colour: leave the guinea pig out if the mark is unclear. Never invent animals.
Answer with ONLY a JSON object, no text around it:
{"guineaPigs": [{"color": "RED", "framesSeen": 5, "framesStill": 3, "feederVisits": 1,
"waterVisits": 0, "avgGroupDistance": 0.4, "anomaly": 0.1, "confidence": 0.9}]}
If you see no marked guinea pig answer {"guineaPigs": []}."""


class Observation(BaseModel):
    """One guinea pig as the model describes it. Anything out of range is rejected, not trusted."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")

    color: MarkColor
    frames_seen: int = Field(ge=1)
    frames_still: int = Field(ge=0)
    feeder_visits: int = Field(default=0, ge=0)
    water_visits: int = Field(default=0, ge=0)
    avg_group_distance: float = Field(default=0.5, ge=0, le=1)
    anomaly: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_observations(answer: str, frame_count: int) -> list[Observation]:
    """The model's text -> valid observations. Bad items are dropped, a bad answer gives []."""
    match = _JSON_OBJECT.search(answer)
    if match is None:
        logger.warning("The model answer has no JSON object")
        return []
    try:
        items = json.loads(match.group(0)).get("guineaPigs", [])
    except json.JSONDecodeError, AttributeError:
        logger.warning("The model answer is not valid JSON")
        return []
    best: dict[MarkColor, Observation] = {}
    for item in items if isinstance(items, list) else []:
        try:
            observation = Observation.model_validate(item)
        except ValidationError as exception:
            logger.info("Dropping an invalid observation: %s", exception.errors()[0]["msg"])
            continue
        if (
            observation.frames_seen > frame_count
            or observation.frames_still > observation.frames_seen
        ):
            logger.info("Dropping an impossible observation for %s", observation.color)
            continue
        # one event per colour: if the model repeats it keep the one backed by more frames
        known = best.get(observation.color)
        if known is None or observation.frames_seen > known.frames_seen:
            best[observation.color] = observation
    return list(best.values())


def to_payload(
    observation: Observation, frame_count: int, window_seconds: int
) -> BehaviorEventPayload:
    still_ratio = observation.frames_still / observation.frames_seen
    seen_ratio = observation.frames_seen / frame_count
    return BehaviorEventPayload(
        color=observation.color,
        window_seconds=window_seconds,
        still_seconds=round(window_seconds * still_ratio, 1),
        feeder_visits=observation.feeder_visits,
        waterer_visits=observation.water_visits,
        avg_group_distance=round(observation.avg_group_distance, 2),
        prob_anomaly=round(observation.anomaly, 2),
        # seen in few frames or unsure -> low confidence, so the backend can ignore the window
        detection_confidence=round(min(1.0, seen_ratio * observation.confidence), 2),
    )


class BedrockBehaviorAnalyzer:
    def __init__(self, runtime: ModelRuntime, model_id: str, window_seconds: int):
        self._runtime = runtime
        self._model_id = model_id
        self._window_seconds = window_seconds

    async def analyze(self, frames: list[bytes]) -> list[BehaviorEventPayload]:
        text = (
            f"{len(frames)} frames, evenly spaced over {self._window_seconds} seconds, "
            "in time order. Report every marked guinea pig."
        )
        answer = await self._runtime.converse(
            model_id=self._model_id, system=SYSTEM_PROMPT, text=text, images=frames
        )
        return [
            to_payload(observation, len(frames), self._window_seconds)
            for observation in parse_observations(answer, len(frames))
        ]
