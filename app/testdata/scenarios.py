import random
from dataclasses import dataclass, field
from typing import Protocol

from app.contracts import AudioEventPayload, BehaviorEventPayload, MarkColor
from app.contracts.events import AudioLabel

DEFAULT_COLORS = (MarkColor.RED, MarkColor.BLUE, MarkColor.GREEN, MarkColor.YELLOW)


class Scenario(Protocol):
    """What the cage does window after window. Window 0 is the first one sent."""

    name: str
    description: str

    def behavior(self, window: int) -> list[BehaviorEventPayload]: ...

    def audio(self, window: int) -> AudioEventPayload | None: ...


def _rng(seed: int, window: int, color: MarkColor) -> random.Random:
    # one generator per window and colour: replaying a scenario always sends the same numbers
    return random.Random(f"{seed}:{window}:{color}")


def healthy_window(
    color: MarkColor, window: int, window_seconds: int, seed: int
) -> BehaviorEventPayload:
    """Inside the backend's default profile: still less than half the window, eats, stays close."""
    rng = _rng(seed, window, color)
    return BehaviorEventPayload(
        color=color,
        window_seconds=window_seconds,
        still_seconds=round(window_seconds * rng.uniform(0.15, 0.45), 1),
        feeder_visits=rng.randint(1, 2),
        waterer_visits=rng.randint(0, 1),
        avg_group_distance=round(rng.uniform(0.2, 0.45), 2),
        prob_anomaly=round(rng.uniform(0.03, 0.25), 2),
        detection_confidence=round(rng.uniform(0.85, 0.97), 2),
    )


def lethargic_window(
    color: MarkColor, window: int, window_seconds: int, seed: int
) -> BehaviorEventPayload:
    """What the backend must flag: almost still the whole window, no food, apart from the group."""
    rng = _rng(seed, window, color)
    return BehaviorEventPayload(
        color=color,
        window_seconds=window_seconds,
        still_seconds=round(window_seconds * rng.uniform(0.88, 0.98), 1),
        feeder_visits=0,
        waterer_visits=0,
        avg_group_distance=round(rng.uniform(0.75, 0.9), 2),
        prob_anomaly=round(rng.uniform(0.82, 0.93), 2),
        detection_confidence=round(rng.uniform(0.88, 0.95), 2),
    )


def _calm_audio() -> AudioEventPayload:
    return AudioEventPayload(label=AudioLabel.NORMAL, probability=0.94, duration_ms=960)


@dataclass(frozen=True)
class Healthy:
    colors: tuple[MarkColor, ...] = DEFAULT_COLORS
    window_seconds: int = 60
    seed: int = 1
    name: str = "healthy"
    description: str = "Every guinea pig behaves normally; nothing should change in the backend."

    def behavior(self, window: int) -> list[BehaviorEventPayload]:
        return [healthy_window(c, window, self.window_seconds, self.seed) for c in self.colors]

    def audio(self, window: int) -> AudioEventPayload | None:
        return _calm_audio()


@dataclass(frozen=True)
class Lethargic:
    colors: tuple[MarkColor, ...] = DEFAULT_COLORS
    sick: MarkColor = MarkColor.RED
    starts_at: int = 2
    window_seconds: int = 60
    seed: int = 1
    name: str = "lethargic"
    description: str = (
        "One guinea pig stops moving and eating from window 2 on. The backend should take it to "
        "OBSERVED after N windows in a row, and up to ALERT and CRITICAL if it lasts."
    )

    def behavior(self, window: int) -> list[BehaviorEventPayload]:
        return [
            lethargic_window(c, window, self.window_seconds, self.seed)
            if c == self.sick and window >= self.starts_at
            else healthy_window(c, window, self.window_seconds, self.seed)
            for c in self.colors
        ]

    def audio(self, window: int) -> AudioEventPayload | None:
        return _calm_audio()


@dataclass(frozen=True)
class Distress:
    colors: tuple[MarkColor, ...] = DEFAULT_COLORS
    window_seconds: int = 60
    seed: int = 1
    name: str = "distress"
    description: str = "Normal behavior, but distress calls every window: a cage-level AUDIO alert."

    def behavior(self, window: int) -> list[BehaviorEventPayload]:
        return Healthy(self.colors, self.window_seconds, self.seed).behavior(window)

    def audio(self, window: int) -> AudioEventPayload | None:
        return AudioEventPayload(label=AudioLabel.DISTRESS, probability=0.91, duration_ms=960)


@dataclass(frozen=True)
class Recovery:
    colors: tuple[MarkColor, ...] = DEFAULT_COLORS
    sick: MarkColor = MarkColor.RED
    sick_windows: int = 8
    window_seconds: int = 60
    seed: int = 1
    name: str = "recovery"
    description: str = (
        "The lethargic guinea pig gets better after 8 windows. After enough normal windows in a "
        "row the backend should bring it back to NORMAL without a new alert."
    )
    _lethargic: Lethargic = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_lethargic", Lethargic(self.colors, self.sick, 0, self.window_seconds, self.seed)
        )

    def behavior(self, window: int) -> list[BehaviorEventPayload]:
        if window < self.sick_windows:
            return self._lethargic.behavior(window)
        return Healthy(self.colors, self.window_seconds, self.seed).behavior(window)

    def audio(self, window: int) -> AudioEventPayload | None:
        return _calm_audio()


SCENARIOS = {
    scenario.name: type(scenario) for scenario in (Healthy(), Lethargic(), Distress(), Recovery())
}


def build_scenario(
    name: str, colors: tuple[MarkColor, ...] = DEFAULT_COLORS, window_seconds: int = 60
) -> Scenario:
    """The scenario called name, for the guinea pigs of these colours. Unknown name: ValueError."""
    try:
        scenario_type = SCENARIOS[name]
    except KeyError:
        raise ValueError(f"Unknown scenario {name!r}; one of: {', '.join(SCENARIOS)}") from None
    scenario = scenario_type(colors=colors, window_seconds=window_seconds)
    sick = getattr(scenario, "sick", None)
    if sick is not None and sick not in colors:
        raise ValueError(f"The scenario needs a {sick} guinea pig among the colours")
    return scenario


def parse_colors(text: str) -> tuple[MarkColor, ...]:
    """'RED, blue' -> (RED, BLUE). Empty: the default four."""
    names = [part.strip().upper() for part in text.split(",") if part.strip()]
    return tuple(MarkColor(name) for name in names) or DEFAULT_COLORS
