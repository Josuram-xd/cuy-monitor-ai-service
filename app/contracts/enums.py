from enum import StrEnum


class EventType(StrEnum):
    BEHAVIOR = "BEHAVIOR"
    AUDIO = "AUDIO"
    WEIGHT = "WEIGHT"


class MarkColor(StrEnum):
    RED = "RED"
    BLUE = "BLUE"
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    PURPLE = "PURPLE"
    BLACK = "BLACK"
    WHITE = "WHITE"
