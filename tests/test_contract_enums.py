from app.contracts.enums import EventType, MarkColor


def test_event_type_values_match_backend_contract():
    assert [event_type.value for event_type in EventType] == ["BEHAVIOR", "AUDIO", "WEIGHT"]


def test_mark_color_values_match_backend_contract():
    assert [color.value for color in MarkColor] == [
        "RED",
        "BLUE",
        "GREEN",
        "YELLOW",
        "ORANGE",
        "PURPLE",
        "BLACK",
        "WHITE",
    ]
