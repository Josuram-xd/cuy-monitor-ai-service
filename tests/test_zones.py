import json

import pytest

from app.config import Settings
from app.vision.zones import NO_ZONES, Zone, parse_cage_zones

SQUARE = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]


def test_a_point_inside_or_outside_a_polygon():
    zone = Zone(tuple(map(tuple, SQUARE)))

    assert zone.contains(0.2, 0.2)
    assert not zone.contains(0.5, 0.2)
    assert not zone.contains(0.2, 0.9)


def test_a_concave_polygon_is_handled():
    # an L shape: the inner corner is outside
    zone = Zone(((0, 0), (0.6, 0), (0.6, 0.2), (0.2, 0.2), (0.2, 0.6), (0, 0.6)))

    assert zone.contains(0.1, 0.5)
    assert zone.contains(0.5, 0.1)
    assert not zone.contains(0.4, 0.4)


def test_cage_zones_come_from_json():
    zones = parse_cage_zones(json.dumps({"cage-1": {"feeder": SQUARE}}))

    assert zones["cage-1"].locate(0.2, 0.2) == (True, False)
    assert zones["cage-1"].locate(0.9, 0.9) == (False, False)


def test_no_zones_means_no_visits():
    assert parse_cage_zones("") == {}
    assert NO_ZONES.locate(0.5, 0.5) == (False, False)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"cage-1": []}',
        '{"cage-1": {"bed": [[0,0],[1,0],[1,1]]}}',
        '{"cage-1": {"feeder": [[0,0],[1,0]]}}',
        '{"cage-1": {"feeder": [[0,0],[1,0],[1,1.5]]}}',
        '{"cage-1": {"feeder": [[0,0],[1],[1,1]]}}',
    ],
)
def test_bad_zones_are_refused(text):
    with pytest.raises(ValueError):
        parse_cage_zones(text)


def test_settings_check_the_zones_at_start_and_give_them_per_cage():
    with pytest.raises(ValueError):
        Settings(_env_file=None, api_key="k", cage_zones="{oops")

    settings = Settings(
        _env_file=None, api_key="k", cage_zones=json.dumps({"cage-1": {"waterer": SQUARE}})
    )
    assert settings.zones_for("cage-1").locate(0.2, 0.2) == (False, True)
    assert settings.zones_for("cage-2") is NO_ZONES
