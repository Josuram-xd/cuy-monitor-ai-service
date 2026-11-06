"""Feeder and waterer zones of each cage, from config (Task 9.1).

Each zone is a polygon in fractions of the frame (x / width, y / height, from 0 to 1), so the same
zones work at any resolution the edge agent sends. They come from the CAGE_ZONES setting:

    CAGE_ZONES='{"cage-1": {"feeder": [[0.05, 0.6], [0.3, 0.6], [0.3, 0.95], [0.05, 0.95]],
                            "waterer": [[0.8, 0.05], [0.95, 0.05], [0.95, 0.3], [0.8, 0.3]]}}'

A cage without zones still gets its BEHAVIOR events, with 0 feeder and waterer visits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True)
class Zone:
    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("A zone needs at least 3 points")
        for x, y in self.points:
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f"Zone point ({x}, {y}) is outside the frame (0-1)")

    def contains(self, x: float, y: float) -> bool:
        """Ray casting: is the point (fractions of the frame) inside the polygon?"""
        inside = False
        points = self.points
        j = len(points) - 1
        for i, (xi, yi) in enumerate(points):
            xj, yj = points[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside


@dataclass(frozen=True)
class CageZones:
    feeder: Zone | None = None
    waterer: Zone | None = None

    def locate(self, x: float, y: float) -> tuple[bool, bool]:
        """(at the feeder, at the waterer) for a point in fractions of the frame."""
        return (
            self.feeder is not None and self.feeder.contains(x, y),
            self.waterer is not None and self.waterer.contains(x, y),
        )


NO_ZONES = CageZones()


def _zone(raw: object, cage_id: str, name: str) -> Zone | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"CAGE_ZONES: {cage_id}.{name} must be a list of [x, y] points")
    points = []
    for point in raw:
        if not (isinstance(point, list | tuple) and len(point) == 2):
            raise ValueError(f"CAGE_ZONES: {cage_id}.{name} has a point that is not [x, y]")
        points.append((float(point[0]), float(point[1])))
    try:
        return Zone(tuple(points))
    except ValueError as exception:
        raise ValueError(f"CAGE_ZONES: {cage_id}.{name}: {exception}") from None


def parse_cage_zones(text: str) -> dict[str, CageZones]:
    """The CAGE_ZONES JSON -> zones per cage. Empty text = no zones. A bad value is an error."""
    if not text.strip():
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exception:
        raise ValueError(f"CAGE_ZONES is not valid JSON: {exception.msg}") from None
    if not isinstance(raw, dict):
        raise ValueError("CAGE_ZONES must be an object with one entry per cage")
    zones = {}
    for cage_id, cage in raw.items():
        if not isinstance(cage, dict):
            raise ValueError(f"CAGE_ZONES: {cage_id} must be an object with feeder and waterer")
        unknown = set(cage) - {"feeder", "waterer"}
        if unknown:
            raise ValueError(f"CAGE_ZONES: {cage_id} has unknown zones {sorted(unknown)}")
        zones[cage_id] = CageZones(
            feeder=_zone(cage.get("feeder"), cage_id, "feeder"),
            waterer=_zone(cage.get("waterer"), cage_id, "waterer"),
        )
    return zones
