import base64

import cv2
import httpx
import numpy as np
import pytest

from app.config import Settings
from app.contracts import MarkColor
from app.vision.references import (
    ENROL_EVERY_SECONDS,
    GALLERY_SIZE,
    REFERENCE_TTL_SECONDS,
    BackendReferenceSource,
    ClipReferenceSource,
)


def photo(side: int, bgr=(40, 80, 150)) -> str:
    image = np.full((side, side, 3), bgr, dtype=np.uint8)
    ok, data = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(data.tobytes()).decode("ascii")


GUINEA_PIGS = [
    {
        "id": 1,
        "markColor": "RED",
        "coatColor": "BROWN",
        "breed": "PERUVIAN",
        "photoContentType": "image/jpeg",
        "photo": photo(120),
    },
    {"id": 2, "markColor": "BLUE", "coatColor": "WHITE", "photo": photo(32)},  # placeholder
    {"id": 3, "markColor": "GREEN", "coatColor": None, "photo": None},
    {"id": 4, "markColor": "YELLOW", "coatColor": "BLACK", "photo": "not base64!"},
    {"id": 5, "markColor": "PINK", "coatColor": "BLACK", "photo": photo(120)},  # not a colour
]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def source_for(handler, clock=None):
    settings = Settings(_env_file=None, api_key="secret", backend_url="http://backend:8080/")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return BackendReferenceSource(settings, None, client, clock or Clock())


@pytest.mark.anyio
async def test_the_registered_photos_become_coat_descriptors_and_bad_ones_are_skipped():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=GUINEA_PIGS, headers={"ETag": '"v1"'})

    source = source_for(handler)

    references = {r.color: r for r in await source.references("cage-1")}

    request = seen[0]
    assert request.url.path == "/api/v1/ingestion/cages/cage-1/guinea-pigs"
    assert request.headers["X-API-Key"] == "secret"
    assert len(references[MarkColor.RED].descriptors) == 1
    assert references[MarkColor.RED].coat_color == "BROWN"
    assert references[MarkColor.BLUE].descriptors == []  # 32 px placeholder: not a photo
    assert references[MarkColor.GREEN].descriptors == []
    assert references[MarkColor.GREEN].coat_color is None
    assert references[MarkColor.YELLOW].descriptors == []  # broken base64
    assert len(references) == 4  # the item with an unknown colour is skipped


@pytest.mark.anyio
async def test_a_304_keeps_the_list_and_renews_its_ttl():
    clock = Clock()
    calls = []

    def handler(request):
        calls.append(request.headers.get("If-None-Match"))
        if len(calls) == 1:
            return httpx.Response(200, json=GUINEA_PIGS[:1], headers={"ETag": '"v1"'})
        return httpx.Response(304)

    source = source_for(handler, clock)
    first = await source.references("cage-1")
    clock.now = REFERENCE_TTL_SECONDS + 1
    second = await source.references("cage-1")
    await source.references("cage-1")  # still fresh: no request

    assert calls == [None, '"v1"']
    assert [r.color for r in second] == [r.color for r in first] == [MarkColor.RED]


@pytest.mark.anyio
async def test_a_failed_refresh_keeps_the_last_good_list():
    clock = Clock()
    answers = [httpx.Response(200, json=GUINEA_PIGS[:1]), httpx.Response(500)]

    def handler(request):
        return answers.pop(0)

    source = source_for(handler, clock)
    good = await source.references("cage-1")
    clock.now = REFERENCE_TTL_SECONDS + 1
    kept = await source.references("cage-1")

    assert [r.color for r in kept] == [r.color for r in good] == [MarkColor.RED]


@pytest.mark.anyio
async def test_a_network_error_with_nothing_before_gives_an_empty_list():
    def handler(request):
        raise httpx.ConnectError("no route to backend")

    assert await source_for(handler).references("cage-1") == []


@pytest.mark.anyio
async def test_an_unknown_cage_has_no_references():
    source = source_for(lambda request: httpx.Response(404))

    assert await source.references("cage-9") == []


@pytest.mark.anyio
async def test_the_gallery_is_bounded_and_takes_one_coat_per_interval():
    clock = Clock()
    source = source_for(lambda request: httpx.Response(200, json=GUINEA_PIGS[:1]), clock)
    descriptor = np.full(4, 0.25, dtype=np.float32)

    source.enrol("cage-1", MarkColor.RED, descriptor)
    clock.now = 3.0
    source.enrol("cage-1", MarkColor.RED, descriptor)  # too soon: refused
    for step in range(1, GALLERY_SIZE + 3):
        clock.now = step * ENROL_EVERY_SECONDS + 10
        source.enrol("cage-1", MarkColor.RED, descriptor)

    [red] = await source.references("cage-1")
    assert len(red.descriptors) == 1 + GALLERY_SIZE  # the photo, then the 8 newest coats


@pytest.mark.anyio
async def test_the_clip_crops_are_described_once_and_never_learn(tmp_path):
    folder = tmp_path
    crop = folder / "RED.jpg"
    cv2.imwrite(str(crop), np.full((120, 120, 3), (40, 80, 150), np.uint8))
    missing = folder / "BLUE.jpg"
    clip = ClipReferenceSource({MarkColor.RED: crop, MarkColor.BLUE: missing}, None)

    [red, blue] = await clip.references("sim-1")

    assert red.color is MarkColor.RED and len(red.descriptors) == 1
    assert blue.descriptors == []
    clip.enrol("sim-1", MarkColor.RED, np.zeros(4, np.float32))
    assert len((await clip.references("sim-1"))[0].descriptors) == 1
