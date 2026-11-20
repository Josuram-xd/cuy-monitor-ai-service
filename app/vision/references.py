"""Where the coats to compare against come from (Task A6).

- BackendReferenceSource: the registered photos of a cage's guinea pigs, from the backend's
  references endpoint (ETag, refreshed after REFERENCE_TTL_SECONDS). Every photo is validated and
  turned into a coat descriptor in a thread. If the backend is down, the last good list stays.
- The gallery: coats learnt from the cage's own confident marks (self-enrolment, in memory only:
  a restart rebuilds it in minutes when the marks are visible).
- ClipReferenceSource: the crops of a simulation clip (sim- cages), which have no backend photos.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import cv2
import httpx
import numpy as np

from app.config import Settings
from app.contracts import MarkColor
from app.vision.animal_detector import OnnxAnimalDetector
from app.vision.appearance import describe_reference
from app.vision.identity import Reference

logger = logging.getLogger(__name__)

REFERENCE_TTL_SECONDS = 300
GALLERY_SIZE = 8
ENROL_EVERY_SECONDS = 10
# a registered photo bigger than this is refused: the backend sends them over the internal network
MAX_PHOTO_BYTES = 400 * 1024
REQUEST_TIMEOUT_SECONDS = 10


class ReferenceSource(Protocol):
    async def references(self, cage_id: str) -> list[Reference]: ...

    def enrol(self, cage_id: str, color: MarkColor, descriptor: np.ndarray) -> None: ...


@dataclass(frozen=True)
class _Fetched:
    etag: str | None
    expires_at: float
    references: list[Reference]


class _Gallery:
    """Coats of a guinea pig seen with a confident mark: GALLERY_SIZE at most, one per 10 s."""

    def __init__(self, clock: Callable[[], float]):
        self._clock = clock
        self._coats: dict[tuple[str, MarkColor], deque[np.ndarray]] = {}
        self._last: dict[tuple[str, MarkColor], float] = {}

    def add(self, cage_id: str, color: MarkColor, descriptor: np.ndarray) -> None:
        key = (cage_id, color)
        now = self._clock()
        if key in self._last and now - self._last[key] < ENROL_EVERY_SECONDS:
            return
        self._last[key] = now
        self._coats.setdefault(key, deque(maxlen=GALLERY_SIZE)).append(descriptor)

    def coats(self, cage_id: str, color: MarkColor) -> list[np.ndarray]:
        return list(self._coats.get((cage_id, color), []))


class BackendReferenceSource:
    def __init__(
        self,
        settings: Settings,
        animals: OnnxAnimalDetector | None,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._base_url = settings.backend_url.rstrip("/")
        self._headers = {"X-API-Key": settings.api_key.get_secret_value()}
        self._animals = animals
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
        self._clock = clock
        self._fetched: dict[str, _Fetched] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._gallery = _Gallery(clock)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def references(self, cage_id: str) -> list[Reference]:
        # one request per cage at a time: frames that arrive together wait for the same answer
        lock = self._locks.setdefault(cage_id, asyncio.Lock())
        async with lock:
            fetched = self._fetched.get(cage_id)
            if fetched is None or self._clock() >= fetched.expires_at:
                fetched = await self._fetch(cage_id, fetched)
                self._fetched[cage_id] = fetched
        return [
            Reference(
                r.color,
                r.coat_color,
                [*r.descriptors, *self._gallery.coats(cage_id, r.color)],
            )
            for r in fetched.references
        ]

    def enrol(self, cage_id: str, color: MarkColor, descriptor: np.ndarray) -> None:
        self._gallery.add(cage_id, color, descriptor)

    async def _fetch(self, cage_id: str, previous: _Fetched | None) -> _Fetched:
        now = self._clock()
        headers = dict(self._headers)
        if previous is not None and previous.etag:
            headers["If-None-Match"] = previous.etag
        url = f"{self._base_url}/api/v1/ingestion/cages/{quote(cage_id, safe='')}/guinea-pigs"
        try:
            response = await self._client.get(url, headers=headers)
        except httpx.HTTPError as exception:
            logger.warning("References of %s not refreshed: %s", cage_id, exception)
            return self._keep(previous, now)
        if response.status_code == 304 and previous is not None:
            return _Fetched(previous.etag, now + REFERENCE_TTL_SECONDS, previous.references)
        if response.status_code == 404:
            return _Fetched(None, now + REFERENCE_TTL_SECONDS, [])
        if response.status_code != 200:
            logger.warning(
                "References of %s: the backend answered %s", cage_id, response.status_code
            )
            return self._keep(previous, now)
        try:
            items = response.json()
        except ValueError:
            logger.warning("References of %s: the answer is not JSON", cage_id)
            return self._keep(previous, now)
        if not isinstance(items, list):
            logger.warning("References of %s: the answer is not a list", cage_id)
            return self._keep(previous, now)
        references = await asyncio.to_thread(self._parse, items)
        return _Fetched(response.headers.get("ETag"), now + REFERENCE_TTL_SECONDS, references)

    def _keep(self, previous: _Fetched | None, now: float) -> _Fetched:
        # the last good answer stays until the TTL ends; with none, the cage has no references yet
        if previous is None:
            return _Fetched(None, now + REFERENCE_TTL_SECONDS, [])
        return _Fetched(previous.etag, now + REFERENCE_TTL_SECONDS, previous.references)

    def _parse(self, items: list[Any]) -> list[Reference]:
        """The backend list -> references. Runs in a thread: it decodes photos and runs a model."""
        references = []
        for item in items:
            try:
                color = MarkColor(item["markColor"])
            except KeyError, TypeError, ValueError:
                logger.warning("A guinea pig with an unknown mark colour was skipped")
                continue
            coat = item.get("coatColor")
            descriptors = []
            photo = _decode_photo(item.get("photo"))
            if photo is not None:
                descriptor = describe_reference(photo, self._animals)
                if descriptor is not None:
                    descriptors.append(descriptor)
            references.append(
                Reference(color, coat if isinstance(coat, str) else None, descriptors)
            )
        return references


def _decode_photo(encoded: object) -> np.ndarray | None:
    """A base64 JPEG/PNG from the backend -> image, or None (logged) when it is not one."""
    if not isinstance(encoded, str):
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error, ValueError:
        logger.warning("A registered photo is not valid base64: skipped")
        return None
    if not raw or len(raw) > MAX_PHOTO_BYTES:
        logger.warning("A registered photo is empty or too big: skipped")
        return None
    try:
        return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None


class ClipReferenceSource:
    """The crops of a simulation clip: one per colour, read and described once."""

    def __init__(self, references: dict[MarkColor, Path], animals: OnnxAnimalDetector | None):
        self._paths = references
        self._animals = animals
        self._loaded: list[Reference] | None = None

    async def references(self, cage_id: str) -> list[Reference]:
        if self._loaded is None:
            self._loaded = await asyncio.to_thread(self._load)
        return self._loaded

    def enrol(self, cage_id: str, color: MarkColor, descriptor: np.ndarray) -> None:
        # the clip is the same every time: nothing to learn from its own marks
        return None

    def _load(self) -> list[Reference]:
        references = []
        for color, path in self._paths.items():
            image = cv2.imread(str(path))
            descriptor = describe_reference(image, self._animals) if image is not None else None
            references.append(Reference(color, None, [] if descriptor is None else [descriptor]))
        return references
