"""The trained behavior chain (Task 10.3): detector -> tracker -> 60 s window -> Random Forest.

Each frame is analysed when it arrives (the detector runs in a thread, never on the event loop),
and only what was seen in it is kept. When the window closes, the features of every guinea pig
are computed and the classifier gives their probAnomaly. The result is the same BEHAVIOR payload
the Bedrock engine gives, so the backend cannot tell them apart.
"""

from __future__ import annotations

import asyncio
import csv
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from app.behavior.classifier import BehaviorClassifier
from app.behavior.features import FEATURE_NAMES, BehaviorFeatures, compute_features
from app.behavior.window import BehaviorWindows, sample_frame
from app.contracts import BehaviorEventPayload
from app.vision.detector import Detection, UnreadableFrame, decode_jpeg
from app.vision.tracker import HungarianTracker
from app.vision.zones import CageZones

logger = logging.getLogger(__name__)

# frames of one cage analysed at the same time; more are dropped, never queued (the edge agent
# sends 1-2 fps, so this only happens if the CPU falls behind)
MAX_FRAMES_IN_FLIGHT = 2


class ImageDetector(Protocol):
    def detect_image(self, image_bgr: np.ndarray) -> list[Detection]: ...


def _detect(detector: ImageDetector, frame: bytes) -> tuple[list[Detection], int, int]:
    image = decode_jpeg(frame)
    height, width = image.shape[:2]
    return detector.detect_image(image), width, height


class TrainedBehaviorEngine:
    def __init__(
        self,
        detector: ImageDetector,
        classifier: BehaviorClassifier,
        zones_for: Callable[[str], CageZones],
        window_seconds: int,
        features_log: Path | None = None,
        tracker_factory: Callable[[], HungarianTracker] = HungarianTracker,
    ):
        self._detector = detector
        self._classifier = classifier
        self._zones_for = zones_for
        self._window_seconds = window_seconds
        self._features_log = features_log
        self._tracker_factory = tracker_factory
        self._trackers: dict[str, HungarianTracker] = {}
        self._windows = BehaviorWindows()
        self._in_flight: dict[str, int] = {}

    def cage_ids(self) -> list[str]:
        return self._windows.cage_ids()

    async def submit_frame(self, cage_id: str, frame: bytes, captured_at: datetime) -> bool:
        """Analyse one frame of a cage. False if it was dropped (unreadable or CPU behind)."""
        if self._in_flight.get(cage_id, 0) >= MAX_FRAMES_IN_FLIGHT:
            logger.debug("Dropping a frame of %s: the detector is behind", cage_id)
            return False
        self._in_flight[cage_id] = self._in_flight.get(cage_id, 0) + 1
        try:
            detections, width, height = await asyncio.to_thread(_detect, self._detector, frame)
        except UnreadableFrame:
            logger.warning("Dropping an unreadable frame of %s", cage_id)
            return False
        except Exception:  # a failing model loses one frame, never the service
            logger.exception("The detector failed on a frame of %s", cage_id)
            return False
        finally:
            self._in_flight[cage_id] -= 1
        # tracker and window are only touched here, on the event loop: no locks needed
        tracker = self._trackers.setdefault(cage_id, self._tracker_factory())
        marks = tracker.update(detections, width, height)
        sample = sample_frame(_as_utc(captured_at), marks, width, height, self._zones_for(cage_id))
        self._windows.for_cage(cage_id).add(sample)
        return True

    async def close_window(self, cage_id: str) -> list[BehaviorEventPayload]:
        """The BEHAVIOR payloads of a cage for the window that just ended; the next starts empty."""
        frames = self._windows.for_cage(cage_id).drain()
        if not frames:
            return []
        features = compute_features(frames, self._window_seconds)
        probabilities = await asyncio.to_thread(self._classifier.prob_anomaly, features)
        if self._features_log is not None:
            self._log(cage_id, features, probabilities)
        return [f.to_payload(p) for f, p in zip(features, probabilities, strict=True)]

    def _log(self, cage_id: str, features: list[BehaviorFeatures], probs: list[float]) -> None:
        # a broken log never stops the events: it is only an aid to collect training data
        try:
            path = self._features_log
            new = not path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if new:
                    writer.writerow(
                        [
                            "closedAt",
                            "cageId",
                            "color",
                            *FEATURE_NAMES,
                            "framesSeen",
                            "detectionConfidence",
                            "probAnomaly",
                            "label",
                        ]
                    )
                closed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                for f, p in zip(features, probs, strict=True):
                    writer.writerow(
                        [
                            closed_at,
                            cage_id,
                            f.color.value,
                            *f.vector(),
                            f.frames_seen,
                            f.detection_confidence,
                            round(p, 3),
                            "",
                        ]
                    )
        except OSError as exception:
            logger.warning("Could not write the behavior log: %s", exception)


def _as_utc(moment: datetime) -> datetime:
    # the edge agent sends capturedAt in UTC; a value without zone is taken as UTC
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
