import logging
from pathlib import Path

from app.vision.animal_detector import OnnxAnimalDetector
from app.vision.detector import OnnxDetector

logger = logging.getLogger(__name__)


def load_detector(path: Path, confidence: float) -> OnnxDetector | None:
    """The detector if its file is there and loads; None otherwise.

    A missing or broken model is never fatal: the service keeps running (mock events or Bedrock)
    and /ai/health says the detector is not there.
    """
    if not path.is_file():
        logger.warning("No detector at %s: running without it", path)
        return None
    try:
        detector = OnnxDetector.load(path, confidence=confidence)
    except Exception as exception:  # any loading problem means "no detector"
        logger.error("The detector at %s could not be loaded: %s", path, exception)
        return None
    logger.info("Detector loaded from %s", path)
    return detector


def load_animal_detector(path: Path, confidence: float) -> OnnxAnimalDetector | None:
    """The animal detector if its file is there and loads; None otherwise (never fatal)."""
    if not path.is_file():
        logger.warning("No animal detector at %s: identifying by marks only", path)
        return None
    try:
        detector = OnnxAnimalDetector.load(path, confidence=confidence)
    except Exception as exception:  # any loading problem means "no animal detector"
        logger.error("The animal detector at %s could not be loaded: %s", path, exception)
        return None
    logger.info("Animal detector loaded from %s", path)
    return detector
