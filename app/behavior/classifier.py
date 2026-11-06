"""The behavior classifier: features of a window -> probAnomaly (Task 10.2).

A Random Forest trained by training/train_behavior.py and saved as a joblib bundle:

    {"format": 1, "features": FEATURE_NAMES, "model": RandomForestClassifier, "metadata": {...}}

The bundle says which features it was trained with; if they are not FEATURE_NAMES in that order the
model is refused instead of giving wrong probabilities. A missing or broken file is never fatal: the
service keeps running with Bedrock and /ai/health says the trained classifier is not there.

joblib files are pickles and can run code when loaded: only load the file the image build
downloaded and checked with BEHAVIOR_MODEL_SHA256, never one uploaded at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from app.behavior.features import FEATURE_NAMES, BehaviorFeatures

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = 1
ANOMALY_CLASS = 1


class ProbabilisticModel(Protocol):
    """The part of a scikit-learn classifier this module uses (a fake one in the tests)."""

    classes_: Any

    def predict_proba(self, rows: list[list[float]]) -> Any: ...


class BehaviorClassifier:
    def __init__(self, model: ProbabilisticModel, features: tuple[str, ...] = FEATURE_NAMES):
        if tuple(features) != FEATURE_NAMES:
            raise ValueError(
                f"The model was trained with features {list(features)}, "
                f"the service computes {list(FEATURE_NAMES)}"
            )
        classes = [int(c) for c in model.classes_]
        if ANOMALY_CLASS not in classes:
            raise ValueError("The model has no anomaly class (1)")
        self._model = model
        self._anomaly_column = classes.index(ANOMALY_CLASS)

    @classmethod
    def load(cls, path: Path) -> BehaviorClassifier:
        import joblib

        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"{path} is not a behavior model bundle (format {BUNDLE_FORMAT})")
        return cls(bundle["model"], tuple(bundle.get("features", ())))

    def prob_anomaly(self, features: list[BehaviorFeatures]) -> list[float]:
        """probAnomaly of each window, in the same order."""
        if not features:
            return []
        probabilities = self._model.predict_proba([f.vector() for f in features])
        return [float(row[self._anomaly_column]) for row in probabilities]


def load_behavior_classifier(path: Path) -> BehaviorClassifier | None:
    if not path.is_file():
        logger.warning("No behavior classifier at %s: running without it", path)
        return None
    try:
        classifier = BehaviorClassifier.load(path)
    except Exception as exception:  # any loading problem means "no classifier"
        logger.error("The behavior classifier at %s could not be loaded: %s", path, exception)
        return None
    logger.info("Behavior classifier loaded from %s", path)
    return classifier
