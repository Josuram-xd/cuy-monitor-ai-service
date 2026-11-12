"""The trained audio classifier: YAMNet embedding -> SVM or KNN -> NORMAL / DISTRESS (Task 11.4).

The model is a joblib bundle written by training/train_audio.py:

    {"format": 1, "embedding": "yamnet-mean-1024", "classes": [...], "model": ..., "metadata": ...}

With both yamnet.onnx and audio_clf.joblib loaded, /ai/v1/audio uses it; without either one it keeps
the signal rule of app/audio/classifier.py, and /ai/health says which. joblib files are pickles:
only the file the image build downloaded and checked with AUDIO_MODEL_SHA256 is loaded, never one
uploaded at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.audio.classifier import AudioResult
from app.audio.embeddings import YamnetEmbedder, load_embedder
from app.audio.wav import read_wav
from app.contracts.events import AudioLabel

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = 1
EMBEDDING = "yamnet-mean-1024"
CLASSES = (AudioLabel.NORMAL.value, AudioLabel.DISTRESS.value)


class ProbabilisticModel(Protocol):
    classes_: Any

    def predict_proba(self, rows: Any) -> Any: ...


class Embedder(Protocol):
    def embed_samples(self, samples: np.ndarray, rate: int) -> np.ndarray: ...


class TrainedAudioClassifier:
    def __init__(self, embedder: Embedder, model: ProbabilisticModel):
        classes = [str(c) for c in model.classes_]
        if sorted(classes) != sorted(CLASSES):
            raise ValueError(f"The audio model has classes {classes}, expected {list(CLASSES)}")
        self._embedder = embedder
        self._model = model
        self._distress_column = classes.index(AudioLabel.DISTRESS.value)

    @classmethod
    def load(cls, embedder: Embedder, path: Path) -> TrainedAudioClassifier:
        import joblib

        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"{path} is not an audio model bundle (format {BUNDLE_FORMAT})")
        if bundle.get("embedding") != EMBEDDING:
            raise ValueError(f"{path} was trained on {bundle.get('embedding')!r}, not {EMBEDDING}")
        return cls(embedder, bundle["model"])

    def classify(self, wav_bytes: bytes) -> AudioResult:
        """Same answer as the signal rule: the label and how sure the model is of that label."""
        samples, rate = read_wav(wav_bytes)
        vector = self._embedder.embed_samples(samples, rate)
        distress = float(self._model.predict_proba(np.asarray([vector]))[0][self._distress_column])
        duration_ms = max(1, round(1000 * samples.size / rate))
        if distress >= 0.5:
            return AudioResult(AudioLabel.DISTRESS, round(distress, 2), duration_ms)
        return AudioResult(AudioLabel.NORMAL, round(1 - distress, 2), duration_ms)


def load_audio_classifier(yamnet_path: Path, model_path: Path) -> TrainedAudioClassifier | None:
    """Both files loaded -> the trained classifier; anything missing or broken -> None, never
    fatal."""
    if not model_path.is_file():
        logger.warning("No audio model at %s: audio uses the signal rule", model_path)
        return None
    embedder: YamnetEmbedder | None = load_embedder(yamnet_path)
    if embedder is None:
        return None
    try:
        classifier = TrainedAudioClassifier.load(embedder, model_path)
    except Exception as exception:  # any loading problem means "no trained audio model"
        logger.error("The audio model at %s could not be loaded: %s", model_path, exception)
        return None
    logger.info("Audio classifier loaded: YAMNet %s + %s", yamnet_path, model_path)
    return classifier
