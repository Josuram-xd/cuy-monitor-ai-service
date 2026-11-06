"""Trains the audio classifier on YAMNet embeddings (Task 11.3): SVM and KNN, keeps the better one.

Runs on a laptop or in Colab, never in the service. It needs models/yamnet.onnx
(training/export_yamnet.py) and onnxruntime + scikit-learn, not TensorFlow. It writes:

- models/audio_clf.joblib: the bundle app/audio/model.py loads.
- models/audio_clf.json: metrics of both models (stratified cross validation), the one chosen, the
  sha256 for AUDIO_MODEL_SHA256 and where the data came from.

Data:

- --data DIR with two folders, DIR/normal and DIR/distress, of 16-bit WAV recordings (any rate,
  any length): each recording is cut into 0.96 s windows. Record the cage (calm sounds, eating,
  the bedding) for normal and the squeals for distress. Recordings stay out of the repo.
- --synthetic N (default without --data): the synthetic clips of app/testdata/audio.py. They only
  prove the chain works; never quote their metrics as results.

    python -m training.train_audio --yamnet models/yamnet.onnx --data recordings/
    python -m training.train_audio --yamnet models/yamnet.onnx --synthetic 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from app.audio.embeddings import SAMPLE_RATE
from app.audio.model import BUNDLE_FORMAT, CLASSES, EMBEDDING
from app.audio.wav import UnsupportedAudio, read_wav, resample
from app.testdata.audio import EXPECTED_LABEL, ClipKind, synthesize_clip

DISTRESS = CLASSES[1]
WINDOW_SECONDS = 0.96
FOLDERS = {"normal": CLASSES[0], "distress": DISTRESS}


def windows(samples: np.ndarray, rate: int) -> Iterator[np.ndarray]:
    """A recording at 16 kHz cut into 0.96 s windows (a last piece of at least half a window is
    kept)."""
    audio = resample(samples, rate, SAMPLE_RATE)
    size = int(WINDOW_SECONDS * SAMPLE_RATE)
    for start in range(0, len(audio), size):
        piece = audio[start : start + size]
        if len(piece) >= size // 2:
            yield piece


def read_recordings(data: Path) -> tuple[list[np.ndarray], list[str], int]:
    """(windows, labels, skipped files) from DIR/normal and DIR/distress."""
    clips, labels, skipped = [], [], 0
    for folder, label in FOLDERS.items():
        for path in sorted((data / folder).glob("*.wav")):
            try:
                samples, rate = read_wav(path.read_bytes())
            except UnsupportedAudio:
                skipped += 1
                continue
            for piece in windows(samples, rate):
                clips.append(piece)
                labels.append(label)
    return clips, labels, skipped


def synthetic_clips(count: int, seed: int = 11) -> tuple[list[np.ndarray], list[str]]:
    """count synthetic windows, the same number of each kind (squeal = DISTRESS)."""
    clips, labels = [], []
    kinds = list(ClipKind)
    for i in range(count):
        kind = kinds[i % len(kinds)]
        samples, rate = read_wav(synthesize_clip(kind, seed=seed + i))
        clips.append(resample(samples, rate, SAMPLE_RATE))
        labels.append(EXPECTED_LABEL[kind].value)
    return clips, labels


def embed_all(embedder, clips: list[np.ndarray]) -> np.ndarray:
    return np.stack([embedder.embed_samples(clip, SAMPLE_RATE) for clip in clips])


def train(features: np.ndarray, labels: list[str], seed: int = 42):
    """(best fitted model, name, metrics of both). Best = higher F1 of DISTRESS, SVM on a tie."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    counts = {label: labels.count(label) for label in CLASSES}
    if min(counts.values()) < 2:
        raise ValueError(f"At least 2 windows of each class are needed, got {counts}")
    folds = min(5, min(counts.values()))
    y = np.array([1 if label == DISTRESS else 0 for label in labels])

    candidates = {
        # calibrated so predict_proba is a real probability (SVC(probability=True) is deprecated)
        "svm": lambda: make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=seed),
                ensemble=False,
            ),
        ),
        "knn": lambda: make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=min(5, max(1, len(y) // folds - 1)))
        ),
    }
    metrics = {}
    for name, build in candidates.items():
        scores = cross_validate(
            build(),
            features,
            y,
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
            scoring=("accuracy", "precision", "recall", "f1"),
        )
        metrics[name] = {
            key.removeprefix("test_"): round(float(value.mean()), 3)
            for key, value in scores.items()
            if key.startswith("test_")
        }
    best = max(candidates, key=lambda name: (metrics[name]["f1"], name == "svm"))
    model = candidates[best]().fit(features, np.array(labels))
    return model, best, {**metrics, "folds": folds}


def save(model, out: Path, metadata: dict) -> dict:
    import joblib

    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format": BUNDLE_FORMAT,
        "embedding": EMBEDDING,
        "classes": list(CLASSES),
        "model": model,
        "metadata": metadata,
    }
    joblib.dump(bundle, out, compress=3)
    report = {
        **metadata,
        "embedding": EMBEDDING,
        "classes": list(CLASSES),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "file": out.name,
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None, embedder=None) -> int:
    parser = argparse.ArgumentParser(description="Train the audio classifier on YAMNet embeddings")
    parser.add_argument("--yamnet", type=Path, default=Path("models/yamnet.onnx"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--data", type=Path, help="folder with normal/ and distress/ WAV recordings"
    )
    source.add_argument("--synthetic", type=int, default=None, help="synthetic clips to use")
    parser.add_argument("--out", type=Path, default=Path("models/audio_clf.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if embedder is None:
        from app.audio.embeddings import YamnetEmbedder

        if not args.yamnet.is_file():
            print(
                f"No YAMNet model at {args.yamnet}: run training/export_yamnet.py", file=sys.stderr
            )
            return 1
        embedder = YamnetEmbedder.load(args.yamnet)

    if args.data:
        clips, labels, skipped = read_recordings(args.data)
        origin = {"data": "real", "source": args.data.name, "skippedFiles": skipped}
    else:
        clips, labels = synthetic_clips(args.synthetic or 200, args.seed)
        origin = {"data": "synthetic", "source": "app/testdata/audio.py"}
    try:
        model, best, metrics = train(embed_all(embedder, clips), labels, args.seed)
    except ValueError as exception:
        print(exception, file=sys.stderr)
        return 1
    report = save(
        model,
        args.out,
        {
            **origin,
            "windows": len(labels),
            "distress": labels.count(DISTRESS),
            "model": best,
            "metrics": metrics,
        },
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
