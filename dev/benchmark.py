"""Measures what each model costs on this machine: time per frame / clip and memory (Task 13.1).

Run it on the EC2 and on the farm laptop with the same models; the numbers decide ADR-005 (analyse
in the cloud or on the laptop) and go in docs/METRICS.md (Task 13.2).

  python -m dev.benchmark --detector models/detector.onnx --yamnet models/yamnet.onnx \
      --behavior-model models/behavior_rf.joblib --audio-model models/audio_clf.joblib

Every model is optional: what is missing is skipped. The frames and clips are synthetic (no camera
needed), so the times are those of the models, not of the network.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2

from app.audio.classifier import classify_clip
from app.testdata.audio import ClipKind, synthesize_clip


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process (MB), or None where the OS doesn't tell (Windows)."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kB, macOS bytes
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def timed(action: Callable[[], object], runs: int, warmup: int = 2) -> dict[str, float]:
    """p50 / p95 / mean in milliseconds over `runs` calls, after a few warm-up calls."""
    for _ in range(warmup):
        action()
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        action()
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return {
        "p50Ms": round(statistics.median(samples), 2),
        "p95Ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 2),
        "meanMs": round(statistics.fmean(samples), 2),
        "runs": runs,
    }


def synthetic_jpegs(count: int, width: int = 640, height: int = 480) -> list[bytes]:
    from training.synthetic_dataset import render_frame

    frames = []
    for seed in range(count):
        image, _ = render_frame(seed)
        image = cv2.resize(image, (width, height))
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            raise RuntimeError("could not encode a synthetic frame")
        frames.append(data.tobytes())
    return frames


def bench_detector(path: Path, frames: list[bytes], runs: int) -> dict:
    from app.vision.detector import OnnxDetector

    detector = OnnxDetector.load(path)
    cycle = iter(frames * (runs // len(frames) + 3))
    result = timed(lambda: detector.detect(next(cycle)), runs)
    result["fps"] = round(1000 / result["meanMs"], 1)
    return result


def bench_window(path: Path | None, detector_path: Path | None, frames: list[bytes]) -> dict:
    """One 60 s window at 2 fps: tracker + features (+ Random Forest when given)."""
    from app.behavior.features import compute_features
    from app.behavior.window import sample_frame
    from app.vision.detector import Detection, OnnxDetector, decode_jpeg
    from app.vision.tracker import HungarianTracker
    from app.vision.zones import NO_ZONES

    detector = OnnxDetector.load(detector_path) if detector_path else None
    start = datetime.now(UTC)
    detections = []
    for i in range(120):
        image = decode_jpeg(frames[i % len(frames)])
        found = detector.detect_image(image) if detector else []
        if not found:  # without a detector, two fake guinea pigs walking around
            from app.contracts import MarkColor

            found = [
                Detection(MarkColor.RED, 0.9, (100 + i, 100, 140 + i, 140)),
                Detection(MarkColor.BLUE, 0.9, (400, 300 - i, 440, 340 - i)),
            ]
        detections.append((image.shape[1], image.shape[0], found))

    classifier = None
    if path:
        from app.behavior.classifier import BehaviorClassifier

        classifier = BehaviorClassifier.load(path)

    def window():
        tracker = HungarianTracker()
        samples = [
            sample_frame(
                start + timedelta(seconds=i / 2), tracker.update(found, w, h), w, h, NO_ZONES
            )
            for i, (w, h, found) in enumerate(detections)
        ]
        features = compute_features(samples, 60)
        if classifier:
            classifier.prob_anomaly(features)

    return timed(window, runs=20, warmup=1)


def bench_audio(yamnet: Path | None, model: Path | None, runs: int) -> dict:
    clips = [synthesize_clip(kind, seconds=1.0, seed=i) for i, kind in enumerate(ClipKind)]
    cycle = iter(clips * (runs // len(clips) + 3))
    result = {"signalRule": timed(lambda: classify_clip(next(cycle)), runs)}
    if yamnet:
        from app.audio.embeddings import YamnetEmbedder

        embedder = YamnetEmbedder.load(yamnet)
        result["yamnetEmbedding"] = timed(lambda: embedder.embed(next(cycle)), runs)
        if model:
            from app.audio.model import TrainedAudioClassifier

            classifier = TrainedAudioClassifier.load(embedder, model)
            result["trainedClassifier"] = timed(lambda: classifier.classify(next(cycle)), runs)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time and memory of each model on this machine")
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--behavior-model", type=Path)
    parser.add_argument("--yamnet", type=Path)
    parser.add_argument("--audio-model", type=Path)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--out", type=Path, help="also write the JSON here")
    args = parser.parse_args(argv)

    for path in (args.detector, args.behavior_model, args.yamnet, args.audio_model):
        if path and not path.is_file():
            parser.error(f"{path} does not exist")

    report: dict = {
        "machine": {
            "host": platform.node(),
            "system": f"{platform.system()} {platform.release()}",
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
        },
        "measuredAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "memoryAtStartMb": peak_memory_mb(),
    }
    frames = synthetic_jpegs(20)
    if args.detector:
        report["detector"] = bench_detector(args.detector, frames, args.runs)
    report["behaviorWindow"] = bench_window(args.behavior_model, args.detector, frames)
    report["audio"] = bench_audio(args.yamnet, args.audio_model, args.runs)
    report["peakMemoryMb"] = peak_memory_mb()

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
