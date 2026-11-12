"""Does the prediction actually work? Drives the parts that run without Bedrock over data whose
answer is known, instead of only asking /ai/health whether a model is loaded.

    uv run python dev/test_prediction.py

Exit code 0 = every prediction landed where it should. It needs no AWS credentials and no trained
model: the audio rule is signal processing and the tracker output is generated here.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# The project is not installed as a package: pytest adds the root through pythonpath = ["."],
# so a script outside tests/ has to do it itself.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.classifier import classify_clip
from app.behavior.features import FEATURE_NAMES, compute_features
from app.behavior.window import sample_frame
from app.contracts.events import AudioLabel, MarkColor
from app.testdata.audio import synthesize_clip
from app.vision.tracker import TrackedMark
from app.vision.zones import CageZones

# What each synthetic clip is: a squeal is the only distress call.
AUDIO_CASES = (
    ("silence", AudioLabel.NORMAL),
    ("rustle", AudioLabel.NORMAL),
    ("chirp", AudioLabel.NORMAL),
    ("squeal", AudioLabel.DISTRESS),
)

FRAME_W, FRAME_H = 640, 480
WINDOW_SECONDS = 60
STILL_RATIO_HIGH = 0.8
STILL_RATIO_LOW = 0.2


def check_audio() -> list[str]:
    print("\n== Audio: loudness x shrillness over synthetic clips ==")
    print(f"{'clip':<10}{'expected':<11}{'got':<11}{'confidence':>11}")
    failures = []
    for kind, expected in AUDIO_CASES:
        result = classify_clip(synthesize_clip(kind))
        ok = result.label == expected
        print(
            f"{kind:<10}{expected.value:<11}{result.label.value:<11}"
            f"{result.probability:>11.2f}   {'ok' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(f"audio/{kind}: expected {expected.value}, got {result.label.value}")
    return failures


def _still_ratio(moves: bool) -> float:
    """One minute of tracker output for a single guinea pig; returns the stillRatio computed."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    frames = []
    for second in range(WINDOW_SECONDS):
        # still: always the same spot. moving: walks across the cage and back.
        x = 100 + (second * 17) % 400 if moves else 100
        mark = TrackedMark(1, MarkColor.RED, 0.9, (x - 10.0, 230.0, x + 10.0, 250.0))
        frames.append(
            sample_frame(start + timedelta(seconds=second), [mark], FRAME_W, FRAME_H, CageZones())
        )
    features = compute_features(frames, WINDOW_SECONDS)
    return features[0].vector()[FEATURE_NAMES.index("stillRatio")]


def check_behavior() -> list[str]:
    print("\n== Behaviour: stillRatio computed from synthetic tracker output ==")
    print(f"{'guinea pig':<14}{'expected':<14}{'stillRatio':>11}")
    failures = []
    for label, moves, bound, ok_test in (
        ("never moves", False, f">= {STILL_RATIO_HIGH}", lambda v: v >= STILL_RATIO_HIGH),
        ("walks around", True, f"<= {STILL_RATIO_LOW}", lambda v: v <= STILL_RATIO_LOW),
    ):
        ratio = _still_ratio(moves=moves)
        ok = ok_test(ratio)
        print(f"{label:<14}{bound:<14}{ratio:>11.2f}   {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"behaviour/{label}: stillRatio {ratio:.2f}, expected {bound}")
    return failures


def main() -> int:
    print("Prediction check - the engines that need no Bedrock and no trained model")
    failures = check_audio() + check_behavior()
    print()
    if failures:
        print(f"{len(failures)} prediction(s) wrong:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Every prediction landed where it should.")
    print("Not covered: the Bedrock vision engine (needs bedrock:InvokeModel, which only the EC2")
    print("role has) and the trained detector/classifier (models/ is empty in a fresh checkout).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
