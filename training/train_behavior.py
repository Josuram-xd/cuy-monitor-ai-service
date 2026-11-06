"""Trains the behavior classifier (Task 10.1): Random Forest over the features of a 60 s window.

Runs in Colab or on a laptop, never in the service. It writes:

- models/behavior_rf.joblib: the bundle the service loads (app/behavior/classifier.py).
- models/behavior_rf.json: features, metrics of a stratified cross validation, sha256 and where the
  data came from, to publish next to it in the GitHub Release and quote in the report.

Data:

- --data windows.csv: real windows. The service writes them when BEHAVIOR_LOG_PATH is set (one row
  per guinea pig and window); fill the empty "label" column with 1 (anomaly) or 0 (normal), or
  ANOMALY / NORMAL, after checking the video or the farm notes. Rows without label are skipped.
- --synthetic N (default when there is no --data): windows drawn from the same profiles as the
  test scenarios (app/testdata/scenarios.py), plus resting and early-sick windows so the classes
  overlap. It proves the chain end to end; it is NOT a replacement for real labelled windows.

    python -m training.train_behavior --synthetic 2000 --out models/behavior_rf.joblib
    python -m training.train_behavior --data behavior-windows.csv --out models/behavior_rf.joblib
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

from app.behavior.classifier import ANOMALY_CLASS, BUNDLE_FORMAT
from app.behavior.features import FEATURE_NAMES
from app.contracts import MarkColor
from app.testdata.scenarios import healthy_window, lethargic_window

_LABELS = {"1": 1, "0": 0, "ANOMALY": 1, "ANOMALOUS": 1, "NORMAL": 0, "SICK": 1, "HEALTHY": 0}


def _row(still_ratio: float, feeder: int, waterer: int, distance: float) -> list[float]:
    return [round(still_ratio, 3), float(feeder), float(waterer), round(distance, 2)]


def synthetic_windows(count: int, seed: int = 7) -> tuple[list[list[float]], list[int]]:
    """count labelled windows, about half normal and half anomalous, in FEATURE_NAMES order."""
    rng = random.Random(seed)
    rows: list[list[float]] = []
    labels: list[int] = []
    for i in range(count):
        kind = rng.random()
        if kind < 0.35:  # the healthy profile of the scenarios
            p = healthy_window(MarkColor.RED, i, 60, seed)
            rows.append(
                _row(p.still_seconds / 60, p.feeder_visits, p.waterer_visits, p.avg_group_distance)
            )
            labels.append(0)
        elif kind < 0.5:  # a healthy guinea pig resting: still a lot, but close to the group
            rows.append(
                _row(
                    rng.uniform(0.5, 0.8),
                    rng.randint(0, 1),
                    rng.randint(0, 1),
                    rng.uniform(0.15, 0.5),
                )
            )
            labels.append(0)
        elif kind < 0.85:  # the lethargic profile of the scenarios
            p = lethargic_window(MarkColor.RED, i, 60, seed)
            rows.append(
                _row(p.still_seconds / 60, p.feeder_visits, p.waterer_visits, p.avg_group_distance)
            )
            labels.append(1)
        else:  # getting sick: still and apart, maybe one last visit to the feeder
            rows.append(
                _row(rng.uniform(0.65, 0.9), rng.choice([0, 0, 1]), 0, rng.uniform(0.55, 0.85))
            )
            labels.append(1)
    return rows, labels


def read_windows(path: Path) -> tuple[list[list[float]], list[int]]:
    """Labelled rows of a CSV with the FEATURE_NAMES columns and a label column."""
    rows: list[list[float]] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = [c for c in (*FEATURE_NAMES, "label") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} has no columns {missing}")
        for line, record in enumerate(reader, start=2):
            label = (record["label"] or "").strip().upper()
            if not label:
                continue
            if label not in _LABELS:
                raise ValueError(f"{path}:{line}: unknown label {record['label']!r}")
            rows.append([float(record[name]) for name in FEATURE_NAMES])
            labels.append(_LABELS[label])
    return rows, labels


def train(rows: list[list[float]], labels: list[int], seed: int = 42):
    """The fitted Random Forest and its cross validation metrics."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate

    if len(set(labels)) < 2:
        raise ValueError("The data needs normal (0) and anomalous (1) windows")
    smallest = min(labels.count(0), labels.count(ANOMALY_CLASS))
    folds = min(5, smallest)
    if folds < 2:
        raise ValueError("At least 2 windows of each class are needed")

    def model():
        # shallow trees: few features and few real windows, avoid learning noise
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
        )

    scores = cross_validate(
        model(),
        rows,
        labels,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        scoring=("accuracy", "precision", "recall", "f1", "roc_auc"),
    )
    metrics = {
        name.removeprefix("test_"): round(float(values.mean()), 3)
        for name, values in scores.items()
        if name.startswith("test_")
    }
    metrics["folds"] = folds
    fitted = model().fit(rows, labels)
    importances = dict(
        zip(FEATURE_NAMES, (round(float(v), 3) for v in fitted.feature_importances_), strict=True)
    )
    return fitted, metrics, importances


def save(model, out: Path, metadata: dict) -> dict:
    import joblib

    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format": BUNDLE_FORMAT,
        "features": FEATURE_NAMES,
        "model": model,
        "metadata": metadata,
    }
    joblib.dump(bundle, out, compress=3)
    report = {
        **metadata,
        "features": list(FEATURE_NAMES),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "file": out.name,
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=Path, help="CSV of labelled real windows")
    source.add_argument("--synthetic", type=int, default=None, help="synthetic windows to draw")
    parser.add_argument("--out", type=Path, default=Path("models/behavior_rf.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.data:
        rows, labels = read_windows(args.data)
        origin = {"data": "real", "source": args.data.name}
    else:
        count = args.synthetic or 2000
        rows, labels = synthetic_windows(count, args.seed)
        origin = {"data": "synthetic", "source": "app/testdata/scenarios.py profiles"}
    try:
        model, metrics, importances = train(rows, labels, args.seed)
    except ValueError as exception:
        print(exception, file=sys.stderr)
        return 1
    report = save(
        model,
        args.out,
        {
            **origin,
            "windows": len(rows),
            "anomalous": labels.count(ANOMALY_CLASS),
            "metrics": metrics,
            "featureImportances": importances,
        },
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
