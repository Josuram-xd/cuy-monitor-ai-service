import json

import joblib
import pytest

from app.behavior.classifier import BUNDLE_FORMAT, BehaviorClassifier, load_behavior_classifier
from app.behavior.features import FEATURE_NAMES, BehaviorFeatures
from app.contracts import MarkColor
from training.train_behavior import main as train_main
from training.train_behavior import read_windows, synthetic_windows, train


def _features(still: float, feeder: int = 0, waterer: int = 0, distance: float = 0.5):
    return BehaviorFeatures(MarkColor.RED, 60, still, feeder, waterer, distance, 0.9, 100)


class _FakeModel:
    classes_ = [0, 1]

    def __init__(self):
        self.rows = None

    def predict_proba(self, rows):
        self.rows = rows
        return [[1 - r[0], r[0]] for r in rows]


def test_prob_anomaly_reads_the_anomaly_column_in_feature_order():
    model = _FakeModel()
    classifier = BehaviorClassifier(model)

    probs = classifier.prob_anomaly([_features(54, 1, 0, 0.8), _features(6)])

    assert probs == pytest.approx([0.9, 0.1])
    assert model.rows[0] == [0.9, 1.0, 0.0, 0.8]
    assert classifier.prob_anomaly([]) == []


def test_a_model_trained_with_other_features_is_refused():
    with pytest.raises(ValueError, match="features"):
        BehaviorClassifier(_FakeModel(), ("stillSeconds", "feederVisits"))


def test_a_model_without_the_anomaly_class_is_refused():
    model = _FakeModel()
    model.classes_ = [0]
    with pytest.raises(ValueError, match="anomaly"):
        BehaviorClassifier(model)


def test_a_missing_or_broken_classifier_is_never_fatal(tmp_path):
    assert load_behavior_classifier(tmp_path / "missing.joblib") is None

    broken = tmp_path / "broken.joblib"
    broken.write_bytes(b"not a joblib file")
    assert load_behavior_classifier(broken) is None

    other = tmp_path / "other.joblib"
    joblib.dump({"format": BUNDLE_FORMAT + 1}, other)
    assert load_behavior_classifier(other) is None


def test_synthetic_windows_are_labelled_and_in_feature_order():
    rows, labels = synthetic_windows(400, seed=1)

    assert len(rows) == len(labels) == 400
    assert all(len(r) == len(FEATURE_NAMES) for r in rows)
    assert 0.3 < sum(labels) / len(labels) < 0.7
    assert all(0 <= r[0] <= 1 and 0 <= r[3] <= 1 for r in rows)


def test_the_trained_forest_separates_lethargic_from_healthy_windows():
    model, metrics, importances = train(*synthetic_windows(600, seed=3))
    classifier = BehaviorClassifier(model)

    lethargic, healthy = classifier.prob_anomaly(
        [_features(56, 0, 0, 0.85), _features(15, 2, 1, 0.3)]
    )

    assert lethargic > 0.8
    assert healthy < 0.2
    assert metrics["roc_auc"] > 0.9
    assert set(importances) == set(FEATURE_NAMES)


def test_training_needs_both_classes():
    with pytest.raises(ValueError):
        train([[0.1, 1, 0, 0.3]] * 10, [0] * 10)


def test_real_windows_are_read_from_the_behavior_log_and_unlabelled_rows_skipped(tmp_path):
    data = tmp_path / "windows.csv"
    data.write_text(
        "closedAt,cageId,color,stillRatio,feederVisits,watererVisits,avgGroupDistance,label\n"
        "t,cage-1,RED,0.9,0,0,0.8,ANOMALY\n"
        "t,cage-1,BLUE,0.2,2,1,0.3,0\n"
        "t,cage-1,GREEN,0.5,1,0,0.4,\n",
        encoding="utf-8",
    )

    rows, labels = read_windows(data)

    assert rows == [[0.9, 0.0, 0.0, 0.8], [0.2, 2.0, 1.0, 0.3]]
    assert labels == [1, 0]


def test_a_csv_with_an_unknown_label_or_missing_columns_is_refused(tmp_path):
    bad_label = tmp_path / "bad.csv"
    bad_label.write_text(
        "stillRatio,feederVisits,watererVisits,avgGroupDistance,label\n0.5,1,0,0.4,maybe\n"
    )
    missing = tmp_path / "missing.csv"
    missing.write_text("stillRatio,label\n0.5,1\n")

    with pytest.raises(ValueError, match="label"):
        read_windows(bad_label)
    with pytest.raises(ValueError, match="columns"):
        read_windows(missing)


def test_the_script_writes_a_bundle_the_service_loads_and_its_report(tmp_path):
    out = tmp_path / "behavior_rf.joblib"

    assert train_main(["--synthetic", "300", "--out", str(out)]) == 0

    classifier = load_behavior_classifier(out)
    assert classifier is not None
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["data"] == "synthetic"
    assert report["features"] == list(FEATURE_NAMES)
    assert len(report["sha256"]) == 64
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(report["metrics"])
