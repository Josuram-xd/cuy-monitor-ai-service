# Metrics (Task 13)

Two kinds of numbers, from two places:

- **Cost per machine** (Task 13.1): `python -m dev.benchmark` with the same models on the EC2 and on the farm
  laptop. It decides ADR-005: analyse in the cloud or on the laptop.
- **Quality per model** (Task 13.2): the `.json` report each training script writes next to its model, from
  real data of the cage. Synthetic data only proves the chain works: never quote it here.

## 1. Cost per machine

```bash
python -m dev.benchmark --detector models/detector.onnx --behavior-model models/behavior_rf.joblib \
    --yamnet models/yamnet.onnx --audio-model models/audio_clf.joblib --out docs/metrics/<machine>.json
```

| Machine | Detector p50 / fps | 60 s behavior window | YAMNet per 1 s clip | Peak memory | Date |
|---|---|---|---|---|---|
| Dev laptop, i7-7500U (4 threads), models of the synthetic data, detector imgsz 320 | 16.1 ms / 61 fps (p95 19.4 ms) | 23.6 ms | 3.4 ms (classifier 6.7 ms) | 211 MB | 2026-11-06 |
| EC2 `c7i-flex.large` | _pending_ | _pending_ | _pending_ | _pending_ | |
| Farm laptop (Celeron) | _pending_ | _pending_ | _pending_ | _pending_ | |

At 2 fps a frame arrives every 500 ms: the detector p95 has to stay well under that on the machine that runs it.

The dev laptop row only shows the order of magnitude: its detector was trained on synthetic frames at 320 px
(a 640 px model does about 4x the work) and nothing else ran at the same time. The rows that decide ADR-005 are
the EC2 and the farm laptop, with the models trained on the cage.

## 2. Quality per model

| Model | Data | Metrics | Report |
|---|---|---|---|
| Detector YOLO26n | _labelled cage frames (Tasks 6.1, 6.3)_ | mAP50, mAP50-95 per colour | `detector.json` |
| Behavior Random Forest | _windows from `BEHAVIOR_LOG_PATH`, labelled by hand_ | accuracy, precision, recall, F1, ROC AUC | `behavior_rf.json` |
| Audio SVM / KNN | _recordings in `normal/` and `distress/`_ | accuracy, precision, recall, F1 of `DISTRESS` | `audio_clf.json` |
| YAMNet export | TF Hub `google/yamnet/1` | max difference ONNX vs TensorFlow: **5.9e-6** | `yamnet.json` |

`DISTRESS` is the hard class (few examples): look at its recall first, a missed squeal is worse than a
false one.
