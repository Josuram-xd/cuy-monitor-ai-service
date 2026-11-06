# Training

Scripts that run **outside** the service (Google Colab or a laptop). Not copied into the Docker image.

| Step | Task | How |
|---|---|---|
| Record the cage (different hours and lights) | 6.1 | Phone or webcam; `dev/simulator.py` replays a video as a camera |
| Label 300–600 frames | 6.2, 6.3 | Follow [LABELING.md](LABELING.md) in Label Studio or Roboflow |
| Prepare the dataset | 6.4 | `python -m training.prepare_dataset --export <export> --out <dataset>` |
| Train YOLO26n and export ONNX | 7.1, 7.2 | In Colab: `python -m training.train_detector --data <dataset>/data.yaml` |
| Publish the model | 7.2 | Upload `models/detector.onnx` and `detector.json` to a GitHub Release |
| Use it | 7.3, 7.4 | Docker build with `DETECTOR_URL` and `DETECTOR_SHA256`, or mount it in `models/` |
| Train the behavior classifier | 10.1 | `python -m training.train_behavior --synthetic 2000` (or `--data windows.csv`) |
| Export YAMNet to ONNX | 11.1 | In Colab: `pip install -r training/requirements-audio.txt` and `python training/export_yamnet.py` |
| Train the audio classifier | 11.3 | `python -m training.train_audio --data recordings/` (folders `normal/` and `distress/`) |
| Publish it | 10.2 | Upload `behavior_rf.joblib` and `behavior_rf.json`; build with `BEHAVIOR_MODEL_URL` and `BEHAVIOR_MODEL_SHA256` |

## Colab

```bash
!git clone https://github.com/Josuram-xd/cuy-monitor-ai-service.git && cd cuy-monitor-ai-service
!pip install -r training/requirements.txt numpy opencv-python-headless pydantic
# dataset: upload the export (or unzip it from Drive), then
!python -m training.prepare_dataset --export /content/export --out /content/cuy-dataset
!python -m training.train_detector --data /content/cuy-dataset/data.yaml --epochs 100
```

## Without the cage yet

`training/synthetic_dataset.py` draws guinea pigs with a coloured mark on bedding under different
lights, with their YOLO labels. It is not a replacement for real frames: it lets the whole chain
(prepare → train → ONNX → `OnnxDetector`) run before recording.

```bash
python -m training.synthetic_dataset --out /tmp/synthetic --frames 300
python -m training.prepare_dataset --export /tmp/synthetic --out /tmp/synthetic-dataset
```

The class order is `app.vision.classes.DETECTOR_CLASSES` (the order of `MarkColor`). Both the dataset
and the service read it from there, so a model can't be trained with one order and read with another.

## Behavior classifier (Task 10)

`training/train_behavior.py` trains a Random Forest over `FEATURE_NAMES`
(`stillRatio, feederVisits, watererVisits, avgGroupDistance`) and writes the bundle plus a JSON
report with cross-validation metrics and the sha256 to pass as `BEHAVIOR_MODEL_SHA256`.

- Without real data, `--synthetic N` draws windows from the test-scenario profiles. The metrics of
  that model only prove the chain works; do not quote them as results.
- To train with real windows, run the service with `BEHAVIOR_LOG_PATH=behavior-windows.csv`, fill
  the `label` column (1/ANOMALY or 0/NORMAL) after checking the video, and run
  `python -m training.train_behavior --data behavior-windows.csv`.
- joblib files are pickles: only use models you trained yourself and always build with the sha256.

## Audio classifier (Task 11)

1. **YAMNet to ONNX, once, in Colab** (TensorFlow does not run on Python 3.14):
   `pip install -r training/requirements-audio.txt && python training/export_yamnet.py`. The script checks the
   ONNX answers like TensorFlow (largest difference under 1e-3) and writes `yamnet.json` with the sha256.
2. **Record the cage**: put WAVs in `recordings/normal/` (calm, eating, bedding, people talking near) and
   `recordings/distress/` (squeals). Any rate, any length: they are cut into 0.96 s windows. Never commit them.
3. **Train** (laptop or Colab, no TensorFlow): `python -m training.train_audio --data recordings/`. It compares an
   SVM and a KNN with a stratified cross validation and keeps the one with the better F1 of `DISTRESS`.
4. **Publish** `yamnet.onnx` and `audio_clf.joblib` (with their `.json`) in a GitHub Release and build with
   `YAMNET_URL`, `YAMNET_SHA256`, `AUDIO_MODEL_URL` and `AUDIO_MODEL_SHA256`.

`--synthetic N` uses the generated clips of `app/testdata/audio.py`: it proves the chain, not the model.
