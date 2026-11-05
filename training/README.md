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
