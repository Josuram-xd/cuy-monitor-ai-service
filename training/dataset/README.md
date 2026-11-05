# Labeled dataset (labels only)

The export of Task 6.4 goes here, made with:

```bash
python -m training.prepare_dataset --export <label-studio-export> --out training/dataset --labels-only
```

- `labels/train/*.txt`, `labels/val/*.txt` — YOLO labels, classes in the detector's order.
- `data.yaml` — the classes in that order.
- `manifest.csv` — frame name, image file, sha256 of the image, split and number of boxes.

The frames themselves are **never committed** (raw cage data, see AGENTS.md): they live in the team's
drive under the same names, and `manifest.csv` tells which image each label belongs to.
Empty until the cage is recorded and labeled (Tasks 6.1 and 6.3).
