# Cuy Monitor AI Service

Python 3.14 + FastAPI service that watches and listens to the cage: it recognises each guinea pig by its
coloured mark, measures how it behaves in 60 s windows, classifies the cage audio and sends `BEHAVIOR` and
`AUDIO` events to the backend, which decides the health state. It also holds the **edge agent** that runs on
the farm laptop and the **training scripts** (Colab or a laptop).

See [`docs/PRD.md`](docs/PRD.md) for requirements, [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the
components and [`TASKS.md`](TASKS.md) for the status.

```
A12 phone ─LAN─► edge_agent (laptop) ─HTTPS─► /ai/v1/frames, /ai/v1/audio ─► this service ─HTTP─► backend
                                                 │
            mock events │ Amazon Bedrock │ trained models (detector, tracker, Random Forest, YAMNet)
```

## Engines

The engine is chosen at start-up from `MOCK_MODE` and the model files that loaded; `/ai/health` says which
models are in use. A missing or broken model is never fatal: the service falls back to the next engine.

| Input | `MOCK_MODE=true` | Real mode, without trained models | Real mode, with trained models |
|---|---|---|---|
| Frames | contract-valid fake events (or a scenario) | Amazon Bedrock (Nova 2 Lite) over up to 6 frames per window | `detector.onnx` → Hungarian tracker → 60 s window → `behavior_rf.joblib` |
| Audio | fake events | signal rule: loud × shrill | `yamnet.onnx` embeddings → `audio_clf.joblib` (SVM or KNN) |

## Models

None is committed: they go in a GitHub Release and the image downloads them at build time, checked against
their sha256 (`docker build --build-arg <NAME>_URL=... --build-arg <NAME>_SHA256=...`).

| Model | File | Trained with | Build args | State |
|---|---|---|---|---|
| Detector YOLO26n (one class per mark colour) | `models/detector.onnx` | `training/train_detector.py` (Colab GPU) | `DETECTOR_URL`, `DETECTOR_SHA256` | Chain proven on synthetic frames; needs the labelled cage frames (Tasks 6.1, 6.3, 7.2) |
| Behavior Random Forest | `models/behavior_rf.joblib` | `training/train_behavior.py` | `BEHAVIOR_MODEL_URL`, `BEHAVIOR_MODEL_SHA256` | Synthetic windows only; retrain with `BEHAVIOR_LOG_PATH` windows labelled by hand |
| YAMNet embeddings | `models/yamnet.onnx` | `training/export_yamnet.py` (TensorFlow, Python ≤ 3.13) | `YAMNET_URL`, `YAMNET_SHA256` | Exported and checked against TensorFlow (max difference 5.9e-6) |
| Audio SVM / KNN | `models/audio_clf.joblib` | `training/train_audio.py` | `AUDIO_MODEL_URL`, `AUDIO_MODEL_SHA256` | Needs recordings of the cage in `normal/` and `distress/` |

Step by step: [`training/README.md`](training/README.md). joblib files are pickles: only use models you
trained yourself and always build with their sha256.

## Setup

Install [uv](https://docs.astral.sh/uv/), copy `.env.example` to `.env` and set `API_KEY`, then:

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000   # http://localhost:8000/ai/health
uv run pytest --cov                                # fails under 95 % coverage (Task 14.1)
uv run ruff check . && uv run ruff format .
docker build -t cuy-monitor-ai-service:local .
```

| Variable | Default | What it does |
|---|---|---|
| `API_KEY` | — | Required. The same key as the backend; `/ai/v1/frames` and `/ai/v1/audio` ask for it |
| `BACKEND_URL` | `http://backend:8080` | Where the events go |
| `CAGE_ID` / `CAGE_IDS` | `cage-1` / empty | The cages this service accepts |
| `MOCK_MODE` | `true` | Fake events instead of analysing what arrives |
| `CAGE_ZONES` | empty | Feeder and waterer polygons per cage (JSON, fractions of the frame) |
| `BEHAVIOR_LOG_PATH` | empty | CSV with the features of every window, to label and retrain |
| `*_PATH` | `models/...` | Where each model file is read from |

The full list is in [`.env.example`](.env.example) and `docs/ARCHITECTURE.md` section 8.

## Edge agent (farm laptop)

`python -m edge_agent` reads the A12 (IP Webcam) and sends 1–2 fps frames and 1 s clips, with retries and no
unbounded queue. Install guide for Windows and Linux: [`edge_agent/README.md`](edge_agent/README.md).

## Metrics

`python -m dev.benchmark` times every model loaded on the machine it runs on (detector per frame, a 60 s
behavior window, YAMNet per clip) and its memory. The numbers of the EC2 and of the laptop go in
[`docs/METRICS.md`](docs/METRICS.md) (Task 13); the accuracy of each model comes from the `.json` report its
training script writes.

## Test data to try the models

Nothing here is a recording of a real cage: every sample is generated in `app/testdata/`, so no audio
or frames end up in the repo.

| What | Where | What it tests |
|---|---|---|
| Synthetic clips: `silence`, `rustle`, `chirp` (normal) and `squeal` (distress) | `app/testdata/audio.py` | The audio classifier (loudness x shrillness) |
| Scenarios: `healthy`, `lethargic`, `distress`, `recovery` | `app/testdata/scenarios.py` | The backend's anomaly chain: NORMAL → OBSERVED → ALERT → CRITICAL and back |
| Recorded answers of the vision model, replayed by `ScriptedRuntime` | `app/testdata/model_answers.py` | The Bedrock analyzer, offline and without cost |

Replay a scenario as the mock producer, in every cage of `CAGE_IDS`:

```bash
MOCK_MODE=true MOCK_SCENARIO=lethargic MOCK_COLORS=RED,BLUE uv run uvicorn app.main:app --port 8000
```

Or inject it straight into a backend, faster than one window a minute:

```bash
uv run python -m dev.inject_test_data --api-key "$API_KEY" --cage-id cage-1 \
  behavior --scenario lethargic --colors RED,BLUE --windows 12 --interval 5 --backend-url http://localhost:8080
uv run python -m dev.inject_test_data --api-key "$API_KEY" audio --kind squeal --count 3 --ai-url http://localhost:8000
```

The colours must be guinea pigs registered in that cage, or the backend drops the windows. With the
backend's default times, ALERT needs the anomaly to last 30 minutes in OBSERVED; for a quick try run
the backend with `--app.health.observed-to-alert=5s --app.health.alert-to-critical=5s`.

## Real mode (Amazon Bedrock)

With `MOCK_MODE=false` the service stops inventing events and looks at what it receives:

- `POST /ai/v1/frames` (JPEG) keeps the frames. Once per window (`WINDOW_SECONDS`, 60 s) it sends up to
  `WINDOW_MAX_FRAMES` of them, evenly spread, to a Bedrock vision model. The model reports, for each
  guinea pig whose mark colour it can identify, how many frames it was still, at the feeder or the
  water bottle, how far from the others and how unusual it looks. Each one becomes a `BEHAVIOR`
  event of the backend contract (one per guinea pig per window).
- `POST /ai/v1/audio` (16-bit WAV) is classified with a signal rule (loud and shrill = `DISTRESS`),
  or with YAMNet + SVM/KNN when those models are loaded, and sent as an `AUDIO` event. Calm clips are reported once per window.
- Whatever the model returns is validated before it is used: invalid or impossible items are
  dropped, and if Bedrock fails the window is skipped (no invented data).

On the EC2 the credentials come from the instance role (`bedrock:InvokeModel`), so no AWS key is
configured. Locally use `AWS_PROFILE=<profile>`. Default model: Amazon Nova 2 Lite
(`us.amazon.nova-2-lite-v1:0`); change `BEDROCK_MODEL_ID` to use Nova Pro or a Claude model.

```bash
docker build -t cuy-monitor-ai-service:local .
docker run --rm -p 8000:8000 -e API_KEY=local-secret -e MOCK_MODE=true cuy-monitor-ai-service:local
```

To replay a local recording at a maximum of one frame per second, set `API_KEY` in the
environment and run:

```powershell
$env:API_KEY = "<local-api-key>"
uv run python dev/simulator.py .\recording.mp4 --ai-url http://localhost:8000 --cage-id cage-1
```

The simulator samples frames according to the source video's frame rate. Use `--source-fps`
if the video does not provide valid frame-rate metadata.
