# Architecture — cuy-monitor-ai-service

> Python 3.14 · FastAPI 0.141 · ONNX Runtime 1.26 · YOLO26n (Ultralytics, exported to ONNX) · scikit-learn 1.9 · httpx · scipy
> Runs as a Docker container (`python:3.14-slim`) next to the backend on the EC2. Last reviewed: 2026-09-26

## 1. Context

```
Farm laptop                                   AWS EC2 (Docker network)
┌──────────────────────┐   HTTPS + X-API-Key  ┌─────────────────────────────────────────────┐
│ edge_agent           │ ───────────────────► │ Caddy /ai/* ─► ai-service :8000             │
│  reads A12 stream    │   POST /ai/frames    │                  │                          │
│  1–2 fps JPEG        │   POST /ai/audio     │                  │ httpx (internal network) │
│  ~1 s WAV clips      │                      │                  ▼                          │
└──────────────────────┘                      │   POST http://backend:8080/api/ingestion/events
        ▲ HTTP (LAN)                          │                  │                          │
 Samsung A12 (IP Webcam)                      │                  ▼                          │
                                              │   cuy-monitor-backend (Java) processes      │
                                              └─────────────────────────────────────────────┘
```

- The service **measures and classifies**; it never decides health state (that's the backend's State + Chain).
- It is the only producer of `BEHAVIOR` and `AUDIO` events, sent to the backend's single ingestion endpoint (no message broker, see backend ADR-007).
- It never touches Postgres.

## 2. Components

```
app/
├── main.py              FastAPI app: /ai/frames, /ai/audio, /ai/health
├── config.py            pydantic-settings: BACKEND_URL, API_KEY, CAGE_ID, thresholds, model paths
├── security.py          X-API-Key dependency
├── contracts/
│   ├── events.py        Pydantic models — faithful copy of backend docs/contracts
│   └── colors.py        MarkColor enum
├── vision/
│   ├── detector.py      YOLO26n ONNX → boxes + class (one class per MarkColor) + confidence
│   ├── tracker.py       Hungarian assignment (scipy.optimize.linear_sum_assignment) keeps IDs across frames
│   └── zones.py         feeder / waterer polygons in image coordinates (per cage, from config)
├── behavior/
│   ├── window.py        60 s sliding window per guinea pig
│   ├── features.py      stillSeconds, feederVisits, watererVisits, avgGroupDistance
│   └── classifier.py    Random Forest (.joblib) → probAnomaly
├── audio/
│   ├── embeddings.py    YAMNet converted to ONNX → 1024-d embeddings
│   └── classifier.py    SVM/KNN over embeddings → NORMAL | DISTRESS + probability
└── messaging/
    └── backend_client.py  httpx.AsyncClient → POST /api/ingestion/events, retry with backoff, bounded buffer
models/                  .onnx / .joblib (not committed if heavy — GitHub Releases or Git LFS)
training/                Colab notebooks and scripts (NOT in the Docker image)
edge_agent/              runs on the farm laptop (NOT in the Docker image)
dev/simulator.py         replays a recorded video as a live camera
tests/
```

## 3. Processing pipeline

### 3.1 Frames → BEHAVIOR events

```
POST /ai/frames (JPEG)
   → decode + resize (320–640 px)
   → detector: [(box, markColor, confidence)]
   → tracker: stable track per markColor
   → window.add(frame_time, tracks, zones)
   → every 60 s per guinea pig:
        features = { stillSeconds, feederVisits, watererVisits, avgGroupDistance }
        probAnomaly = random_forest.predict_proba(features)
        send BEHAVIOR event → POST /api/ingestion/events
```

- Response to the edge_agent is fast (`202 Accepted`); inference must not block the event loop — run ONNX in a thread pool (`run_in_threadpool` / `asyncio.to_thread`).
- If a color is not detected with enough confidence during the window, the event is still sent with low `detectionConfidence` so the backend's `ValidationHandler` can decide. Never guess a color.
- `avgGroupDistance` is normalized to the image diagonal (0–1).

### 3.2 Audio → AUDIO events

```
POST /ai/audio (WAV, ~1 s, 16 kHz mono)
   → YAMNet ONNX embeddings
   → SVM/KNN → label NORMAL | DISTRESS, probability
   → send AUDIO event → POST /api/ingestion/events  (per clip, or aggregated every 10 s)
```

Audio events are cage-level (no guinea pig).

## 4. Contracts (copied from the backend)

Source of truth: `cuy-monitor-backend/docs/contracts/`. `app/contracts/events.py` must match it exactly.

```json
{
  "eventId": "uuid",
  "type": "BEHAVIOR | AUDIO",
  "cageId": "cage-1",
  "timestamp": "2026-10-05T14:32:00Z",
  "source": "ai-service",
  "schemaVersion": 1,
  "payload": { }
}
```

```jsonc
// BEHAVIOR
{ "color": "RED", "windowSeconds": 60, "stillSeconds": 48, "feederVisits": 0,
  "watererVisits": 1, "avgGroupDistance": 0.72, "probAnomaly": 0.81, "detectionConfidence": 0.93 }
// AUDIO
{ "label": "DISTRESS", "probability": 0.88, "durationMs": 960 }
```

- Sent as JSON to `POST {BACKEND_URL}/api/ingestion/events` with header `X-API-Key`. Field names in `camelCase` (use Pydantic `alias_generator=to_camel`).
- `eventId` is generated once (uuid4) and **kept on retries**, so the backend can drop duplicates.
- Backend responses: `202` ok · `400`/`401` log and drop (retrying won't help) · `5xx` or network error → retry with backoff (1 s → 30 s), keep at most ~500 pending events in memory, drop the oldest.
- `MarkColor`: `RED, BLUE, GREEN, YELLOW, ORANGE, PURPLE, BLACK, WHITE`.

## 5. HTTP API

| Method and path | Auth | Body | Response |
|---|---|---|---|
| `POST /ai/frames` | `X-API-Key` | `multipart/form-data`: `file` (JPEG), `capturedAt` (ISO-8601), `cageId` | `202` |
| `POST /ai/audio` | `X-API-Key` | `multipart/form-data`: `file` (WAV), `capturedAt`, `cageId` | `202` |
| `GET /ai/health` | none | — | `{ "status": "UP", "models": { "detector": true, "behavior": true, "audio": true }, "backend": true }` |

**Routing note:** Caddy uses `handle /ai/*` (it does **not** strip the prefix), so FastAPI routes must be declared with the `/ai` prefix.

## 6. Models

| Model | Trained where | Runtime format | File |
|---|---|---|---|
| Detector YOLO26n (classes = mark colors) | Google Colab (GPU) with Ultralytics | ONNX | `models/detector.onnx` |
| Behavior classifier | Colab / local, scikit-learn | joblib | `models/behavior_rf.joblib` |
| YAMNet embeddings | Converted once in Colab (TF → ONNX) | ONNX | `models/yamnet.onnx` |
| Audio classifier (SVM/KNN) | scikit-learn | joblib | `models/audio_clf.joblib` |

- **TensorFlow is not a runtime dependency** (it doesn't support Python 3.14). It's only used inside Colab for the one-time YAMNet export. Plan B if export fails: Python 3.13 + TF 2.21 image.
- Heavy model files are not committed; the Dockerfile downloads them from a GitHub Release (or they're mounted as a volume).
- If a model file is missing, the service starts in **mock mode** (random but contract-valid values) and `/ai/health` reports it.

## 7. edge_agent (farm laptop)

- Python 3.14 script, runs as a service (systemd on Linux; NSSM or Task Scheduler on Windows).
- Reads the A12 stream from IP Webcam (`http://<A12-IP>:8080/video` and audio endpoint) over the LAN.
- Sends 1–2 fps JPEGs and ~1 s WAV clips to `https://cuymonitor.duckdns.org/ai/...` with `X-API-Key`.
- Retries with backoff if the internet drops; drops frames rather than building an unbounded queue.
- Config in `edge_agent/.env` (template `config.example.env`): `A12_URL`, `AI_URL`, `API_KEY`, `CAGE_ID`, `FPS`.

## 8. Configuration

| Variable | Default | Notes |
|---|---|---|
| `BACKEND_URL` | `http://backend:8080` | Set by Compose (internal Docker network, no HTTPS needed) |
| `API_KEY` | — | Same `API_KEY` as the backend (`infra/.env`) |
| `CAGE_ID` | `cage-1` | |
| `WINDOW_SECONDS` | `60` | |
| `MIN_DETECTION_CONFIDENCE` | `0.5` | |
| `MOCK_MODE` | `false` | Force fake events |

## 9. Deployment

- Built by `cuy-monitor-backend/infra/docker-compose.yml` with `build: ../../cuy-monitor-ai-service` → clone both repos side by side on the EC2.
- Starts only with the `ai` profile: `docker compose --profile ai up -d --build`.
- Port 8000 is never published; only reachable through Caddy.
- Needs the EC2 resized to **c7i-flex.large** (4 GB) before enabling it with real models.
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

## 10. Testing

| Level | What |
|---|---|
| Unit | `features.py` with synthetic tracks; tracker ID stability; `events.py` serializes exactly to the contract JSON |
| Contract | Sample JSONs from `cuy-monitor-backend/docs/contracts/examples/` must validate against `events.py` |
| Integration | `dev/simulator.py` + backend running locally (`docker-compose.dev.yml` for Postgres + backend from the IDE) |
| Performance | fps and RAM on the EC2 and on the Celeron laptop (input for ADR-005) |

## 11. Decisions

| Decision | Why |
|---|---|
| Separate Python service (ADR-002) | Ultralytics and scikit-learn are Python-native; backend stays pattern-focused |
| ONNX Runtime for all inference | Light image, CPU-friendly, works on Python 3.14 |
| YOLO26n | Fastest of the family on CPU, NMS-free |
| Identification by mark color | No face/body re-ID needed; simple and explainable |
| Cloud first, maybe laptop later (ADR-005) | Measure bandwidth and fps in October, then decide |
