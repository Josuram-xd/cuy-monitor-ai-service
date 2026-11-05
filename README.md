# Cuy Monitor AI Service

Python 3.14 service for processing cage camera and audio input. See
[`docs/PRD.md`](docs/PRD.md) for requirements, [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for component boundaries, and [`TASKS.md`](TASKS.md) for implementation status.

## Development

Install [uv](https://docs.astral.sh/uv/), then run:

Copy `.env.example` to `.env` and replace `API_KEY` with a local secret.

```bash
uv sync
uv run pytest
uv run ruff check .
```

The service exposes authenticated `POST /ai/frames` and `POST /ai/audio` multipart routes.
In mock mode they accept supported media and a background producer sends contract-valid
`BEHAVIOR` and `AUDIO` events to the backend once per configured window.

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

- `POST /ai/frames` (JPEG) keeps the frames. Once per window (`WINDOW_SECONDS`, 60 s) it sends up to
  `WINDOW_MAX_FRAMES` of them, evenly spread, to a Bedrock vision model. The model reports, for each
  guinea pig whose mark colour it can identify, how many frames it was still, at the feeder or the
  water bottle, how far from the others and how unusual it looks. Each one becomes a `BEHAVIOR`
  event of the backend contract (one per guinea pig per window).
- `POST /ai/audio` (16-bit WAV) is classified with a signal rule (loud and shrill = `DISTRESS`)
  and sent as an `AUDIO` event. Calm clips are reported once per window.
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
