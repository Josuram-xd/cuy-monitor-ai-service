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
