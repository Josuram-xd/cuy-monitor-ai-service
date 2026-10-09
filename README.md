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

To replay a local recording at a maximum of one frame per second, set `API_KEY` in the
environment and run:

```powershell
$env:API_KEY = "<local-api-key>"
uv run python dev/simulator.py .\recording.mp4 --ai-url http://localhost:8000 --cage-id cage-1
```

The simulator samples frames according to the source video's frame rate. Use `--source-fps`
if the video does not provide valid frame-rate metadata.
