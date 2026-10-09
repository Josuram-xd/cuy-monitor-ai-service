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
