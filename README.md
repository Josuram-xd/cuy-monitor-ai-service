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

Runtime configuration and API routes are introduced in the following project tasks.
