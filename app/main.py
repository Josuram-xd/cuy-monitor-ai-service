import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.api.ingestion import router as ingestion_router
from app.config import get_settings
from app.messaging.backend_client import BackendEventClient

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = BackendEventClient(get_settings())
    app.state.backend_event_client = client
    worker = asyncio.create_task(client.run())
    try:
        yield
    finally:
        if client.pending_count:
            logger.warning("Stopping with %s backend events still buffered", client.pending_count)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        await client.close()


class ModelStatus(BaseModel):
    detector: bool = False
    behavior: bool = False
    audio: bool = False


class HealthResponse(BaseModel):
    status: Literal["UP"] = "UP"
    models: ModelStatus
    mock_mode: bool = Field(serialization_alias="mockMode")


app = FastAPI(title="Cuy Monitor AI Service", lifespan=lifespan)
app.include_router(ingestion_router)


@app.get("/ai/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(models=ModelStatus(), mock_mode=settings.mock_mode)
