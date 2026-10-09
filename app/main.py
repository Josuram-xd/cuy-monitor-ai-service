import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.api.ingestion import router as ingestion_router
from app.bedrock.runtime import BedrockRuntime
from app.config import get_settings
from app.messaging.backend_client import BackendEventClient
from app.mock.producer import run_mock_producer
from app.vision.analysis import BedrockBehaviorAnalyzer
from app.vision.pipeline import run_behavior_pipeline
from app.vision.window import FrameWindow

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = BackendEventClient(settings)
    app.state.backend_event_client = client
    worker = asyncio.create_task(client.run())
    # mock mode: fake events. Real mode: Amazon Bedrock looks at the frames gathered each window.
    app.state.frame_window = FrameWindow()
    app.state.last_normal_audio_at = -settings.window_seconds - time.monotonic()
    if settings.mock_mode:
        producer = asyncio.create_task(run_mock_producer(client, settings))
    else:
        analyzer = BedrockBehaviorAnalyzer(
            BedrockRuntime(settings.aws_region), settings.bedrock_model_id, settings.window_seconds
        )
        producer = asyncio.create_task(
            run_behavior_pipeline(client, analyzer, app.state.frame_window, settings)
        )
    try:
        yield
    finally:
        if producer:
            producer.cancel()
            try:
                await producer
            except asyncio.CancelledError:
                pass
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
    # real mode: the vision model is Bedrock and the audio is a signal rule; none is a local file
    real = not settings.mock_mode
    return HealthResponse(
        models=ModelStatus(detector=real, behavior=real, audio=real), mock_mode=settings.mock_mode
    )
