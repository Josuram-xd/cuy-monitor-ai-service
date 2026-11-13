import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from app.api.ingestion import router as ingestion_router
from app.audio.model import load_audio_classifier
from app.audio.throttle import CalmAudioThrottle
from app.bedrock.runtime import BedrockRuntime
from app.behavior.classifier import load_behavior_classifier
from app.behavior.engine import TrainedBehaviorEngine
from app.config import Settings, get_settings
from app.messaging.backend_client import BackendEventClient
from app.mock.producer import run_mock_producer, run_scenario_producer
from app.testdata.scenarios import build_scenario, parse_colors
from app.vision.analysis import BedrockBehaviorAnalyzer
from app.vision.model_store import load_detector
from app.vision.pipeline import run_behavior_pipeline, run_trained_behavior_pipeline
from app.vision.window import FrameWindows

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = BackendEventClient(settings)
    app.state.backend_event_client = client
    worker = asyncio.create_task(client.run())
    # mock mode: fake events. Real mode: the trained chain when both models loaded, otherwise
    # Amazon Bedrock looks at the frames gathered each window.
    app.state.frame_windows = FrameWindows()
    app.state.simulations = {}  # cage id -> the running demo task (Simulación button)
    app.state.calm_audio_throttle = CalmAudioThrottle(settings.window_seconds)
    app.state.detector = load_detector(settings.detector_path, settings.detector_confidence)
    app.state.behavior_classifier = load_behavior_classifier(settings.behavior_model_path)
    app.state.audio_classifier = load_audio_classifier(
        settings.yamnet_path, settings.audio_model_path
    )
    app.state.behavior_engine = None
    if settings.mock_mode:
        producer = asyncio.create_task(_mock_producer(client, settings))
    elif app.state.detector is not None and app.state.behavior_classifier is not None:
        # both trained models are there: they replace Bedrock for the frames (audio is the same)
        engine = TrainedBehaviorEngine(
            app.state.detector,
            app.state.behavior_classifier,
            settings.zones_for,
            settings.window_seconds,
            settings.behavior_log_path,
        )
        app.state.behavior_engine = engine
        logger.info("Behavior engine: trained detector + tracker + Random Forest")
        producer = asyncio.create_task(run_trained_behavior_pipeline(client, engine, settings))
    else:
        logger.info("Behavior engine: Amazon Bedrock (%s)", settings.bedrock_model_id)
        analyzer = BedrockBehaviorAnalyzer(
            BedrockRuntime(settings.aws_region), settings.bedrock_model_id, settings.window_seconds
        )
        producer = asyncio.create_task(
            run_behavior_pipeline(client, analyzer, app.state.frame_windows, settings)
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


def _mock_producer(client: BackendEventClient, settings: Settings):
    if not settings.mock_scenario:
        return run_mock_producer(client, settings)
    scenario = build_scenario(
        settings.mock_scenario, parse_colors(settings.mock_colors), settings.window_seconds
    )
    logger.info("Mock mode replaying the %s scenario: %s", scenario.name, scenario.description)
    return run_scenario_producer(client, settings, scenario)


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
def health(request: Request) -> HealthResponse:
    settings = get_settings()
    # real mode: the vision model is Bedrock and the audio is a signal rule. The detector is also
    # there when the trained detector.onnx was loaded, in any mode.
    real = not settings.mock_mode
    onnx_detector = getattr(request.app.state, "detector", None) is not None
    trained_behavior = getattr(request.app.state, "behavior_classifier", None) is not None
    trained_audio = getattr(request.app.state, "audio_classifier", None) is not None
    return HealthResponse(
        models=ModelStatus(
            detector=real or onnx_detector,
            behavior=real or trained_behavior,
            audio=real or trained_audio,
        ),
        mock_mode=settings.mock_mode,
    )
