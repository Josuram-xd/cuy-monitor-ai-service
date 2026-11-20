import asyncio
import itertools
import logging
import random
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from app.audio.classifier import SIGNAL_RULE, AudioClassifier, UnsupportedAudio
from app.config import Settings, get_settings
from app.contracts import (
    AudioEventPayload,
    EventEnvelope,
    EventType,
    MarkColor,
    WeightEventPayload,
)
from app.contracts.events import AudioLabel
from app.security import require_api_key
from app.testdata.audio import ClipKind, synthesize_clip
from app.testdata.simulation import run_simulation
from app.testdata.video import (
    MAX_VIDEO_SIMULATIONS,
    ReportWindow,
    VideoClip,
    is_simulated,
    run_video_simulation,
    scripted_window,
    source_for,
)
from app.vision.frame_sink import frame_sink_for

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai/v1",
    dependencies=[Depends(require_api_key)],
)

MAX_FRAME_BYTES = 5 * 1024 * 1024
MAX_AUDIO_BYTES = 2 * 1024 * 1024


class AcceptedResponse(BaseModel):
    status: Literal["accepted"] = "accepted"
    mock_mode: bool = Field(default=True, serialization_alias="mockMode")


def _require_content_type(file: UploadFile, accepted_types: set[str]) -> None:
    if file.content_type not in accepted_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {file.content_type}",
        )


def _require_known_cage(cage_id: str, settings: Settings) -> None:
    if not settings.accepts_cage(cage_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Unknown cage: {cage_id}")


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail="The file is too large")
    return data


@router.post("/frames", status_code=status.HTTP_202_ACCEPTED, response_model=AcceptedResponse)
async def submit_frame(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    captured_at: Annotated[datetime, Form(alias="capturedAt")],
    cage_id: Annotated[str, Form(alias="cageId", min_length=1, max_length=50)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AcceptedResponse:
    _require_content_type(file, {"image/jpeg"})
    if settings.mock_mode:
        return AcceptedResponse()
    _require_known_cage(cage_id, settings)
    frame = await _read_limited(file, MAX_FRAME_BYTES)
    request.app.state.latest_frames[cage_id] = (captured_at, frame)
    # after the 202: the trained chain detects the frame, Bedrock buffers it for its window
    background_tasks.add_task(
        frame_sink_for(request.app.state).submit_frame, cage_id, frame, captured_at
    )
    return AcceptedResponse(mock_mode=False)


@router.get("/frames/latest", responses={404: {"description": "No frame of this cage yet"}})
def latest_frame(
    request: Request,
    cage_id: Annotated[str, Query(alias="cageId", min_length=1, max_length=50)],
) -> Response:
    latest = request.app.state.latest_frames.get(cage_id)
    if latest is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No frame yet")
    captured_at, frame = latest
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"X-Captured-At": captured_at.isoformat(), "Cache-Control": "no-store"},
    )


@router.post("/audio", status_code=status.HTTP_202_ACCEPTED, response_model=AcceptedResponse)
async def submit_audio(
    request: Request,
    file: Annotated[UploadFile, File()],
    captured_at: Annotated[datetime, Form(alias="capturedAt")],
    cage_id: Annotated[str, Form(alias="cageId", min_length=1, max_length=50)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AcceptedResponse:
    _require_content_type(file, {"audio/wav", "audio/x-wav", "audio/wave"})
    if settings.mock_mode:
        return AcceptedResponse()
    _require_known_cage(cage_id, settings)
    clip = await _read_limited(file, MAX_AUDIO_BYTES)
    try:
        await _classify_and_report(request.app.state, cage_id, clip, captured_at)
    except UnsupportedAudio as exception:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exception)) from None
    return AcceptedResponse(mock_mode=False)


async def _classify_and_report(
    state, cage_id: str, clip: bytes, captured_at: datetime | None = None
) -> None:
    classifier: AudioClassifier = getattr(state, "audio_classifier", None) or SIGNAL_RULE
    # trained or signal rule, both in a thread: inference never runs on the event loop
    result = await asyncio.to_thread(classifier.classify, clip)
    # every distress clip is reported; calm clips only once per window, or the backend would be
    # flooded with "all quiet" events
    if result.label is AudioLabel.NORMAL and not state.calm_audio_throttle.should_report(cage_id):
        return
    await state.backend_event_client.enqueue_event(
        EventEnvelope(
            event_id=uuid4(),
            type=EventType.AUDIO,
            cage_id=cage_id,
            timestamp=captured_at or datetime.now(UTC),
            source=source_for(cage_id),
            schema_version=1,
            payload=AudioEventPayload(
                label=result.label, probability=result.probability, duration_ms=result.duration_ms
            ),
        )
    )


class SimulationRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    cage_id: str = Field(min_length=1, max_length=50)
    # the marks of the cage's guinea pigs; the first one is drawn sick
    colors: list[MarkColor] = Field(min_length=1, max_length=len(MarkColor))


@router.post("/simulation", status_code=status.HTTP_202_ACCEPTED, response_model=AcceptedResponse)
async def start_simulation(
    request: Request,
    body: SimulationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AcceptedResponse:
    """The dashboard's demo, asked by the backend (which already checked who owns the cage, so
    CAGE_IDS does not apply). Drawn frames go into the real chain for 35 minutes, and one synthetic
    squeal goes through the audio classifier, so a cage alert shows up right away."""
    if body.cage_id in settings.physical_cages:
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="Physical cage is reserved for monitoring"
        )
    if settings.mock_mode:
        return AcceptedResponse()
    state = request.app.state
    cage_id = body.cage_id
    if is_simulated(cage_id):
        _start_video_simulation(state, cage_id)
        return AcceptedResponse(mock_mode=False)
    colors = list(dict.fromkeys(body.colors))  # one guinea pig per mark
    sink = frame_sink_for(state)

    async def submit(frame: bytes) -> None:
        await sink.submit_frame(cage_id, frame, datetime.now(UTC))

    running = state.simulations.get(cage_id)
    # pressing the button again while it runs does not start a second one
    if running is None or running.done():
        state.simulations[cage_id] = asyncio.create_task(run_simulation(cage_id, colors, submit))
    await _classify_and_report(state, cage_id, synthesize_clip(ClipKind.SQUEAL))
    return AcceptedResponse(mock_mode=False)


def _start_video_simulation(state, cage_id: str) -> None:
    """A sim- cage replays the prepared clip through the trained chain (app/testdata/video.py)."""
    clip: VideoClip | None = getattr(state, "sim_clip", None)
    if clip is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="No video clip")
    if getattr(state, "behavior_engine", None) is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Trained models not loaded")
    running = {c for c, task in state.simulations.items() if is_simulated(c) and not task.done()}
    if cage_id in running:
        return  # pressing the button again while it runs does not start a second one
    if len(running) >= MAX_VIDEO_SIMULATIONS:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Too many video simulations")
    info = {"startedAt": datetime.now(UTC), "framesSent": 0}
    state.simulation_info[cage_id] = info
    state.simulations[cage_id] = asyncio.create_task(_run_video(state, cage_id, clip, info))


async def _run_video(state, cage_id: str, clip: VideoClip, info: dict) -> None:
    sink = frame_sink_for(state)

    async def submit(frame: bytes) -> None:
        now = datetime.now(UTC)
        state.latest_frames[cage_id] = (now, frame)  # the dashboard's live picture
        await sink.submit_frame(cage_id, frame, now)

    def progress(sent: int) -> None:
        info["framesSent"] = sent

    try:
        await run_video_simulation(
            cage_id, clip, submit, _video_window_reporter(state, cage_id, clip), progress
        )
    finally:
        state.simulation_info.pop(cage_id, None)


def _video_window_reporter(state, cage_id: str, clip: VideoClip) -> ReportWindow:
    """Each window of frames: the scripted weight (WEIGHT event) and one audio clip, classified
    by the real audio classifier."""
    windows = itertools.count()
    rng = random.Random()

    async def report_window() -> None:
        grams, kind = scripted_window(next(windows), clip.base_weight_grams, rng)
        await state.backend_event_client.enqueue_event(
            EventEnvelope(
                event_id=uuid4(),
                type=EventType.WEIGHT,
                cage_id=cage_id,
                timestamp=datetime.now(UTC),
                source=source_for(cage_id),
                schema_version=1,
                payload=WeightEventPayload(grams=grams, stable=True),
            )
        )
        await _classify_and_report(state, cage_id, synthesize_clip(kind))

    return report_window


@router.get("/simulation/clip")
def simulation_clip(request: Request) -> dict:
    clip: VideoClip | None = getattr(request.app.state, "sim_clip", None)
    if clip is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="No video clip")
    return {
        "colors": [color.value for color in clip.colors],
        "credit": clip.credit,
        "frameCount": len(clip.frames),
    }


@router.get("/simulation/{cage_id}")
def simulation_status(request: Request, cage_id: str) -> dict:
    state = request.app.state
    info = state.simulation_info.get(cage_id)
    clip: VideoClip | None = getattr(state, "sim_clip", None) if is_simulated(cage_id) else None
    return {
        "running": info is not None,
        "startedAt": info["startedAt"].isoformat() if info else None,
        "framesSent": info["framesSent"] if info else 0,
        "frameCount": len(clip.frames) if clip else 0,
        "credit": clip.credit if clip else None,
    }


@router.delete("/simulation/{cage_id}", status_code=status.HTTP_204_NO_CONTENT)
def stop_simulation(request: Request, cage_id: str) -> None:
    task = request.app.state.simulations.pop(cage_id, None)
    if task is not None:
        task.cancel()
    request.app.state.simulation_info.pop(cage_id, None)
