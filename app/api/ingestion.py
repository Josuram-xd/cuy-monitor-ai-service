import logging
import time
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from app.audio.classifier import UnsupportedAudio, classify_clip
from app.config import Settings, get_settings
from app.contracts import AudioEventPayload, EventEnvelope, EventType
from app.contracts.events import AudioLabel
from app.security import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai",
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
    if cage_id != settings.cage_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Unknown cage: {cage_id}")


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, detail="The file is too large")
    return data


@router.post("/frames", status_code=status.HTTP_202_ACCEPTED, response_model=AcceptedResponse)
async def submit_frame(
    request: Request,
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
    # the analysis happens once per window, over the frames gathered by then
    request.app.state.frame_window.add(frame)
    return AcceptedResponse(mock_mode=False)


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
        result = classify_clip(clip)
    except UnsupportedAudio as exception:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exception)) from None

    # every distress clip is reported; calm clips only once per window, or the backend would be
    # flooded with "all quiet" events
    state = request.app.state
    now = time.monotonic()
    if result.label is AudioLabel.NORMAL:
        if now - state.last_normal_audio_at < settings.window_seconds:
            return AcceptedResponse(mock_mode=False)
        state.last_normal_audio_at = now
    await state.backend_event_client.enqueue_event(
        EventEnvelope(
            event_id=uuid4(),
            type=EventType.AUDIO,
            cage_id=settings.cage_id,
            timestamp=datetime.now(UTC),
            source="ai-service",
            schema_version=1,
            payload=AudioEventPayload(
                label=result.label, probability=result.probability, duration_ms=result.duration_ms
            ),
        )
    )
    return AcceptedResponse(mock_mode=False)
