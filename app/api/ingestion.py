from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.security import require_api_key

router = APIRouter(
    prefix="/ai",
    dependencies=[Depends(require_api_key)],
)


class AcceptedResponse(BaseModel):
    status: Literal["accepted"] = "accepted"
    mock_mode: Literal[True] = Field(default=True, serialization_alias="mockMode")


def _require_mock_mode(settings: Settings) -> None:
    if not settings.mock_mode:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Frame and audio inference are not available outside mock mode",
        )


def _require_content_type(file: UploadFile, accepted_types: set[str]) -> None:
    if file.content_type not in accepted_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {file.content_type}",
        )


@router.post("/frames", status_code=status.HTTP_202_ACCEPTED, response_model=AcceptedResponse)
async def submit_frame(
    file: Annotated[UploadFile, File()],
    captured_at: Annotated[datetime, Form(alias="capturedAt")],
    cage_id: Annotated[str, Form(alias="cageId", min_length=1, max_length=50)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AcceptedResponse:
    _require_mock_mode(settings)
    _require_content_type(file, {"image/jpeg"})
    return AcceptedResponse()


@router.post("/audio", status_code=status.HTTP_202_ACCEPTED, response_model=AcceptedResponse)
async def submit_audio(
    file: Annotated[UploadFile, File()],
    captured_at: Annotated[datetime, Form(alias="capturedAt")],
    cage_id: Annotated[str, Form(alias="cageId", min_length=1, max_length=50)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AcceptedResponse:
    _require_mock_mode(settings)
    _require_content_type(file, {"audio/wav", "audio/x-wav", "audio/wave"})
    return AcceptedResponse()
