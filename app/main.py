from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import get_settings


class ModelStatus(BaseModel):
    detector: bool = False
    behavior: bool = False
    audio: bool = False


class HealthResponse(BaseModel):
    status: Literal["UP"] = "UP"
    models: ModelStatus
    mock_mode: bool = Field(serialization_alias="mockMode")


app = FastAPI(title="Cuy Monitor AI Service")


@app.get("/ai/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(models=ModelStatus(), mock_mode=settings.mock_mode)
