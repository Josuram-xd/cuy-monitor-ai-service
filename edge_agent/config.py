"""Configuration of the edge agent, from edge_agent/.env (template: edge_agent/config.example.env).

The API key is never in the code: it is the same API_KEY the AI service and the backend use.
"""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).with_name(".env")


class EdgeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE, env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # IP Webcam on the A12, e.g. http://192.168.1.50:8080 (shown on the phone screen)
    a12_url: str
    # the public AI service, e.g. https://cuymonitor.duckdns.org (the agent adds /ai/...)
    ai_url: str
    api_key: SecretStr
    # the cage code shown in the dashboard ("Mis jaulas")
    cage_id: str = Field(min_length=1, max_length=50)
    # frames per second sent to the service: 1-2 is enough for behavior and keeps the upload small
    fps: float = Field(default=1.0, gt=0, le=2)
    audio_enabled: bool = True
    # length of each audio clip sent to /ai/v1/audio
    audio_clip_seconds: float = Field(default=1.0, ge=0.5, le=5)
    request_timeout_seconds: float = Field(default=10.0, gt=0)

    @field_validator("a12_url", "ai_url")
    @classmethod
    def no_trailing_slash(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value
