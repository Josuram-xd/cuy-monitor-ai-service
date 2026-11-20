"""Configuration of the edge agent, from edge_agent/.env (template: edge_agent/config.example.env).

The API key is never in the code: it is the same API_KEY the AI service and the backend use.
"""

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).with_name(".env")


class EdgeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE, env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # IP Webcam on the A12, e.g. http://192.168.1.50:8080 (shown on the phone screen)
    a12_url: str = ""
    # RTSP stream of a TP-Link Tapo camera, e.g. rtsp://user:pass@192.168.1.60:554/stream1.
    # When set, frames and audio come from it instead of the phone (needs ffmpeg on PATH).
    camera_rtsp_url: str = ""
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
    def no_trailing_slash(cls, value: str, info: ValidationInfo) -> str:
        value = value.strip().rstrip("/")
        if info.field_name == "a12_url" and not value:
            return value  # optional: an RTSP camera does not use the phone
        if not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value

    @field_validator("camera_rtsp_url")
    @classmethod
    def rtsp_scheme(cls, value: str) -> str:
        value = value.strip()
        if value and not value.startswith("rtsp://"):
            raise ValueError("must start with rtsp://")
        return value

    @model_validator(mode="after")
    def needs_a_camera(self):
        if not (self.a12_url or self.camera_rtsp_url):
            raise ValueError("set A12_URL (IP Webcam) or CAMERA_RTSP_URL (Tapo)")
        return self


def mask_url(url: str) -> str:
    """The URL with its password replaced by ***, for logs: rtsp://user:***@192.168.1.60:554/..."""
    parts = urlsplit(url)
    userinfo, at, hostport = parts.netloc.rpartition("@")
    if not at or ":" not in userinfo:
        return url
    user = userinfo.partition(":")[0]
    return urlunsplit(parts._replace(netloc=f"{user}:***@{hostport}"))
