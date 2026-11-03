from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    backend_url: str = "http://backend:8080"
    api_key: SecretStr
    # the cage used when nothing else says which one (and the only one by default)
    cage_id: str = "cage-1"
    # every cage this service accepts frames and audio for, comma separated
    # ("cage-1,cage-0a1b2c3d"). Empty = only cage_id. The backend has the final word: it rejects
    # cages that don't exist.
    cage_ids: str = ""
    window_seconds: int = Field(default=60, ge=1)
    mock_mode: bool = True

    # real mode: Amazon Bedrock looks at the frames. On the EC2 the credentials come from the
    # instance role, so there is no key to configure.
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "us.amazon.nova-2-lite-v1:0"
    # frames sent to the model per window (evenly spread): more is better but costs more
    window_max_frames: int = Field(default=6, ge=1, le=20)

    @property
    def allowed_cages(self) -> list[str]:
        cages = [cage.strip() for cage in self.cage_ids.split(",") if cage.strip()]
        return cages or [self.cage_id]


@lru_cache
def get_settings() -> Settings:
    return Settings()
