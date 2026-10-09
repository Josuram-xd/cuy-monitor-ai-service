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
    cage_id: str = "cage-1"
    window_seconds: int = Field(default=60, ge=1)
    mock_mode: bool = True

    # real mode: Amazon Bedrock looks at the frames. On the EC2 the credentials come from the
    # instance role, so there is no key to configure.
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "us.amazon.nova-2-lite-v1:0"
    # frames sent to the model per window (evenly spread): more is better but costs more
    window_max_frames: int = Field(default=6, ge=1, le=20)


@lru_cache
def get_settings() -> Settings:
    return Settings()
