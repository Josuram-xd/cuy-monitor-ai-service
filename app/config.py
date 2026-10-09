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


@lru_cache
def get_settings() -> Settings:
    return Settings()
