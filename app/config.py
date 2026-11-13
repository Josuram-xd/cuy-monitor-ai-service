from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.vision.zones import NO_ZONES, CageZones, parse_cage_zones


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
    # mock mode only: replay a test scenario (healthy, lethargic, distress, recovery) instead of
    # random events, for the guinea pigs of these mark colours. Empty = random events as before.
    mock_scenario: str = ""
    mock_colors: str = "RED,BLUE,GREEN,YELLOW"

    # real mode: Amazon Bedrock looks at the frames. On the EC2 the credentials come from the
    # instance role, so there is no key to configure.
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    # frames sent to the model per window (evenly spread): more is better but costs more
    window_max_frames: int = Field(default=6, ge=1, le=20)

    # the trained detector (Task 7). Missing file = no detector, never an error: see /ai/health
    detector_path: Path = Path("models/detector.onnx")
    detector_confidence: float = Field(default=0.35, gt=0, lt=1)

    # the behavior classifier (Task 10). With it and the detector, the frames go through the
    # trained chain (detector -> tracker -> 60 s window -> Random Forest) instead of Bedrock.
    behavior_model_path: Path = Path("models/behavior_rf.joblib")
    # feeder and waterer polygons per cage, in fractions of the frame (see app/vision/zones.py)
    cage_zones: str = ""
    # optional: append the features of every window to this CSV, to label real windows later and
    # retrain the classifier with them (training/train_behavior.py --data)
    behavior_log_path: Path | None = None

    # the trained audio classifier (Task 11): YAMNet embeddings + SVM/KNN. Without both files the
    # audio keeps the signal rule of app/audio/classifier.py
    yamnet_path: Path = Path("models/yamnet.onnx")
    audio_model_path: Path = Path("models/audio_clf.joblib")

    @field_validator("cage_zones")
    @classmethod
    def zones_are_valid(cls, value: str) -> str:
        parse_cage_zones(value)  # a typo in the zones fails at start, not in the middle of a window
        return value

    @field_validator("behavior_log_path", mode="before")
    @classmethod
    def empty_log_path_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    def zones_for(self, cage_id: str) -> CageZones:
        return parse_cage_zones(self.cage_zones).get(cage_id, NO_ZONES)

    @property
    def allowed_cages(self) -> list[str]:
        cages = [cage.strip() for cage in self.cage_ids.split(",") if cage.strip()]
        return cages or [self.cage_id]

    def accepts_cage(self, cage_id: str) -> bool:
        return cage_id in self.allowed_cages


@lru_cache
def get_settings() -> Settings:
    return Settings()
