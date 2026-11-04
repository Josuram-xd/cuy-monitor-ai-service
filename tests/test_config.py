import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings


def test_settings_load_from_environment(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "https://backend.example.test")
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("CAGE_ID", "cage-test")
    monkeypatch.setenv("WINDOW_SECONDS", "90")
    monkeypatch.setenv("MOCK_MODE", "false")

    settings = Settings(_env_file=None)

    assert settings.backend_url == "https://backend.example.test"
    assert settings.api_key == SecretStr("test-secret")
    assert settings.cage_id == "cage-test"
    assert settings.window_seconds == 90
    assert settings.mock_mode is False


def test_settings_reject_non_positive_window(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("WINDOW_SECONDS", "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_the_allowed_cages_default_to_the_single_cage():
    settings = Settings(_env_file=None, api_key="k", cage_id="cage-7")

    assert settings.allowed_cages == ["cage-7"]


def test_several_cages_are_a_comma_separated_list():
    settings = Settings(_env_file=None, api_key="k", cage_ids=" cage-1 ,cage-0a1b2c3d,, ")

    assert settings.allowed_cages == ["cage-1", "cage-0a1b2c3d"]


def test_a_settings_object_knows_which_cages_it_accepts():
    settings = Settings(_env_file=None, api_key="k", cage_ids="cage-1,cage-2")

    assert settings.accepts_cage("cage-2")
    assert not settings.accepts_cage("cage-9")
