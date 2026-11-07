"""Task 14.1: the paths where the models do load, and the build step that downloads them."""

import hashlib
from types import SimpleNamespace

import joblib
import pytest
from fastapi.testclient import TestClient

import app.audio.model as audio_model
import app.main as main_module
import app.vision.model_store as model_store
from app.audio.model import BUNDLE_FORMAT, EMBEDDING, TrainedAudioClassifier
from app.config import get_settings
from scripts import fetch_models

MODEL_VARS = [name for url, sha, path, _, _ in fetch_models.MODELS for name in (url, sha, path)]


@pytest.fixture
def no_model_vars(monkeypatch):
    for name in MODEL_VARS:
        monkeypatch.delenv(name, raising=False)


def test_without_urls_the_image_is_built_without_models(
    no_model_vars, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)

    assert fetch_models.main() == 0

    out = capsys.readouterr().out
    assert "without the detector" in out and "without the audio classifier" in out
    assert out.count("No ") == len(fetch_models.MODELS)
    assert (tmp_path / "models").is_dir()


def test_each_model_is_downloaded_where_its_path_says(no_model_vars, tmp_path, monkeypatch):
    source = tmp_path / "behavior_rf.joblib"
    source.write_bytes(b"bundle")
    monkeypatch.setenv("BEHAVIOR_MODEL_URL", source.as_uri())
    monkeypatch.setenv("BEHAVIOR_MODEL_SHA256", hashlib.sha256(b"bundle").hexdigest())
    monkeypatch.setenv("BEHAVIOR_MODEL_PATH", str(tmp_path / "models" / "rf.joblib"))

    assert fetch_models.main() == 0
    assert (tmp_path / "models" / "rf.joblib").read_bytes() == b"bundle"


def test_a_url_without_its_sha256_fails_the_build(no_model_vars, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AUDIO_MODEL_URL", (tmp_path / "x").as_uri())
    monkeypatch.setenv("AUDIO_MODEL_PATH", str(tmp_path / "audio.joblib"))

    assert fetch_models.main() == 1
    assert "AUDIO_MODEL_SHA256 is required" in capsys.readouterr().err


def test_a_detector_that_loads_is_returned(tmp_path, monkeypatch):
    path = tmp_path / "detector.onnx"
    path.write_bytes(b"onnx")
    loaded = object()
    monkeypatch.setattr(model_store.OnnxDetector, "load", classmethod(lambda cls, p, **o: loaded))

    assert model_store.load_detector(path, 0.35) is loaded


class _Embedder:
    def embed_samples(self, samples, rate):
        return [0.0]


def test_yamnet_and_a_good_bundle_give_the_trained_audio_classifier(tmp_path, monkeypatch):
    model_path = tmp_path / "audio_clf.joblib"
    joblib.dump(
        {
            "format": BUNDLE_FORMAT,
            "embedding": EMBEDDING,
            "model": SimpleNamespace(classes_=["DISTRESS", "NORMAL"]),
        },
        model_path,
    )
    monkeypatch.setattr(audio_model, "load_embedder", lambda path: _Embedder())

    classifier = audio_model.load_audio_classifier(tmp_path / "yamnet.onnx", model_path)

    assert isinstance(classifier, TrainedAudioClassifier)


def test_yamnet_with_a_broken_bundle_keeps_the_signal_rule(tmp_path, monkeypatch):
    model_path = tmp_path / "audio_clf.joblib"
    model_path.write_bytes(b"not joblib")
    monkeypatch.setattr(audio_model, "load_embedder", lambda path: _Embedder())

    assert audio_model.load_audio_classifier(tmp_path / "yamnet.onnx", model_path) is None


def test_mock_mode_can_replay_a_scenario(monkeypatch):
    started = []

    async def fake_scenario_producer(client, settings, scenario):
        started.append(scenario.name)

    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("MOCK_SCENARIO", "lethargic")
    monkeypatch.setattr(main_module, "run_scenario_producer", fake_scenario_producer)
    get_settings.cache_clear()
    with TestClient(main_module.app):
        pass
    get_settings.cache_clear()

    assert started == ["lethargic"]
