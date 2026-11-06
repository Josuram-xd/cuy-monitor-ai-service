"""Task 11: YAMNet embeddings (ONNX) + SVM/KNN, with fake sessions and a spectral fake embedder."""

import io
import json
import wave
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.audio.embeddings import EMBEDDING_SIZE, MIN_SAMPLES, YamnetEmbedder, load_embedder
from app.audio.model import (
    BUNDLE_FORMAT,
    EMBEDDING,
    TrainedAudioClassifier,
    load_audio_classifier,
)
from app.audio.wav import UnsupportedAudio, read_wav, resample
from app.config import get_settings
from app.contracts.events import AudioLabel
from app.testdata.audio import ClipKind, synthesize_clip
from training.train_audio import (
    main as train_main,
)
from training.train_audio import (
    read_recordings,
    synthetic_clips,
    train,
    windows,
)


def _wav(samples: np.ndarray, rate: int = 16_000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(channels)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


class SpectralEmbedder:
    """Stands in for YAMNet: 1024 band energies of the spectrum (enough to tell a squeal apart)."""

    def embed_samples(self, samples, rate):
        audio = resample(np.asarray(samples, dtype=np.float64), rate, 16_000)
        spectrum = np.abs(np.fft.rfft(audio, n=2 * EMBEDDING_SIZE))[:EMBEDDING_SIZE]
        return np.log1p(spectrum).astype(np.float32)


class _Session:
    """A fake onnxruntime session with YAMNet's interface: waveform in, (frames, 1024) out."""

    def __init__(self, outputs=("embeddings",), shape_ok=True):
        self.inputs = []
        self._outputs = outputs
        self._shape_ok = shape_ok

    def get_inputs(self):
        return [SimpleNamespace(name="waveform")]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in self._outputs]

    def run(self, names, feed):
        waveform = feed["waveform"]
        self.inputs.append(waveform)
        frames = max(1, (waveform.size - MIN_SAMPLES) // 7680 + 1)
        width = EMBEDDING_SIZE if self._shape_ok else 521
        rows = np.tile(np.arange(width, dtype=np.float32), (frames, 1)) * float(
            np.abs(waveform).mean()
        )
        return [np.zeros((frames, 521), np.float32), rows]


# ---- reading and resampling


def test_a_stereo_44k_clip_is_read_as_mono_and_resampled_to_16k():
    left = np.sin(np.arange(44_100) * 2 * np.pi * 440 / 44_100) * 0.5
    stereo = np.column_stack([left, left]).ravel()

    samples, rate = read_wav(_wav(stereo, 44_100, channels=2))

    assert rate == 44_100 and samples.size == 44_100
    assert resample(samples, rate, 16_000).size == 16_000


def test_a_clip_that_is_not_16_bit_pcm_is_refused():
    with pytest.raises(UnsupportedAudio):
        read_wav(b"not a wav")


# ---- Task 11.2: YAMNet embeddings


def test_the_embedder_sends_a_16k_float_waveform_and_averages_the_frames():
    session = _Session()
    embedder = YamnetEmbedder(session)

    vector = embedder.embed(_wav(np.full(32_000, 0.25), 32_000))

    [waveform] = session.inputs
    assert waveform.dtype == np.float32 and waveform.size == 16_000 + 0  # 1 s at 16 kHz
    assert vector.shape == (EMBEDDING_SIZE,)


def test_a_short_clip_is_padded_to_one_yamnet_window():
    session = _Session()
    YamnetEmbedder(session).embed(_wav(np.full(4_000, 0.1)))

    assert session.inputs[0].size == MIN_SAMPLES


def test_an_export_without_the_embeddings_output_is_refused():
    with pytest.raises(ValueError, match="1024"):
        YamnetEmbedder(_Session(outputs=("a", "b"), shape_ok=False)).embed(_wav(np.zeros(16_000)))


def test_a_missing_or_broken_yamnet_is_never_fatal(tmp_path):
    assert load_embedder(tmp_path / "missing.onnx") is None
    broken = tmp_path / "yamnet.onnx"
    broken.write_bytes(b"not onnx")
    assert load_embedder(broken) is None


# ---- Task 11.3: training


def test_recordings_are_cut_into_096_s_windows():
    # 3 windows of 0.96 s and a 0.12 s piece, too short to keep
    assert len(list(windows(np.zeros(16_000 * 3), 16_000))) == 3
    # 1 s at 44.1 kHz is resampled first: one window
    assert len(list(windows(np.zeros(44_100), 44_100))) == 1
    # a last piece of at least half a window is kept
    assert len(list(windows(np.zeros(16_000 + 8_000), 16_000))) == 2
    assert len(list(windows(np.zeros(16_000 // 4), 16_000))) == 0


def test_recordings_come_from_the_normal_and_distress_folders(tmp_path):
    (tmp_path / "normal").mkdir()
    (tmp_path / "distress").mkdir()
    (tmp_path / "normal" / "calm.wav").write_bytes(_wav(np.zeros(16_000 * 2)))
    (tmp_path / "distress" / "squeal.wav").write_bytes(synthesize_clip(ClipKind.SQUEAL))
    (tmp_path / "distress" / "broken.wav").write_bytes(b"nope")

    clips, labels, skipped = read_recordings(tmp_path)

    assert labels == ["NORMAL", "NORMAL", "DISTRESS"]
    assert skipped == 1


def test_svm_or_knn_learn_to_tell_a_squeal_from_calm_sounds():
    clips, labels = synthetic_clips(80)
    features = np.stack([SpectralEmbedder().embed_samples(c, 16_000) for c in clips])

    model, best, metrics = train(features, labels)

    assert best in {"svm", "knn"}
    assert metrics["svm"]["f1"] > 0.9 and "knn" in metrics
    assert sorted(model.classes_) == ["DISTRESS", "NORMAL"]


def test_training_needs_both_classes():
    with pytest.raises(ValueError):
        train(np.zeros((6, 4)), ["NORMAL"] * 6)


def test_the_script_writes_a_bundle_the_service_loads(tmp_path):
    out = tmp_path / "audio_clf.joblib"

    assert train_main(["--synthetic", "60", "--out", str(out)], embedder=SpectralEmbedder()) == 0

    report = json.loads(out.with_suffix(".json").read_text())
    assert report["data"] == "synthetic" and report["embedding"] == EMBEDDING
    assert len(report["sha256"]) == 64
    classifier = TrainedAudioClassifier.load(SpectralEmbedder(), out)
    squeal = classifier.classify(synthesize_clip(ClipKind.SQUEAL, seed=999))
    calm = classifier.classify(synthesize_clip(ClipKind.RUSTLE, seed=999))
    assert squeal.label is AudioLabel.DISTRESS and squeal.probability >= 0.5
    assert calm.label is AudioLabel.NORMAL and calm.probability >= 0.5
    assert squeal.duration_ms == 960


def test_without_yamnet_the_script_says_how_to_get_it(tmp_path, capsys):
    assert train_main(["--yamnet", str(tmp_path / "none.onnx"), "--synthetic", "10"]) == 1
    assert "export_yamnet" in capsys.readouterr().err


# ---- Task 11.4: the trained classifier in the service


def test_a_bundle_of_another_format_embedding_or_classes_is_refused(tmp_path):
    good_model = SimpleNamespace(classes_=["DISTRESS", "NORMAL"])
    for bundle in (
        {"format": BUNDLE_FORMAT + 1, "embedding": EMBEDDING, "model": good_model},
        {"format": BUNDLE_FORMAT, "embedding": "other", "model": good_model},
        {
            "format": BUNDLE_FORMAT,
            "embedding": EMBEDDING,
            "model": SimpleNamespace(classes_=["CALM", "LOUD"]),
        },
    ):
        path = tmp_path / "bad.joblib"
        joblib.dump(bundle, path)
        with pytest.raises(ValueError):
            TrainedAudioClassifier.load(SpectralEmbedder(), path)


def test_missing_audio_models_keep_the_signal_rule(tmp_path):
    assert load_audio_classifier(tmp_path / "yamnet.onnx", tmp_path / "audio.joblib") is None
    model = tmp_path / "audio.joblib"
    model.write_bytes(b"x")
    assert load_audio_classifier(tmp_path / "missing.onnx", model) is None


class _Classifier:
    def __init__(self):
        self.clips = []

    def classify(self, clip):
        from app.audio.classifier import AudioResult

        self.clips.append(clip)
        return AudioResult(AudioLabel.DISTRESS, 0.93, 960)


class _Client:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


def test_with_the_trained_models_audio_goes_through_them(monkeypatch):
    classifier = _Classifier()
    monkeypatch.setenv("API_KEY", "test-secret")
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setattr(main_module, "load_audio_classifier", lambda *_: classifier)
    # the app object outlives this test: what its startup stores must not leak into the next ones
    monkeypatch.setattr(main_module.app.state, "audio_classifier", None, raising=False)
    get_settings.cache_clear()
    with TestClient(main_module.app) as client:
        main_module.app.state.backend_event_client = backend = _Client()
        clip = synthesize_clip(ClipKind.RUSTLE)

        response = client.post(
            "/ai/audio",
            headers={"X-API-Key": "test-secret"},
            data={"capturedAt": "2026-11-05T12:00:00Z", "cageId": "cage-1"},
            files={"file": ("c.wav", clip, "audio/wav")},
        )

        assert response.status_code == 202
        assert classifier.clips == [clip]
        [event] = backend.events
        assert event.payload.label is AudioLabel.DISTRESS and event.payload.probability == 0.93
        assert client.get("/ai/health").json()["models"]["audio"] is True
    get_settings.cache_clear()
