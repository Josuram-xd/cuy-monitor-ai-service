import io
import wave

import numpy as np
import pytest

from app.audio.classifier import UnsupportedAudio, classify_clip
from app.contracts.events import AudioLabel

RATE = 16_000


def _wav(samples: np.ndarray, rate: int = RATE, width: int = 2, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(channels)
        clip.setsampwidth(width)
        clip.setframerate(rate)
        clip.writeframes(samples.tobytes())
    return buffer.getvalue()


def _tone(frequency: float, amplitude: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (amplitude * 32767 * np.sin(2 * np.pi * frequency * t)).astype("<i2")


def test_a_loud_shrill_squeal_is_distress():
    result = classify_clip(_wav(_tone(4000, 0.7)))

    assert result.label is AudioLabel.DISTRESS
    assert result.probability >= 0.5
    assert result.duration_ms == 1000


def test_a_loud_low_rumble_is_not_distress():
    assert classify_clip(_wav(_tone(200, 0.7))).label is AudioLabel.NORMAL


def test_a_quiet_high_tone_is_not_distress():
    assert classify_clip(_wav(_tone(4000, 0.005))).label is AudioLabel.NORMAL


def test_silence_is_normal_with_high_probability():
    result = classify_clip(_wav(np.zeros(RATE, dtype="<i2")))

    assert result.label is AudioLabel.NORMAL
    assert result.probability == 1.0


def test_the_duration_comes_from_the_clip():
    assert classify_clip(_wav(_tone(4000, 0.7, seconds=0.5))).duration_ms == 500


def test_stereo_is_mixed_down():
    stereo = np.repeat(_tone(4000, 0.7), 2)

    assert classify_clip(_wav(stereo, channels=2)).label is AudioLabel.DISTRESS


@pytest.mark.parametrize(
    "clip",
    [b"", b"mock media", _wav(np.zeros(100, dtype="u1"), width=1), _wav(np.zeros(0, dtype="<i2"))],
)
def test_unsupported_audio_is_refused(clip):
    with pytest.raises(UnsupportedAudio):
        classify_clip(clip)
