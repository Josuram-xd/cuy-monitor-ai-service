import io
import math
import wave
from dataclasses import dataclass

import numpy as np

from app.contracts.events import AudioLabel

# Distress calls (squeals) are loud and shrill: a lot of the energy sits above 2 kHz.
SHRILL_FROM_HZ = 2000
SHRILL_TO_HZ = 8000
QUIET_DBFS = -40.0
LOUD_DBFS = -10.0
DISTRESS_THRESHOLD = 0.5


class UnsupportedAudio(ValueError):
    """Not a 16-bit PCM WAV, or empty."""


@dataclass(frozen=True)
class AudioResult:
    label: AudioLabel
    probability: float
    duration_ms: int


def _samples(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(wav_bytes)) as clip:
            if clip.getsampwidth() != 2:
                raise UnsupportedAudio("only 16-bit PCM WAV is supported")
            rate = clip.getframerate()
            channels = clip.getnchannels()
            raw = clip.readframes(clip.getnframes())
    except (wave.Error, EOFError) as exception:
        raise UnsupportedAudio("not a readable WAV file") from exception
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        data = data[: len(data) - len(data) % channels].reshape(-1, channels).mean(axis=1)
    if data.size == 0:
        raise UnsupportedAudio("the clip is empty")
    return data, rate


def classify_clip(wav_bytes: bytes) -> AudioResult:
    """Loudness x shrillness. A signal-processing rule, not a trained model: it only has to tell
    a loud squeal from quiet rustling until real recordings exist to train something better."""
    samples, rate = _samples(wav_bytes)
    rms = math.sqrt(float(np.mean(samples**2)))
    dbfs = 20 * math.log10(max(rms, 1e-9))
    loud = min(1.0, max(0.0, (dbfs - QUIET_DBFS) / (LOUD_DBFS - QUIET_DBFS)))

    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size))) ** 2
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / rate)
    total = float(spectrum.sum())
    shrill_energy = float(
        spectrum[(frequencies >= SHRILL_FROM_HZ) & (frequencies <= SHRILL_TO_HZ)].sum()
    )
    shrill = min(1.0, max(0.0, (shrill_energy / total - 0.25) / 0.5)) if total > 0 else 0.0

    distress = loud * shrill
    duration_ms = max(1, round(1000 * samples.size / rate))
    if distress >= DISTRESS_THRESHOLD:
        return AudioResult(AudioLabel.DISTRESS, round(distress, 2), duration_ms)
    return AudioResult(AudioLabel.NORMAL, round(1 - distress, 2), duration_ms)
