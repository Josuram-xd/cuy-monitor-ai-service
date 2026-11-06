import math
from dataclasses import dataclass

import numpy as np

from app.audio.wav import UnsupportedAudio, read_wav
from app.contracts.events import AudioLabel

__all__ = ["AudioResult", "UnsupportedAudio", "classify_clip"]

# Distress calls (squeals) are loud and shrill: a lot of the energy sits above 2 kHz.
SHRILL_FROM_HZ = 2000
SHRILL_TO_HZ = 8000
QUIET_DBFS = -40.0
LOUD_DBFS = -10.0
DISTRESS_THRESHOLD = 0.5


@dataclass(frozen=True)
class AudioResult:
    label: AudioLabel
    probability: float
    duration_ms: int


def classify_clip(wav_bytes: bytes) -> AudioResult:
    """Loudness x shrillness. A signal-processing rule, not a trained model: it only has to tell
    a loud squeal from quiet rustling until real recordings exist to train something better."""
    samples, rate = read_wav(wav_bytes)
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
