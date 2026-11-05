import io
import wave
from enum import StrEnum

import numpy as np

from app.contracts.events import AudioLabel

SAMPLE_RATE = 16_000


class ClipKind(StrEnum):
    SILENCE = "silence"
    RUSTLE = "rustle"
    CHIRP = "chirp"
    SQUEAL = "squeal"


# what the loudness x shrillness rule must answer for each kind
EXPECTED_LABEL: dict[ClipKind, AudioLabel] = {
    ClipKind.SILENCE: AudioLabel.NORMAL,
    ClipKind.RUSTLE: AudioLabel.NORMAL,
    ClipKind.CHIRP: AudioLabel.NORMAL,
    ClipKind.SQUEAL: AudioLabel.DISTRESS,
}


def _silence(time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0, 0.0005, time.size)


def _rustle(time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # bedding moving: quiet noise with its energy low in the spectrum
    noise = rng.normal(0, 0.02, time.size)
    return np.convolve(noise, np.ones(32) / 32, mode="same")


def _chirp(time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # a calm "wheek": a soft tone around 600 Hz in short bursts
    bursts = (np.sin(2 * np.pi * 4 * time) > 0).astype(float)
    return 0.05 * np.sin(2 * np.pi * 600 * time) * bursts + rng.normal(0, 0.002, time.size)


def _squeal(time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # a distress call: loud, shrill, with harmonics between 3 and 7 kHz
    tone = sum(np.sin(2 * np.pi * f * time) / (i + 1) for i, f in enumerate((3500, 5200, 6900)))
    return 0.6 * tone / 1.8 + rng.normal(0, 0.01, time.size)


_GENERATORS = {
    ClipKind.SILENCE: _silence,
    ClipKind.RUSTLE: _rustle,
    ClipKind.CHIRP: _chirp,
    ClipKind.SQUEAL: _squeal,
}


def synthesize_clip(kind: ClipKind, seconds: float = 0.96, seed: int = 7) -> bytes:
    """A mono 16-bit PCM WAV of the given kind. Same kind and seed, same bytes."""
    time = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    signal = _GENERATORS[kind](time, np.random.default_rng(seed))
    pcm = (np.clip(signal, -1, 1) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(SAMPLE_RATE)
        clip.writeframes(pcm.tobytes())
    return buffer.getvalue()
