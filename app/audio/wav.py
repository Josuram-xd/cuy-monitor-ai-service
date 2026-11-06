"""Reading the WAV clips the edge agent sends: 16-bit PCM, any rate, mono or stereo."""

import io
import wave
from math import gcd

import numpy as np
from scipy.signal import resample_poly


class UnsupportedAudio(ValueError):
    """Not a 16-bit PCM WAV, or empty."""


def read_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """(mono samples in -1..1, sample rate). Stereo is averaged to mono."""
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


def resample(samples: np.ndarray, rate: int, target_rate: int) -> np.ndarray:
    """The same sound at target_rate (polyphase filter, so no aliasing from a 44.1 kHz phone)."""
    if rate == target_rate:
        return samples
    divisor = gcd(rate, target_rate)
    return resample_poly(samples, target_rate // divisor, rate // divisor)
