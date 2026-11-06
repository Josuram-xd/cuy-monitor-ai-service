"""YAMNet embeddings from ONNX (Task 11.2).

YAMNet (Google, trained on AudioSet) turns a 16 kHz mono waveform into one 1024-number embedding
every 0.48 s (windows of 0.96 s). It is exported once to ONNX in Colab (training/export_yamnet.py)
because TensorFlow does not run on Python 3.14; here only onnxruntime is needed. A clip gives
several embeddings: their mean is the vector the audio classifier reads.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.audio.wav import read_wav, resample

SAMPLE_RATE = 16_000
EMBEDDING_SIZE = 1024
# YAMNet needs at least one 0.96 s window; shorter clips are padded with silence
MIN_SAMPLES = 15_600


class InferenceSession(Protocol):
    """The part of onnxruntime.InferenceSession the embedder uses (a fake one in the tests)."""

    def get_inputs(self) -> list[Any]: ...

    def get_outputs(self) -> list[Any]: ...

    def run(self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]) -> list: ...


class YamnetEmbedder:
    def __init__(self, session: InferenceSession):
        self._session = session
        self._input_name = session.get_inputs()[0].name
        names = [output.name for output in session.get_outputs()]
        # the export names it "embeddings"; any other export with a 1024-wide output also works
        self._output_name = "embeddings" if "embeddings" in names else None

    @classmethod
    def load(cls, path: Path) -> YamnetEmbedder:
        import onnxruntime

        return cls(onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"]))

    def embed_samples(self, samples: np.ndarray, rate: int) -> np.ndarray:
        """Mono samples in -1..1 at any rate -> one 1024-number vector (mean over YAMNet frames)."""
        waveform = resample(np.asarray(samples, dtype=np.float64), rate, SAMPLE_RATE)
        if waveform.size < MIN_SAMPLES:
            waveform = np.pad(waveform, (0, MIN_SAMPLES - waveform.size))
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)
        names = [self._output_name] if self._output_name else None
        outputs = self._session.run(names, {self._input_name: waveform})
        frames = self._pick_embeddings(outputs)
        return frames.mean(axis=0)

    def embed(self, wav_bytes: bytes) -> np.ndarray:
        samples, rate = read_wav(wav_bytes)
        return self.embed_samples(samples, rate)

    async def embed_async(self, wav_bytes: bytes) -> np.ndarray:
        # inference is CPU bound: never on the event loop (AGENTS rule 6)
        return await asyncio.to_thread(self.embed, wav_bytes)

    @staticmethod
    def _pick_embeddings(outputs: list) -> np.ndarray:
        for output in outputs:
            array = np.asarray(output, dtype=np.float32)
            if array.ndim == 2 and array.shape[1] == EMBEDDING_SIZE and array.shape[0] > 0:
                return array
        raise ValueError("The YAMNet model gave no (frames, 1024) embeddings")


def load_embedder(path: Path) -> YamnetEmbedder | None:
    """The embedder if yamnet.onnx is there and loads; None otherwise (never fatal)."""
    import logging

    logger = logging.getLogger(__name__)
    if not path.is_file():
        logger.warning("No YAMNet model at %s: audio uses the signal rule", path)
        return None
    try:
        return YamnetEmbedder.load(path)
    except Exception as exception:  # any loading problem means "no embedder"
        logger.error("YAMNet at %s could not be loaded: %s", path, exception)
        return None
