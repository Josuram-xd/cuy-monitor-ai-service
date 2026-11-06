"""Exports YAMNet to ONNX, once, in Google Colab (Task 11.1).

TensorFlow does not run on Python 3.14, so the service can't load YAMNet from TF Hub. This script
runs where TensorFlow does (Colab) and writes:

- models/yamnet.onnx: input "waveform" (float32, 16 kHz mono, any length >= 15600 samples), output
  "embeddings" (frames x 1024). It is what app/audio/embeddings.py loads with onnxruntime.
- models/yamnet.json: source, opset, sha256 (for YAMNET_SHA256 at image build) and the largest
  difference between TensorFlow and ONNX on a test signal, so a broken export never reaches the
  service.

In Colab:
    !pip install -r training/requirements-audio.txt
    !python training/export_yamnet.py --out models/yamnet.onnx

It imports nothing from app/ on purpose: Colab's Python may be older than the service's.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
OPSET = 17
TOLERANCE = 1e-3


def test_signal(seconds: float = 2.0, rate: int = 16_000) -> np.ndarray:
    """A sweep plus noise: exercises the whole spectrum, the same every run."""
    time = np.arange(int(seconds * rate)) / rate
    sweep = np.sin(2 * np.pi * (200 + 3500 * time / seconds) * time)
    noise = np.random.default_rng(0).normal(0, 0.05, time.size)
    return (0.5 * sweep + noise).astype(np.float32)


def export(out: Path) -> dict:
    import onnx
    import onnxruntime
    import tensorflow as tf
    import tensorflow_hub as hub
    import tf2onnx
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    model = hub.load(YAMNET_HANDLE)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32, name="waveform")])
    def embeddings(waveform):
        _, frames, _ = model(waveform)
        return frames

    # the weights of a TF Hub model are variables: freeze them into constants, or the ONNX graph
    # asks for every weight as one more input
    frozen = convert_variables_to_constants_v2(embeddings.get_concrete_function())
    out.parent.mkdir(parents=True, exist_ok=True)
    converted, _ = tf2onnx.convert.from_graph_def(
        frozen.graph.as_graph_def(),
        input_names=[tensor.name for tensor in frozen.inputs],
        output_names=[tensor.name for tensor in frozen.outputs],
        opset=OPSET,
    )
    _rename(converted, converted.graph.input[0].name, "waveform")
    _rename(converted, converted.graph.output[0].name, "embeddings")
    onnx.checker.check_model(converted)
    onnx.save(converted, str(out))

    signal = test_signal()
    expected = embeddings(tf.constant(signal)).numpy()
    session = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    if [i.name for i in session.get_inputs()] != ["waveform"]:
        raise ValueError(f"The ONNX model wants inputs {[i.name for i in session.get_inputs()]}")
    actual = session.run(["embeddings"], {"waveform": signal})[0]
    if actual.shape != expected.shape or actual.shape[1] != 1024:
        raise ValueError(f"ONNX gives {actual.shape}, TensorFlow {expected.shape}")
    difference = float(np.max(np.abs(actual - expected)))
    if difference > TOLERANCE:
        raise ValueError(f"ONNX differs from TensorFlow by {difference} (> {TOLERANCE})")

    report = {
        "source": YAMNET_HANDLE,
        "opset": OPSET,
        "input": "waveform float32 [samples] 16 kHz mono",
        "output": "embeddings float32 [frames, 1024]",
        "maxDifferenceVsTensorFlow": difference,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "file": out.name,
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _rename(model, old: str, new: str) -> None:
    """Gives a graph input or output a stable name ("waveform:0" -> "waveform")."""
    for value in [*model.graph.input, *model.graph.output]:
        if value.name == old:
            value.name = new
    for node in model.graph.node:
        node.input[:] = [new if name == old else name for name in node.input]
        node.output[:] = [new if name == old else name for name in node.output]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export YAMNet (TF Hub) to ONNX")
    parser.add_argument("--out", type=Path, default=Path("models/yamnet.onnx"))
    args = parser.parse_args(argv)
    try:
        report = export(args.out)
    except ValueError as exception:
        print(exception, file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
