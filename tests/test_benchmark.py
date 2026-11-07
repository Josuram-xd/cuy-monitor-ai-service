"""Task 13.1: the benchmark runs with whatever models there are (none here) and writes a report."""

import json

from dev.benchmark import main, peak_memory_mb, timed


def test_without_models_it_still_times_the_window_and_the_signal_rule(tmp_path, capsys):
    out = tmp_path / "metrics" / "laptop.json"

    assert main(["--runs", "3", "--out", str(out)]) == 0

    report = json.loads(out.read_text())
    assert report["machine"]["python"]
    assert "detector" not in report
    assert report["behaviorWindow"]["runs"] == 20
    assert set(report["audio"]) == {"signalRule"}
    assert report["audio"]["signalRule"]["p50Ms"] >= 0
    assert json.loads(capsys.readouterr().out) == report


def test_timing_reports_the_percentiles():
    result = timed(lambda: sum(range(100)), runs=10, warmup=1)

    assert result["runs"] == 10
    assert 0 <= result["p50Ms"] <= result["p95Ms"]


def test_peak_memory_is_in_megabytes_where_the_os_tells_it():
    memory = peak_memory_mb()
    assert memory is None or 1 < memory < 100_000


def test_audio_timings_share_the_clips_without_running_out(monkeypatch):
    import dev.benchmark as benchmark

    class _Embedder:
        @classmethod
        def load(cls, path):
            return cls()

        def embed(self, clip):
            return [0.0]

    monkeypatch.setattr("app.audio.embeddings.YamnetEmbedder", _Embedder)

    result = benchmark.bench_audio(yamnet=object(), model=None, runs=30)

    assert set(result) == {"signalRule", "yamnetEmbedding"}
