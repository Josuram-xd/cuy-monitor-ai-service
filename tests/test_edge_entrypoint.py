"""Task 14.1: how the edge agent starts (python -m edge_agent) and wires its loops."""

import asyncio

import pytest

import edge_agent.__main__ as entrypoint
import edge_agent.agent as agent
from edge_agent.config import EdgeSettings


def _settings(**overrides) -> EdgeSettings:
    values = {
        "a12_url": "http://phone:8080",
        "ai_url": "https://ai.example",
        "api_key": "secret",
        "cage_id": "cage-1",
    }
    return EdgeSettings(_env_file=None, **(values | overrides))


@pytest.mark.anyio
@pytest.mark.parametrize(("audio", "loops"), [(True, 3), (False, 1)])
async def test_run_starts_the_frame_loop_and_the_audio_loops_only_when_enabled(
    monkeypatch, audio, loops
):
    started = []

    def fake(name):
        async def loop(*args, **kwargs):
            started.append(name)

        return loop

    monkeypatch.setattr(agent, "run_frames", fake("frames"))
    monkeypatch.setattr(agent, "record_audio", fake("record"))
    monkeypatch.setattr(agent, "send_audio", fake("send"))

    await agent.run(_settings(audio_enabled=audio))

    assert len(started) == loops
    assert started[0] == "frames"


def test_main_runs_the_agent_with_the_settings(monkeypatch):
    seen = []

    async def fake_run(settings):
        seen.append(settings.cage_id)

    monkeypatch.setattr(entrypoint, "EdgeSettings", lambda: _settings(cage_id="cage-9"))
    monkeypatch.setattr(entrypoint, "run", fake_run)

    assert entrypoint.main() == 0
    assert seen == ["cage-9"]


def test_main_names_the_missing_settings_without_their_values(monkeypatch, capsys):
    for name in ("A12_URL", "AI_URL", "API_KEY", "CAGE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        entrypoint, "EdgeSettings", lambda: EdgeSettings(_env_file=None, api_key="top-secret")
    )

    assert entrypoint.main() == 2

    err = capsys.readouterr().err
    assert "A12_URL" in err and "CAGE_ID" in err
    assert "top-secret" not in err


def test_ctrl_c_stops_the_agent_cleanly(monkeypatch):
    def interrupted(coroutine):
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(entrypoint, "EdgeSettings", _settings)
    monkeypatch.setattr(entrypoint.asyncio, "run", interrupted)

    assert entrypoint.main() == 0


def test_the_clip_buffer_waits_for_the_next_clip():
    async def scenario():
        buffer = agent.ClipBuffer()
        waiting = asyncio.create_task(buffer.get())
        await asyncio.sleep(0)
        buffer.put(b"late", None)
        return await waiting

    assert asyncio.run(scenario()) == (b"late", None)
