import asyncio
import random

import pytest

from app.contracts import MarkColor
from app.testdata.audio import ClipKind
from app.testdata.video import (
    CALM_CLIPS,
    SCRIPT_DROP_GRAMS,
    SCRIPT_START_WINDOW,
    is_simulated,
    load_clip,
    run_video_simulation,
    scripted_window,
    source_for,
)
from tests.clip_helpers import write_clip


def test_source_for():
    assert is_simulated("sim-0a1b2c3d")
    assert not is_simulated("cage-1")
    assert source_for("sim-0a1b2c3d") == "simulation"
    assert source_for("cage-1") == "ai-service"


def test_a_valid_clip_is_loaded(tmp_path):
    write_clip(tmp_path)

    clip = load_clip(tmp_path)

    assert clip is not None
    assert clip.colors == (MarkColor.RED, MarkColor.BLUE)
    assert [frame.name for frame in clip.frames] == ["0001.jpg", "0002.jpg"]
    assert clip.credit["license"] == "CC BY 4.0"
    assert clip.base_weight_grams == 2700
    assert clip.zones.feeder is not None and clip.zones.waterer is None


def test_the_reference_crops_of_a_clip_are_listed_by_colour(tmp_path):
    write_clip(tmp_path, references=True)

    clip = load_clip(tmp_path)

    assert clip.references == {
        MarkColor.RED: tmp_path / "refs" / "RED.jpg",
        MarkColor.BLUE: tmp_path / "refs" / "BLUE.jpg",
    }


def test_an_old_clip_without_references_has_none(tmp_path):
    write_clip(tmp_path)

    assert load_clip(tmp_path).references == {}


def test_a_listed_crop_that_is_missing_makes_the_clip_broken(tmp_path):
    write_clip(tmp_path, references=True)
    (tmp_path / "refs" / "RED.jpg").unlink()

    assert load_clip(tmp_path) is None


def test_a_missing_clip_folder_gives_none(tmp_path):
    assert load_clip(tmp_path / "absent") is None


def test_a_colour_that_is_not_a_mark_gives_none(tmp_path):
    write_clip(tmp_path, colors=("PINK",))

    assert load_clip(tmp_path) is None


def test_the_clip_loops_and_each_window_of_frames_is_reported(tmp_path):
    write_clip(tmp_path)
    clip = load_clip(tmp_path)
    sent, reports, progress, pauses = [], [], [], []

    async def submit(frame):
        sent.append(frame)

    async def report_window():
        reports.append(len(sent))

    async def sleep(seconds):
        pauses.append(seconds)

    # 10 s per frame: 6 frames make a window, and 1 minute is 6 frames: a loop of 2 frames, 3 times
    demo = run_video_simulation(
        "sim-0a1b2c3d",
        clip,
        submit,
        report_window,
        progress.append,
        minutes=1,
        interval=10,
        sleep=sleep,
    )
    asyncio.run(demo)

    assert len(sent) == 6
    assert sent[0] == sent[2] == sent[4]  # the two frames of the clip, in turns
    assert reports == [6]
    assert progress == [1, 2, 3, 4, 5, 6]
    assert pauses == [10] * 6


def test_the_scripted_weight_falls_and_the_distress_calls_start_later():
    rng = random.Random(1)
    base = 2700.0

    calm, _ = scripted_window(0, base, rng)
    dropping, first_clip = scripted_window(SCRIPT_START_WINDOW, base, rng)
    second, second_clip = scripted_window(SCRIPT_START_WINDOW + 1, base, rng)
    floor, floor_clip = scripted_window(SCRIPT_START_WINDOW + 10, base, rng)

    assert abs(calm - base) <= 3
    assert base - dropping > 15 and first_clip is ClipKind.SQUEAL
    assert base - second > base - dropping and second_clip is ClipKind.SQUEAL
    assert base - floor == pytest.approx(3 * SCRIPT_DROP_GRAMS, abs=3)
    assert floor_clip in CALM_CLIPS
