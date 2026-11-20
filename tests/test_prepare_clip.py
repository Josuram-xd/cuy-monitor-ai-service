"""Task A7: the marks of the simulation clip are interpolated, painted and packed correctly."""

import hashlib
import zipfile

import numpy as np

from app.contracts import MarkColor
from dev.prepare_clip import interpolate_marks, pack_clip, paint_marks, zone_fractions
from training.synthetic_dataset import MARK_BGR


def test_a_mark_moves_in_a_straight_line_between_two_clicks():
    clicks = {"RED": {0: (100.0, 200.0), 4: (300.0, 200.0)}}

    frames = interpolate_marks(clicks, 6)

    assert frames[0] == {"RED": [100, 200]}
    assert frames[2] == {"RED": [200, 200]}
    assert frames[4] == {"RED": [300, 200]}
    assert frames[5] == {}  # after the last click: not drawn


def test_a_not_visible_click_ends_the_mark_until_the_next_visible_one():
    clicks = {"BLUE": {0: (10.0, 10.0), 2: None, 4: (50.0, 50.0), 6: (60.0, 60.0)}}

    frames = interpolate_marks(clicks, 8)

    assert frames[0] == {"BLUE": [10, 10]}
    assert frames[1] == {}  # the segment towards the 'not visible' click is empty
    assert frames[3] == {}
    assert frames[5] == {"BLUE": [55, 55]}


def test_the_marks_of_two_colours_stay_apart():
    clicks = {"RED": {0: (1.0, 1.0)}, "GREEN": {0: (2.0, 2.0)}}

    frames = interpolate_marks(clicks, 1)

    assert frames == [{"RED": [1, 1], "GREEN": [2, 2]}]


def test_a_painted_mark_has_the_colour_the_detector_was_trained_with():
    image = np.zeros((100, 200, 3), np.uint8)

    paint_marks(image, {"RED": [50, 50]})

    assert tuple(int(v) for v in image[50, 50]) == MARK_BGR[MarkColor.RED]
    assert image[50, 150].sum() == 0


def test_zones_are_stored_as_fractions_of_the_frame():
    assert zone_fractions([(64.0, 48.0), (320.0, 0.0)], 640, 480) == [[0.1, 0.1], [0.5, 0.0]]


def test_the_packed_clip_holds_the_manifest_and_frames_and_its_sha256(tmp_path):
    folder = tmp_path / "sim_clip"
    (folder / "frames").mkdir(parents=True)
    (folder / "manifest.json").write_text("{}", encoding="utf-8")
    (folder / "frames" / "0001.jpg").write_bytes(b"jpeg")

    digest = pack_clip(folder, tmp_path / "sim_clip.zip")

    with zipfile.ZipFile(tmp_path / "sim_clip.zip") as bundle:
        assert sorted(bundle.namelist()) == ["frames/0001.jpg", "manifest.json"]
    assert digest == hashlib.sha256((tmp_path / "sim_clip.zip").read_bytes()).hexdigest()
