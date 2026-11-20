import numpy as np
import pytest

from app.vision.appearance import (
    contradicts,
    describe,
    describe_reference,
    dominant_tone,
    similarity,
)

BOX = (10.0, 10.0, 90.0, 90.0)


def coat(bgr, size=100, noise=6, seed=0):
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), bgr, dtype=np.float32) + rng.normal(0, noise, (size, size, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


WHITE = (235, 235, 235)
BLACK = (20, 20, 20)
BROWN = (40, 80, 150)  # BGR: a reddish brown
CREAM = (190, 215, 235)


def test_the_same_coat_is_very_similar_to_itself():
    a = describe(coat(BROWN, seed=1), BOX)
    b = describe(coat(BROWN, seed=2), BOX)

    assert a.dtype == np.float32 and a.sum() == pytest.approx(1, abs=1e-3)
    assert similarity(a, b) > 0.95


def test_white_and_black_have_nothing_in_common():
    assert similarity(describe(coat(WHITE), BOX), describe(coat(BLACK), BOX)) < 0.05


def test_a_cream_coat_is_closer_to_white_than_a_brown_one_is():
    brown = describe(coat(BROWN), BOX)
    cream = describe(coat(CREAM), BOX)
    white = describe(coat(WHITE), BOX)

    # saturated brown and washed-out cream fall in different saturation bins: no overlap at all
    assert similarity(brown, cream) < similarity(brown, brown)
    assert similarity(cream, white) > similarity(brown, white)


def test_the_bedding_in_the_corners_is_left_out():
    image = coat(WHITE)
    image[:12, :12] = (0, 0, 255)  # red corner, outside the central ellipse

    assert similarity(describe(image, BOX), describe(coat(WHITE), BOX)) > 0.95


def test_tones_and_vetoes():
    assert dominant_tone(describe(coat(WHITE), BOX)) == "light"
    assert dominant_tone(describe(coat(BLACK), BOX)) == "dark"
    assert dominant_tone(describe(coat(BROWN), BOX)) == "colour"
    assert contradicts("WHITE", describe(coat(BLACK), BOX))
    assert contradicts("CREAM", describe(coat(BLACK), BOX))
    assert contradicts("BLACK", describe(coat(WHITE), BOX))
    assert not contradicts("BLACK", describe(coat(BLACK), BOX))
    assert not contradicts("BICOLOR", describe(coat(BLACK), BOX))
    assert not contradicts(None, describe(coat(WHITE), BOX))


def test_a_photo_smaller_than_64_pixels_is_not_a_reference():
    assert describe_reference(np.zeros((32, 32, 3), np.uint8), None) is None
    assert describe_reference(coat(BROWN, size=120), None) is not None


def test_an_empty_box_is_refused():
    with pytest.raises(ValueError):
        describe(coat(BROWN), (500.0, 500.0, 600.0, 600.0))
