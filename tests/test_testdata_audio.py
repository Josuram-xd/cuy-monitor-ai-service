import pytest

from app.audio.classifier import classify_clip
from app.contracts.events import AudioLabel
from app.testdata.audio import EXPECTED_LABEL, ClipKind, synthesize_clip


@pytest.mark.parametrize("kind", list(ClipKind))
def test_the_classifier_gives_the_expected_label_for_every_synthetic_clip(kind):
    result = classify_clip(synthesize_clip(kind))

    assert result.label is EXPECTED_LABEL[kind]
    assert 900 <= result.duration_ms <= 1000


def test_a_squeal_is_distress_with_high_probability_whatever_the_seed():
    for seed in range(5):
        result = classify_clip(synthesize_clip(ClipKind.SQUEAL, seed=seed))
        assert result.label is AudioLabel.DISTRESS
        assert result.probability >= 0.8


def test_the_same_kind_and_seed_give_the_same_bytes():
    assert synthesize_clip(ClipKind.CHIRP, seed=3) == synthesize_clip(ClipKind.CHIRP, seed=3)
    assert synthesize_clip(ClipKind.CHIRP, seed=3) != synthesize_clip(ClipKind.CHIRP, seed=4)
