from scripts.fetch_models import MODELS


def test_the_animal_detector_is_downloaded_with_its_own_variables():
    rows = {row[0]: row for row in MODELS}

    assert rows["ANIMAL_DETECTOR_URL"] == (
        "ANIMAL_DETECTOR_URL",
        "ANIMAL_DETECTOR_SHA256",
        "ANIMAL_DETECTOR_PATH",
        "models/animal_detector.onnx",
        "animal detector",
    )
