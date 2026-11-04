from app.audio.throttle import CalmAudioThrottle


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_one_calm_clip_per_cage_and_window():
    clock = _Clock()
    throttle = CalmAudioThrottle(window_seconds=60, clock=clock)

    assert throttle.should_report("cage-1") is True
    assert throttle.should_report("cage-1") is False
    assert throttle.should_report("cage-2") is True

    clock.now += 60
    assert throttle.should_report("cage-1") is True
