from app.vision.window import FrameWindow


def _frames(count: int) -> list[bytes]:
    return [bytes([i]) for i in range(count)]


def test_drain_returns_everything_when_there_are_few_frames():
    window = FrameWindow()
    for frame in _frames(3):
        window.add(frame)

    assert window.drain(6) == _frames(3)
    assert len(window) == 0


def test_drain_samples_evenly_and_keeps_the_first_and_last_frame():
    window = FrameWindow()
    for frame in _frames(101):
        window.add(frame)

    sample = window.drain(6)

    assert [frame[0] for frame in sample] == [0, 20, 40, 60, 80, 100]


def test_a_sample_of_one_is_the_middle_frame():
    window = FrameWindow()
    for frame in _frames(11):
        window.add(frame)

    assert window.drain(1) == [bytes([5])]


def test_the_window_starts_empty_after_a_drain():
    window = FrameWindow()
    window.add(b"x")
    window.drain(3)

    assert window.drain(3) == []


def test_old_frames_are_dropped_when_the_loop_falls_behind():
    window = FrameWindow(max_frames=3)
    for frame in _frames(5):
        window.add(frame)

    assert window.drain(10) == [bytes([2]), bytes([3]), bytes([4])]
