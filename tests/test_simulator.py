from pathlib import Path

import pytest

from dev import simulator


class FakeCapture:
    def __init__(self, frames, fps=4.0, opened=True):
        self.frames = frames
        self.fps = fps
        self.opened = opened
        self.released = False
        self.position = 0

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        assert property_id == simulator.cv2.CAP_PROP_FPS
        return self.fps

    def read(self):
        if self.position == len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


class FakeJpeg:
    def __init__(self, frame):
        self.frame = frame

    def tobytes(self):
        return f"jpeg-{self.frame}".encode()


class FakeHttpClient:
    def __init__(self, response_status=202):
        self.response_status = response_status
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return type("Response", (), {"status_code": self.response_status})()


def test_simulator_samples_frames_and_posts_authenticated_multipart(monkeypatch, tmp_path):
    video = tmp_path / "recording.mp4"
    video.touch()
    capture = FakeCapture(frames=list(range(5)), fps=4.0)
    http_client = FakeHttpClient()
    delays = []
    monkeypatch.setattr(simulator.cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(
        simulator.cv2,
        "imencode",
        lambda extension, frame: (extension == ".jpg", FakeJpeg(frame)),
    )
    monkeypatch.setattr(simulator.httpx, "Client", lambda timeout: http_client)
    monkeypatch.setattr(simulator.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(simulator.time, "sleep", delays.append)

    sent = simulator.replay_video(
        video=video,
        ai_url="http://ai.local/",
        cage_id="cage-test",
        api_key="test-secret",
        fps=2,
    )

    assert sent == 3
    assert capture.released
    assert delays == [0.5, 0.5, 0.5]
    assert [url for url, _ in http_client.requests] == [
        "http://ai.local/ai/frames",
    ] * 3
    first_request = http_client.requests[0][1]
    assert first_request["headers"] == {"X-API-Key": "test-secret"}
    assert first_request["data"]["cageId"] == "cage-test"
    assert first_request["data"]["capturedAt"].endswith("Z")
    assert first_request["files"]["file"] == (
        "frame.jpg",
        b"jpeg-0",
        "image/jpeg",
    )


def test_simulator_releases_video_when_upload_is_rejected(monkeypatch, tmp_path):
    video = tmp_path / "recording.mp4"
    video.touch()
    capture = FakeCapture(frames=[0])
    monkeypatch.setattr(simulator.cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(simulator.cv2, "imencode", lambda *_: (True, FakeJpeg(0)))
    monkeypatch.setattr(simulator.httpx, "Client", lambda timeout: FakeHttpClient(401))
    monkeypatch.setattr(simulator.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(simulator.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="HTTP 401"):
        simulator.replay_video(video, "http://ai.local", "cage-1", "secret", fps=1)

    assert capture.released


def test_simulator_requires_valid_video_frame_rate(monkeypatch, tmp_path):
    video = tmp_path / "recording.mp4"
    video.touch()
    capture = FakeCapture(frames=[], fps=0)
    monkeypatch.setattr(simulator.cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(ValueError, match="--source-fps"):
        simulator.replay_video(video, "http://ai.local", "cage-1", "secret", fps=1)

    assert capture.released


def test_simulator_rejects_missing_video_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        simulator.replay_video(
            Path(tmp_path / "missing.mp4"),
            "http://ai.local",
            "cage-1",
            "secret",
            fps=1,
        )
