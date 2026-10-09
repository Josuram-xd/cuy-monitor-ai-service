import argparse
import logging
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import httpx

logger = logging.getLogger("cuy-monitor-simulator")


def _positive_float(value: str) -> float:
    try:
        parsed_value = float(value)
    except ValueError as exception:
        raise argparse.ArgumentTypeError("must be a positive number") from exception
    if not math.isfinite(parsed_value) or parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed_value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay video frames to the Cuy Monitor AI service as a live camera."
    )
    parser.add_argument("video", type=Path, help="path to the recorded video")
    parser.add_argument(
        "--ai-url",
        default=os.getenv("AI_URL", "http://localhost:8000"),
        help="base URL of the AI service (default: AI_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--cage-id",
        default=os.getenv("CAGE_ID", "cage-1"),
        help="cage identifier (default: CAGE_ID or cage-1)",
    )
    parser.add_argument(
        "--fps",
        type=_positive_float,
        default=1.0,
        help="maximum frame upload rate (default: 1)",
    )
    parser.add_argument(
        "--source-fps",
        type=_positive_float,
        help="override the source video's frame rate if its metadata is missing",
    )
    return parser


def replay_video(
    video: Path,
    ai_url: str,
    cage_id: str,
    api_key: str,
    fps: float,
    source_fps_override: float | None = None,
) -> int:
    if not video.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive number")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video file: {video}")

    source_fps = (
        source_fps_override if source_fps_override is not None else capture.get(cv2.CAP_PROP_FPS)
    )
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise ValueError("Video has no valid frame rate; provide --source-fps")

    upload_fps = min(fps, source_fps)
    frame_stride = max(1, round(source_fps / upload_fps))
    frame_period = 1 / upload_fps
    endpoint = f"{ai_url.rstrip('/')}/ai/frames"
    headers = {"X-API-Key": api_key}
    frame_index = 0
    frames_sent = 0

    try:
        with httpx.Client(timeout=10.0) as client:
            while True:
                has_frame, frame = capture.read()
                if not has_frame:
                    break

                if frame_index % frame_stride == 0:
                    started_at = time.monotonic()
                    encoded, jpeg = cv2.imencode(".jpg", frame)
                    if not encoded:
                        raise RuntimeError(f"Could not encode video frame {frame_index} as JPEG")

                    response = client.post(
                        endpoint,
                        headers=headers,
                        data={
                            "capturedAt": (
                                datetime.now(UTC)
                                .isoformat(timespec="milliseconds")
                                .replace("+00:00", "Z")
                            ),
                            "cageId": cage_id,
                        },
                        files={"file": ("frame.jpg", jpeg.tobytes(), "image/jpeg")},
                    )
                    if response.status_code != 202:
                        raise RuntimeError(f"Frame upload failed with HTTP {response.status_code}")

                    frames_sent += 1
                    logger.info("Uploaded frame %d", frame_index)
                    time.sleep(max(0, frame_period - (time.monotonic() - started_at)))

                frame_index += 1
    finally:
        capture.release()

    logger.info("Finished replay: uploaded %d frames", frames_sent)
    return frames_sent


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    api_key = os.getenv("API_KEY")
    if not api_key:
        parser.error("API_KEY must be set in the environment")

    try:
        return replay_video(
            video=args.video,
            ai_url=args.ai_url,
            cage_id=args.cage_id,
            api_key=api_key,
            fps=args.fps,
            source_fps_override=args.source_fps,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError, httpx.HTTPError) as exception:
        parser.error(str(exception))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
