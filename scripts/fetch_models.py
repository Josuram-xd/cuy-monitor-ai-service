"""Downloads the trained detector at image build time (Task 7.4).

Without DETECTOR_URL it does nothing and the service starts without a detector (/ai/health says so).
With it, the file is checked against DETECTOR_SHA256 before it is kept: a wrong or truncated
download fails the build instead of shipping a broken model.
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path


def fetch(url: str, sha256: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as file:
        while chunk := response.read(1 << 20):
            file.write(chunk)
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    if sha256 and digest != sha256.lower():
        partial.unlink()
        raise ValueError(f"sha256 of {url} is {digest}, expected {sha256}")
    partial.replace(target)
    return target


def main() -> int:
    url = os.getenv("DETECTOR_URL", "").strip()
    target = Path(os.getenv("DETECTOR_PATH", "models/detector.onnx"))
    target.parent.mkdir(parents=True, exist_ok=True)
    if not url:
        print("No DETECTOR_URL: the image is built without a detector")
        return 0
    sha256 = os.getenv("DETECTOR_SHA256", "").strip()
    if not sha256:
        print("DETECTOR_SHA256 is required with DETECTOR_URL", file=sys.stderr)
        return 1
    print(f"Detector downloaded to {fetch(url, sha256, target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
