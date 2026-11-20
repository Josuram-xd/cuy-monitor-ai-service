"""Downloads the trained models at image build time (Tasks 7.4, 10.2 and 11).

Without DETECTOR_URL / BEHAVIOR_MODEL_URL that model is skipped and the service starts without it
(/ai/health says so). With it, the file is checked against its sha256 before it is kept: a wrong
or truncated download fails the build instead of shipping a broken model.
"""

import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


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


def extract_clip(archive: Path, folder: Path) -> Path:
    """Unpacks the video simulation clip into folder. A name that leaves the folder is an error."""
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute():
                raise ValueError(f"{archive} has an absolute path: {name}")
            if ".." in PurePosixPath(name).parts or ".." in PureWindowsPath(name).parts:
                raise ValueError(f"{archive} has a path outside its folder: {name}")
        bundle.extractall(folder)
    return folder


# (URL variable, sha256 variable, path variable, default path, what it is)
MODELS = (
    ("DETECTOR_URL", "DETECTOR_SHA256", "DETECTOR_PATH", "models/detector.onnx", "detector"),
    (
        "BEHAVIOR_MODEL_URL",
        "BEHAVIOR_MODEL_SHA256",
        "BEHAVIOR_MODEL_PATH",
        "models/behavior_rf.joblib",
        "behavior classifier",
    ),
    ("YAMNET_URL", "YAMNET_SHA256", "YAMNET_PATH", "models/yamnet.onnx", "YAMNet model"),
    (
        "AUDIO_MODEL_URL",
        "AUDIO_MODEL_SHA256",
        "AUDIO_MODEL_PATH",
        "models/audio_clf.joblib",
        "audio classifier",
    ),
    # a zip of the prepared video (Task A7): unpacked next to it, see main()
    ("SIM_CLIP_URL", "SIM_CLIP_SHA256", "SIM_CLIP_PATH", "models/sim_clip.zip", "simulation clip"),
)


def main() -> int:
    for url_var, sha_var, path_var, default_path, name in MODELS:
        url = os.getenv(url_var, "").strip()
        target = Path(os.getenv(path_var, default_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        if not url:
            print(f"No {url_var}: the image is built without the {name}")
            continue
        sha256 = os.getenv(sha_var, "").strip()
        if not sha256:
            # the behavior model is a pickle: never load one that was not checked
            print(f"{sha_var} is required with {url_var}", file=sys.stderr)
            return 1
        downloaded = fetch(url, sha256, target)
        print(f"The {name} was downloaded to {downloaded}")
        if downloaded.suffix == ".zip":
            folder = extract_clip(downloaded, downloaded.with_suffix(""))
            downloaded.unlink()
            print(f"The {name} was unpacked to {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
