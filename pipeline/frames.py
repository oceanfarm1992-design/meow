"""Frame extraction (ffmpeg) and motion scoring (OpenCV)."""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

STUDY_FPS = 2          # frames per second sampled for study
STUDY_WIDTH = 360      # downscale for fast CPU work
BLUR_KERNEL = (5, 5)   # suppress sensor noise before differencing


@dataclass(frozen=True)
class Frame:
    index: int
    t: float
    path: Path


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    kinds = {s["codec_type"] for s in data.get("streams", [])}
    return {"duration": float(data["format"]["duration"]), "has_audio": "audio" in kinds}


def extract_frames(video: Path, out_dir: Path) -> list[Frame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("f_*.jpg"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vf", f"fps={STUDY_FPS},scale={STUDY_WIDTH}:-2",
         "-q:v", "3", str(out_dir / "f_%05d.jpg")],
        check=True,
    )
    paths = sorted(out_dir.glob("f_*.jpg"))
    # ffmpeg's fps filter centres samples in each interval
    return [Frame(i, round((i + 0.5) / STUDY_FPS, 2), p) for i, p in enumerate(paths)]


def motion_scores(frames: list[Frame]) -> list[float]:
    """Mean absolute grey-level change vs. previous frame, scaled to 0..1."""
    raw: list[float] = []
    prev = None
    for f in frames:
        grey = cv2.GaussianBlur(cv2.imread(str(f.path), cv2.IMREAD_GRAYSCALE), BLUR_KERNEL, 0)
        raw.append(0.0 if prev is None else float(np.mean(cv2.absdiff(grey, prev))))
        prev = grey
    if len(raw) > 1:
        raw[0] = raw[1]  # first frame has no predecessor
    peak = max(raw) or 1.0
    return [round(r / peak, 3) for r in raw]


def audio_loudness(video: Path, times: list[float]) -> list[float] | None:
    """RMS loudness (0..1) around each frame time, or None if no audio."""
    import librosa  # heavy import; only needed when audio exists

    y, sr = librosa.load(str(video), sr=16000, mono=True)
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    peak = float(rms.max()) or 1.0
    idx = [min(int(t * sr / hop), len(rms) - 1) for t in times]
    return [round(float(rms[i]) / peak, 3) for i in idx]
