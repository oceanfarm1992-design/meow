"""Render the final Short: clip window + tightened voice + ducked music + banner.

Usage:
    python pipeline/render.py --clip in.mp4 --voice voice.wav --music music.wav \
        --out final.mp4 [--start 0 --length 30] [--banner "KAREES TALKS"] [--tempo 0.9]
"""
import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 720, 1280      # same as the reference Shorts
BANNER_RATIO = 0.085           # banner height as a share of the frame
MAX_GAP = 0.22                 # longest pause kept between sentences (s)
LEAD_IN = 0.3                  # silence before the first word (s)
SILENCE_DB = -40.0             # frames quieter than this vs. peak count as silence
FRAME = 0.02                   # analysis frame (s)
MUSIC_GAIN = 0.35              # music level before ducking
FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path)) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"{path}: expected 16-bit PCM, got {8 * width}-bit")
    return np.frombuffer(raw, dtype=np.int16).reshape(-1, ch), sr, ch


def tighten(src: Path, dst: Path) -> float:
    """Shorten every pause longer than MAX_GAP; returns the new duration (s)."""
    pcm, sr, ch = read_wav(src)
    mono = pcm.astype(np.float32).mean(axis=1)
    hop = int(FRAME * sr)
    frames = len(mono) // hop
    rms = np.sqrt((mono[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1)) + 1e-9
    loud = 20 * np.log10(rms / rms.max()) > SILENCE_DB

    keep = np.ones(frames, dtype=bool)
    max_gap, lead = int(MAX_GAP / FRAME), int(LEAD_IN / FRAME)
    i = 0
    while i < frames:
        if loud[i]:
            i += 1
            continue
        j = i
        while j < frames and not loud[j]:
            j += 1
        allowed = lead if i == 0 else max_gap
        if j - i > allowed:
            half = allowed // 2
            keep[i + half: j - (allowed - half)] = False
        i = j
    mask = np.repeat(keep, hop)
    mask = np.concatenate([mask, np.ones(len(pcm) - len(mask), dtype=bool)])
    out = pcm[mask]
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(out.tobytes())
    return len(out) / sr


def probe_seconds(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def ff_path(path: str) -> str:
    """Escape a path for use inside an ffmpeg filter argument."""
    return path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def build_filters(hold: float, banner_file: Path | None, font: str | None, tempo: float) -> str:
    video = (f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
             f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30")
    if hold > 0:
        video += f",tpad=stop_mode=clone:stop_duration={hold:.2f}"
    if banner_file and font:
        bar = int(HEIGHT * BANNER_RATIO)
        video += (f",drawbox=x=0:y=0:w=iw:h={bar}:color=black@1:t=fill,"
                  f"drawtext=fontfile='{ff_path(font)}':textfile='{ff_path(str(banner_file))}':"
                  f"fontcolor=white:fontsize={int(bar * 0.55)}:x=(w-tw)/2:y=({bar}-th)/2")
    lead_ms = int(LEAD_IN * 1000)
    # atempo changes speaking speed without changing pitch (0.9 = 10% slower)
    audio = (f"[1:a]aresample=44100,aformat=channel_layouts=stereo,atempo={tempo},"
             f"adelay={lead_ms}|{lead_ms},"
             f"apad,asplit=2[vo1][vo2];"
             f"[2:a]aresample=44100,aformat=channel_layouts=stereo,volume={MUSIC_GAIN}[mu];"
             f"[mu][vo1]sidechaincompress=threshold=0.02:ratio=12:attack=15:release=350[duck];"
             f"[duck][vo2]amix=inputs=2:normalize=0,alimiter=limit=0.95[a]")
    return f"{video}[v];{audio}"


def render(args: argparse.Namespace) -> None:
    if not 0.5 <= args.tempo <= 2.0:
        raise SystemExit("--tempo must be between 0.5 and 2.0")
    clip_len = args.length or probe_seconds(args.clip) - args.start
    with tempfile.TemporaryDirectory() as tmp:
        tight = Path(tmp) / "voice_tight.wav"
        if args.no_tighten:
            tight.write_bytes(args.voice.read_bytes())
            voice_len = probe_seconds(args.voice) / args.tempo + LEAD_IN
        else:
            voice_len = tighten(args.voice, tight) / args.tempo + LEAD_IN
        hold = max(0.0, voice_len + 0.6 - clip_len)  # never cut the narration
        total = clip_len + hold

        banner_file, font = None, None
        if args.banner:
            banner_file = Path(tmp) / "banner.txt"
            banner_file.write_text(args.banner, encoding="utf-8")
            font = next((f for f in FONTS if Path(f).exists()), None)

        cmd = ["ffmpeg", "-v", "error", "-y",
               "-ss", f"{args.start:.2f}", "-t", f"{clip_len:.2f}", "-i", str(args.clip),
               "-i", str(tight), "-stream_loop", "-1", "-i", str(args.music),
               "-filter_complex", build_filters(hold, banner_file, font, args.tempo),
               "-map", "[v]", "-map", "[a]", "-t", f"{total:.2f}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(args.out)]
        subprocess.run(cmd, check=True)
    print(f"rendered {args.out}: clip {args.start:.1f}s+{clip_len:.1f}s, "
          f"voice {voice_len:.1f}s (tempo {args.tempo}, pauses <= {MAX_GAP}s), "
          f"hold {hold:.1f}s, total {total:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--voice", type=Path, required=True)
    ap.add_argument("--music", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--length", type=float, default=0.0, help="0 = to the end of the clip")
    ap.add_argument("--banner", default="")
    ap.add_argument("--no-tighten", action="store_true",
                    help="keep the voice timing as is (story mode: scenes are cut to it)")
    ap.add_argument("--tempo", type=float, default=1.0,
                    help="voice speed: 0.9 = 10%% slower, same pitch (0.5-2.0)")
    render(ap.parse_args())


if __name__ == "__main__":
    main()
