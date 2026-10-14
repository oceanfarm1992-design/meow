"""Narrate a script with an ElevenLabs voice, one request per line.

Writes the same layout as the cloned-voice engine so render.py and
check_voice.py work unchanged:
    <out>/voice.wav               all lines joined with short pauses
    <out>/parts/part_XX.wav       one file per line
    <out>/texts/part_XX.txt       the clean text of each line (tags removed)

Usage:
    python pipeline/tts_elevenlabs.py script.ta.txt.tagged --voice karees_mass_hero_ta --out el/
Key: ELEVENLAB_API. Voices are listed in voices/voices.json.
"""
import argparse
import json
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import requests

from check_voice import CUT_DB, tail_db
from keys import load_key

ROOT = Path(__file__).resolve().parent.parent
TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
SR = 44100
GAP_SECONDS = 0.3
STABILITY = 0.5   # eleven_v3: 0.0 creative, 0.5 natural, 1.0 robust
TAG = re.compile(r"^\s*\[[a-z ]+\]\s*")
MAX_TRIES = 3     # regenerate a line whose ending comes back cut off


def load_voice(name: str) -> dict:
    voices = json.loads((ROOT / "voices" / "voices.json").read_text(encoding="utf-8"))
    if name not in voices:
        raise SystemExit(f"unknown voice {name!r}; known: {', '.join(voices)}")
    return voices[name]


def synth(key: str, voice: dict, text: str, dest: Path) -> None:
    resp = requests.post(
        TTS_URL.format(voice_id=voice["voice_id"]),
        headers={"xi-api-key": key},
        json={"text": text, "model_id": voice["model_id"], "language_code": voice["language"],
              "voice_settings": {"stability": STABILITY}},
        timeout=120,
    )
    if not resp.ok:
        raise RuntimeError(f"ElevenLabs {resp.status_code}: {resp.text[:300]}")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fh:
        fh.write(resp.content)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", fh.name, "-ac", "1", "-ar", str(SR),
                    "-sample_fmt", "s16", str(dest)], check=True)
    Path(fh.name).unlink()


def synth_checked(key: str, voice: dict, text: str, dest: Path) -> int:
    """Synthesise, regenerating while the last 120 ms is still loud (cut mid-word).
    Keeps the attempt with the quietest ending; returns the number of tries used."""
    best = None
    for attempt in range(1, MAX_TRIES + 1):
        trial = dest.with_name(f"{dest.stem}.try{attempt}.wav")
        synth(key, voice, text, trial)
        level, _ = tail_db(trial)
        if best is None or level < best[0]:
            best = (level, trial)
        if level <= CUT_DB:
            break
    best[1].replace(dest)
    for leftover in dest.parent.glob(f"{dest.stem}.try*.wav"):
        leftover.unlink()
    return attempt


def read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def join(parts: list[Path], dest: Path) -> None:
    gap = np.zeros(int(GAP_SECONDS * SR), dtype=np.int16)
    pcm = np.concatenate([x for p in parts for x in (read_pcm(p), gap)][:-1])
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", type=Path, help="one sentence per line, optional [tag] prefix")
    ap.add_argument("--voice", default="karees_mass_hero_ta")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    voice = load_voice(args.voice)
    key = load_key("ELEVENLAB_API")
    lines = [l.strip() for l in args.script.read_text(encoding="utf-8").splitlines() if l.strip()]
    (args.out / "parts").mkdir(parents=True, exist_ok=True)
    (args.out / "texts").mkdir(parents=True, exist_ok=True)

    parts = []
    for i, line in enumerate(lines):
        wav = args.out / "parts" / f"part_{i:02d}.wav"
        tries = synth_checked(key, voice, line, wav)
        (args.out / "texts" / f"part_{i:02d}.txt").write_text(TAG.sub("", line), encoding="utf-8")
        parts.append(wav)
        print(f"wrote {wav.name} (engine=elevenlabs:{voice['model_id']}, tries {tries}) {line}")
    join(parts, args.out / "voice.wav")
    print(f"joined {len(parts)} parts into {args.out / 'voice.wav'}")


if __name__ == "__main__":
    main()
