"""Check narrated voice parts against their script text.

For each part_XX.wav with a matching part_XX.txt:
  - ending: is the last 120 ms still loud (speech cut mid-word)?
  - words: transcribe with OpenAI and compare to the script (similarity 0..1)

Usage:
    python pipeline/check_voice.py --wavs <dir> --texts <dir> [--lang ta] [--json out.json]
Prints a Markdown table (for $GITHUB_STEP_SUMMARY). Key: OPEN_AI_API.
"""
import argparse
import json
import os
import re
import wave
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import requests

from keys import load_key

STT_URL = "https://api.openai.com/v1/audio/transcriptions"
STT_MODEL = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-transcribe")
TAIL_SECONDS = 0.12
CUT_DB = -20.0          # tail louder than this (vs. part peak) = cut mid-word
WORDS_OK = 0.80         # transcript similarity at or above this = words OK
SCRIPT_RANGES = {"ta": "஀-௿", "en": "A-Za-z"}


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {2: np.int16, 4: np.int32}[width]
    y = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    return y.reshape(-1, ch).mean(axis=1), sr


def tail_db(path: Path) -> tuple[float, float]:
    """(level of the last TAIL_SECONDS in dB vs. peak, duration in s)."""
    y, sr = read_wav(path)
    peak = float(np.abs(y).max()) or 1.0
    tail = y[-int(TAIL_SECONDS * sr):]
    rms = float(np.sqrt(np.mean(tail ** 2))) or 1e-9
    return round(20 * np.log10(rms / peak), 1), round(len(y) / sr, 2)


def transcribe(key: str, path: Path, lang: str) -> str:
    with path.open("rb") as fh:
        resp = requests.post(
            STT_URL,
            headers={"Authorization": f"Bearer {key}"},
            data={"model": STT_MODEL, "language": lang, "response_format": "json"},
            files={"file": (path.name, fh, "audio/wav")},
            timeout=120,
        )
    if not resp.ok:
        raise RuntimeError(f"OpenAI {resp.status_code}: {resp.text[:300]}")
    return resp.json()["text"].strip()


def normalise(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def check_part(key: str, wav: Path, script: str, lang: str) -> dict:
    level, dur = tail_db(wav)
    heard = transcribe(key, wav, lang)
    sim = round(SequenceMatcher(None, normalise(script), normalise(heard)).ratio(), 2)
    issues = []
    if level > CUT_DB:
        issues.append("cut ending")
    script_chars = SCRIPT_RANGES.get(lang)
    if script_chars and not re.search(f"[{script_chars}]", heard):
        issues.append("unclear speech")
    elif sim < WORDS_OK:
        issues.append("words differ")
    return {"part": wav.stem, "seconds": dur, "tail_db": level, "similarity": sim,
            "script": script, "heard": heard, "status": ", ".join(issues) or "ok"}


def markdown(results: list[dict]) -> str:
    bad = sum(r["status"] != "ok" for r in results)
    lines = [f"## Voice check ({len(results) - bad}/{len(results)} parts ok, STT {STT_MODEL})", "",
             "| Part | Status | Ending dB | Match | Script | Heard |", "|---|---|---|---|---|---|"]
    for r in results:
        mark = "✅" if r["status"] == "ok" else "⚠️ " + r["status"]
        lines.append(f"| {r['part']} | {mark} | {r['tail_db']} | {r['similarity']:.2f} "
                     f"| {r['script']} | {r['heard']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", type=Path, required=True)
    ap.add_argument("--texts", type=Path, required=True)
    ap.add_argument("--lang", default="ta")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    key = load_key("OPEN_AI_API")
    texts = {p.stem: p.read_text(encoding="utf-8").strip() for p in args.texts.rglob("part_*.txt")}
    wavs = sorted(args.wavs.rglob("part_*.wav"), key=lambda p: p.stem)
    results = [check_part(key, w, texts[w.stem], args.lang) for w in wavs if w.stem in texts]
    if not results:
        raise SystemExit("no part_XX.wav with matching part_XX.txt found")
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(markdown(results))


if __name__ == "__main__":
    main()
