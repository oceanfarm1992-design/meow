"""Generate a comedy music bed in code (no samples, so no copyright issues).

Styles:
  bouncy  cartoon oom-pah: tuba bass on 1 and 3, marimba chord stabs on 2 and 4,
          swung xylophone melody, woodblock ticks, slide-whistle every 8 bars
  sneaky  tiptoe heist: minor key, creeping chromatic pizzicato bass, staccato
          marimba hits with grace notes, soft woodblock, slide-whistle every 8 bars

Usage:
    python pipeline/music.py --style bouncy --seconds 60 --out music.wav [--seed 7]
"""
import argparse
import wave
from pathlib import Path

import numpy as np

SR = 44100
SWING = 0.62            # off-beat eighths land at 62% of the beat (swing feel)
STYLES = {
    # tempo, bass pattern per bar (MIDI, None = rest), chord per bar, melody scale
    "bouncy": {
        "bpm": 132,
        "bars": [[36, None, 43, None], [41, None, 36, None], [43, None, 38, None], [36, None, 43, None]],
        "chords": [[60, 64, 67], [60, 65, 69], [59, 62, 67], [60, 64, 67]],
        "scale": [72, 74, 76, 79, 81, 84],
    },
    "sneaky": {
        "bpm": 104,
        "bars": [[45, 48, 50, 51], [52, None, 40, None], [45, 48, 50, 51], [52, 51, 50, 48]],
        "chords": [[57, 60, 64], [56, 59, 64], [57, 60, 64], [56, 59, 62]],
        "scale": [69, 71, 72, 74, 75, 76],
    },
}


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def marimba(midi: float, seconds: float = 0.5) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f = hz(midi)
    tone = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-t / 0.03)
    return tone * env(n, 0.002, 0.18)


def tuba(midi: float, seconds: float = 0.28) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f = hz(midi)
    tone = sum(np.sin(2 * np.pi * k * f * t) / k ** 1.2 for k in range(1, 6))  # brassy saw-ish
    return np.tanh(1.6 * tone) * env(n, 0.02, 0.16)


def pizz(midi: float, rng: np.random.Generator, seconds: float = 0.25) -> np.ndarray:
    """Karplus-Strong pluck, heavily damped = pizzicato."""
    n = int(seconds * SR)
    period = max(2, int(SR / hz(midi)))
    buf = rng.uniform(-1, 1, period)
    out = np.empty(n)
    for i in range(n):
        out[i] = buf[i % period]
        buf[i % period] = 0.985 * 0.5 * (buf[i % period] + buf[(i + 1) % period])
    return out * env(n, 0.001, 0.09)


def woodblock(rng: np.random.Generator) -> np.ndarray:
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1250 * t) + 0.3 * rng.uniform(-1, 1, n)) * env(n, 0.0005, 0.012)


def slide_whistle(up: bool, seconds: float = 0.6) -> np.ndarray:
    n = int(seconds * SR)
    lo, hi = 900.0, 2200.0
    sweep = np.linspace(lo, hi, n) if up else np.linspace(hi, lo, n)
    phase = 2 * np.pi * np.cumsum(sweep) / SR
    vibrato = 1 + 0.02 * np.sin(2 * np.pi * 6 * np.arange(n) / SR)
    return np.sin(phase * vibrato) * env(n, 0.03, 0.5) * np.linspace(1, 0.6, n)


def add(track: np.ndarray, sound: np.ndarray, at: float, gain: float) -> None:
    start = int(at * SR)
    end = min(len(track), start + len(sound))
    if 0 <= start < len(track):
        track[start:end] += gain * sound[: end - start]


def render(style: str, seconds: float, seed: int) -> np.ndarray:
    cfg, rng = STYLES[style], np.random.default_rng(seed)
    beat = 60 / cfg["bpm"]
    track = np.zeros(int(seconds * SR) + SR)
    bar = 0
    while bar * 4 * beat < seconds:
        t0 = bar * 4 * beat
        bass, chord = cfg["bars"][bar % 4], cfg["chords"][bar % 4]
        for b in range(4):
            t = t0 + b * beat
            if bass[b] is not None:
                if style == "bouncy":
                    add(track, tuba(bass[b]), t, 0.55)
                else:
                    add(track, pizz(bass[b], rng), t, 0.7)
            if style == "bouncy" and b in (1, 3):  # oom-PAH chord stabs
                for note in chord:
                    add(track, marimba(note, 0.25), t, 0.16)
            if style == "sneaky" and b in (0, 2) and rng.random() < 0.7:  # tiptoe stab with grace note
                top = chord[-1] + 12
                add(track, marimba(top - 1, 0.1), t - 0.06, 0.12)
                add(track, marimba(top, 0.2), t, 0.22)
            add(track, woodblock(rng), t + SWING * beat, 0.10 if style == "bouncy" else 0.06)
            for half in (0.0, SWING):  # swung melody, with rests
                if rng.random() < (0.55 if style == "bouncy" else 0.25):
                    note = rng.choice(cfg["scale"])
                    add(track, marimba(int(note), 0.3), t + half * beat, 0.22)
        if bar % 8 == 7:  # comic accent at the end of every 8 bars
            add(track, slide_whistle(up=bool(rng.random() < 0.5)), t0 + 3 * beat, 0.18)
        bar += 1
    track = track[: int(seconds * SR)]
    fade = min(len(track), int(1.5 * SR))
    track[-fade:] *= np.linspace(1, 0, fade)
    return track / (np.abs(track).max() or 1.0) * 0.85


def write_wav(path: Path, mono: np.ndarray) -> None:
    pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="bouncy", choices=sorted(STYLES))
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    write_wav(args.out, render(args.style, args.seconds, args.seed))


if __name__ == "__main__":
    main()
