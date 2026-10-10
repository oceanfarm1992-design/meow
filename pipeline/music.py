"""Generate a light, bouncy comedy music bed (no samples, so no copyright issues).

Plucked bass + melody (Karplus-Strong strings) and soft hi-hat ticks on a
I-V-vi-IV progression. Usage:
    python pipeline/music.py --seconds 60 --out music.wav [--seed 7]
"""
import argparse
import wave
from pathlib import Path

import numpy as np

SR = 44100
BPM = 116
BEAT = 60 / BPM
# C major: I - V - vi - IV, one chord per bar (4 beats); MIDI root notes
PROGRESSION = [48, 43, 45, 41]
CHORD_TONES = {48: [60, 64, 67], 43: [59, 62, 67], 45: [57, 60, 64], 41: [57, 60, 65]}
PENTATONIC = [60, 62, 64, 67, 69, 72, 74, 76]


def midi_hz(n: int) -> float:
    return 440.0 * 2 ** ((n - 69) / 12)


def pluck(freq: float, seconds: float, rng: np.random.Generator, damp: float = 0.996) -> np.ndarray:
    """Karplus-Strong plucked string."""
    n = int(seconds * SR)
    period = max(2, int(SR / freq))
    buf = rng.uniform(-1, 1, period)
    out = np.empty(n)
    for i in range(n):
        out[i] = buf[i % period]
        buf[i % period] = damp * 0.5 * (buf[i % period] + buf[(i + 1) % period])
    return out * np.linspace(1, 0.2, n)


def hat(seconds: float, rng: np.random.Generator) -> np.ndarray:
    n = int(seconds * SR)
    noise = np.diff(rng.uniform(-1, 1, n + 1))  # crude high-pass
    return noise * np.exp(-np.linspace(0, 40, n))


def add(track: np.ndarray, sound: np.ndarray, at: float, gain: float) -> None:
    start = int(at * SR)
    end = min(len(track), start + len(sound))
    if start < len(track):
        track[start:end] += gain * sound[: end - start]


def render(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    track = np.zeros(int(seconds * SR) + SR)
    beat, bar = 0, 0
    while beat * BEAT < seconds:
        t = beat * BEAT
        root = PROGRESSION[bar % len(PROGRESSION)]
        if beat % 4 in (0, 2):  # bouncy bass on 1 and 3, fifth on the "and"
            add(track, pluck(midi_hz(root), BEAT * 0.9, rng, 0.994), t, 0.55)
            add(track, pluck(midi_hz(root + 7), BEAT * 0.4, rng, 0.99), t + BEAT / 2, 0.3)
        add(track, hat(0.05, rng), t + BEAT / 2, 0.08)
        tones = CHORD_TONES[root]
        for step in range(2):  # two melody notes per beat, some rests
            if rng.random() < 0.6:
                note = rng.choice(tones if step == 0 else PENTATONIC)
                add(track, pluck(midi_hz(int(note)), BEAT * 0.6, rng), t + step * BEAT / 2, 0.22)
        beat += 1
        if beat % 4 == 0:
            bar += 1
    track = track[: int(seconds * SR)]
    fade = int(1.5 * SR)
    track[-fade:] *= np.linspace(1, 0, fade)
    return track / (np.abs(track).max() or 1.0) * 0.8


def write_wav(path: Path, mono: np.ndarray) -> None:
    pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
    stereo = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    write_wav(args.out, render(args.seconds, args.seed))


if __name__ == "__main__":
    main()
