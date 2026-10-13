"""Build a told-story montage from a story plan: narrate each scene, fetch a stock
clip for it and cut the clip to exactly the length of its narration.

Usage:
    python pipeline/story_build.py story/story.json --voice karees_mass_hero_ta \
        --tempo 0.9 --out story/
Writes <out>/montage.mp4 (silent), <out>/voice.wav (in sync), <out>/parts, <out>/texts,
<out>/credits.txt. Then render.py adds music + banner (use --no-tighten).
Keys: ELEVENLAB_API, PEXELS_API.
"""
import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

from clip_rank import ClipRanker
from fetch_pexels import download, search
from keys import load_key
from tts_elevenlabs import SR, load_voice, read_pcm, synth_checked

ROOT = Path(__file__).resolve().parent.parent
WIDTH, HEIGHT, FPS = 720, 1280, 30
SCENE_GAP = 0.35        # breath after each scene's narration (s)
MIN_SCENE = 2.5         # shortest scene on screen (s)
CLIP_OFFSET = 1.0       # skip the first second of stock clips (often a slow start)
CANDIDATES = 10         # search results compared visually per scene
PORTRAIT_BONUS = 0.01   # CLIP cosine scores sit around 0.2-0.35; small nudges only
SHORT_PENALTY = 0.01    # looping a short clip is fine but less nice
HERO = "black cat"      # story_plan keeps the hero's shots to black cats


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def seconds(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def narrate(key: str, voice: dict, text: str, tempo: float, dest: Path) -> tuple[float, int]:
    raw = dest.with_name(dest.stem + ".raw.wav")
    tries = synth_checked(key, voice, text, raw)
    run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af", f"atempo={tempo}",
         "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(dest)])
    raw.unlink()
    return seconds(dest), tries


def pick_clip(key: str, ranker: ClipRanker, query: str, need: float, used: set[int]) -> tuple[dict, float]:
    """Best visual match (CLIP) among the top results, preferring portrait clips that are long
    enough; Pexels' own ranking is loose (a 'black cat jumping' search returned a white cat)."""
    found = [v for v in search(key, query) if v["id"] not in used][:CANDIDATES]
    subject = HERO if HERO in query.lower() else None
    if subject:  # also look at plain hero shots: a black cat sitting beats a white cat jumping
        seen = {v["id"] for v in found}
        found += [v for v in search(key, subject) if v["id"] not in used | seen][:CANDIDATES // 2]
    if not found:
        raise SystemExit(f"no Pexels clips for {query!r}")
    scores = ranker.score(query, found)
    if subject:  # the hero must look right (colour), not just the action
        hero = ranker.score(subject, found)
        scores = [(s + h) / 2 for s, h in zip(scores, hero)]

    def rank(pair: tuple[dict, float]) -> float:
        v, s = pair
        portrait = PORTRAIT_BONUS if v["height"] > v["width"] else 0.0
        too_short = SHORT_PENALTY if v["duration"] < need + CLIP_OFFSET else 0.0
        return s + portrait - too_short

    return max(zip(found, scores), key=rank)


def cut_scene(clip: Path, need: float, dest: Path) -> None:
    length = seconds(clip)
    offset = min(CLIP_OFFSET, max(0.0, length - need))
    loop = ["-stream_loop", "-1"] if length - offset < need else []
    run(["ffmpeg", "-v", "error", "-y", *loop, "-ss", f"{offset:.2f}", "-t", f"{need:.3f}",
         "-i", str(clip), "-an",
         "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={WIDTH}:{HEIGHT},setsar=1,fps={FPS}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(dest)])


def write_voice(parts: list[tuple[Path, float]], dest: Path) -> None:
    """Each scene's narration padded with silence to its scene length."""
    chunks = []
    for wav, scene_len in parts:
        pcm = read_pcm(wav)
        chunks += [pcm, np.zeros(max(0, round(scene_len * SR) - len(pcm)), dtype=np.int16)]
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.concatenate(chunks).tobytes())


def build(plan: dict, voice_name: str, tempo: float, out: Path) -> None:
    voice, el_key, px_key = load_voice(voice_name), load_key("ELEVENLAB_API"), load_key("PEXELS_API")
    for sub in ("parts", "texts", "scenes"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ranker = ClipRanker()
    used: set[int] = set()
    voice_parts, scene_files, credits = [], [], []
    for i, scene in enumerate(plan["scenes"]):
        tagged = (f"[{scene['tag']}] " if scene["tag"] != "none" else "") + scene["text"].strip()
        wav = out / "parts" / f"part_{i:02d}.wav"
        said = out / "texts" / f"part_{i:02d}.txt"
        if wav.exists() and said.exists() and said.read_text(encoding="utf-8") == scene["text"].strip():
            dur, tries = seconds(wav), 0  # re-run: reuse narration, no credits spent
        else:
            dur, tries = narrate(el_key, voice, tagged, tempo, wav)
            said.write_text(scene["text"].strip(), encoding="utf-8")
        need = max(MIN_SCENE, dur + SCENE_GAP)

        video, match = pick_clip(px_key, ranker, scene["query"], need, used)
        used.add(video["id"])
        clip = download(video, ROOT / "inbox")
        meta = json.loads(clip.with_suffix(".json").read_text(encoding="utf-8"))
        credits.append(f"{meta['credit']} ({meta['page']})")

        scene_mp4 = out / "scenes" / f"scene_{i:02d}.mp4"
        cut_scene(clip, need, scene_mp4)
        voice_parts.append((wav, need))
        scene_files.append(scene_mp4)
        print(f"scene {i + 1}: {need:4.1f}s  voice {dur:4.1f}s (tries {tries})  "
              f"clip {video['id']} '{scene['query']}' match {match:.3f}  "
              f"engine=elevenlabs:{voice['model_id']}")

    concat = out / "scenes" / "list.txt"
    concat.write_text("".join(f"file '{p.name}'\n" for p in scene_files), encoding="utf-8")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-c", "copy", str(out / "montage.mp4")])
    write_voice(voice_parts, out / "voice.wav")
    (out / "credits.txt").write_text("\n".join(credits) + "\n", encoding="utf-8")
    total = sum(n for _, n in voice_parts)
    print(f"montage {total:.1f}s from {len(scene_files)} scenes -> {out / 'montage.mp4'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    ap.add_argument("--voice", default="karees_mass_hero_ta")
    ap.add_argument("--tempo", type=float, default=0.9)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    build(json.loads(args.plan.read_text(encoding="utf-8")), args.voice, args.tempo, args.out)


if __name__ == "__main__":
    main()
