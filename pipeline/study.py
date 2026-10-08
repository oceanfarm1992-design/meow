"""Free frame study: motion + CLIP labels (+ audio) -> timeline.json + contact sheet.

Usage: python pipeline/study.py inbox/clip.mp4 --out out/
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from clip_labels import ClipLabeller
from frames import audio_loudness, extract_frames, motion_scores, probe

log = logging.getLogger("study")

SMOOTH_WINDOW = 5         # frames (2.5 s) for majority-vote label smoothing
MIN_EVENT_SECONDS = 1.5   # shorter label runs are merged into neighbours
SHEET_COLS = 6
SHEET_MAX_TILES = 24


def smooth(labels: list[str], window: int = SMOOTH_WINDOW) -> list[str]:
    half = window // 2
    return [
        Counter(labels[max(0, i - half): i + half + 1]).most_common(1)[0][0]
        for i in range(len(labels))
    ]


def build_events(records: list[dict], step: float) -> list[dict]:
    """Group consecutive frames with the same smoothed action + mood into events."""
    events: list[dict] = []
    for r in records:
        key = (r["action_smooth"], r["mood_smooth"])
        if events and events[-1]["key"] == key:
            events[-1]["frames"].append(r)
        else:
            events.append({"key": key, "frames": [r]})

    # absorb too-short runs into the previous event
    merged: list[dict] = []
    for e in events:
        if merged and (len(e["frames"]) * step < MIN_EVENT_SECONDS or merged[-1]["key"] == e["key"]):
            merged[-1]["frames"].extend(e["frames"])
        else:
            merged.append(e)

    out = []
    for e in merged:
        fr = e["frames"]
        motion = [f["motion"] for f in fr]
        out.append({
            "start": fr[0]["t"],
            "end": round(fr[-1]["t"] + step, 2),
            "action": e["key"][0],
            "mood": e["key"][1],
            "avg_motion": round(sum(motion) / len(motion), 3),
            "peak_motion": max(motion),
            "peak_t": fr[motion.index(max(motion))]["t"],
        })
    return out


def contact_sheet(records: list[dict], dest: Path) -> None:
    step = max(1, len(records) // SHEET_MAX_TILES)
    picks = records[::step][:SHEET_MAX_TILES]
    tw, th = Image.open(picks[0]["path"]).size
    label_h = 34
    rows = (len(picks) + SHEET_COLS - 1) // SHEET_COLS
    sheet = Image.new("RGB", (tw * SHEET_COLS, (th + label_h) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(picks):
        x, y = (i % SHEET_COLS) * tw, (i // SHEET_COLS) * (th + label_h)
        sheet.paste(Image.open(r["path"]), (x, y))
        draw.text((x + 4, y + th + 2), f"{r['t']:.1f}s  {r['action_smooth']}", fill="black")
        draw.text((x + 4, y + th + 17), f"{r['mood']}  motion {r['motion']:.2f}", fill="dimgray")
    sheet.save(dest, quality=85)


def summary_md(video: Path, info: dict, events: list[dict]) -> str:
    lines = [
        f"## Frame study: `{video.name}`",
        f"Duration {info['duration']:.1f}s · audio: {'yes' if info['has_audio'] else 'no'}",
        "",
        "| Time | Action | Mood | Avg motion | Peak |",
        "|---|---|---|---|---|",
    ]
    for e in events:
        bar = "█" * round(e["avg_motion"] * 10)
        lines.append(f"| {e['start']:.1f}–{e['end']:.1f}s | {e['action']} | {e['mood']} "
                     f"| {bar} {e['avg_motion']:.2f} | {e['peak_motion']:.2f} @ {e['peak_t']}s |")
    return "\n".join(lines) + "\n"


def study(video: Path, out: Path) -> dict:
    info = probe(video)
    log.info("probe: %.1fs, audio=%s", info["duration"], info["has_audio"])

    frames = extract_frames(video, out / "frames")
    log.info("extracted %d frames", len(frames))
    motion = motion_scores(frames)
    labels = ClipLabeller().label([f.path for f in frames])
    loud = audio_loudness(video, [f.t for f in frames]) if info["has_audio"] else None

    act_s = smooth([lab["action"] for lab in labels])
    mood_s = smooth([lab["mood"] for lab in labels])
    records = [
        {"t": f.t, "path": str(f.path), "motion": m, **lab,
         "action_smooth": act_s[i], "mood_smooth": mood_s[i],
         "loudness": None if loud is None else loud[i]}
        for i, (f, m, lab) in enumerate(zip(frames, motion, labels))
    ]
    step = frames[1].t - frames[0].t if len(frames) > 1 else 0.5
    events = build_events(records, step)

    result = {"video": video.name, **info, "events": events,
              "frames": [{k: v for k, v in r.items() if k != "path"} for r in records]}
    (out / "timeline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    contact_sheet(records, out / "contact_sheet.jpg")
    (out / "summary.md").write_text(summary_md(video, info, events), encoding="utf-8")
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()
    out = args.out / args.video.stem
    out.mkdir(parents=True, exist_ok=True)
    study(args.video, out)
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
