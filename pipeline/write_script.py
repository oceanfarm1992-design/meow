"""Write the cat's inner-voice script from a frame-study timeline (OpenAI).

Usage:
    python pipeline/write_script.py out/<clip>/timeline.json --lang ta --out scripts/<clip>.ta.txt

Writes <out> (one sentence per line, the text sent to the voice engine) and
<out>.json (lines with their target start times). Key: OPEN_AI_API.
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path

import requests

from keys import load_key

log = logging.getLogger("write_script")

CHAT_URL = "https://api.openai.com/v1/chat/completions"
MODEL = os.environ.get("OPENAI_SCRIPT_MODEL", "gpt-5.5")
# Script length budget per second of clip. Measured: Tamil ~11 chars/s and English ~9.9
# with pauses; Tamil raised to 12 since render.py shortens pauses (and may slow the voice).
CHARS_PER_SECOND = {"ta": 12.0, "en": 9.5}
BUDGET_TOLERANCE = 0.12
MIN_LINE_CHARS = 25  # the voice engine splits on . ! ? and very short parts come out garbled
MAX_LINE_CHARS = 90
MAX_ATTEMPTS = 3

LANG_RULES = {
    "ta": (
        "Write in colloquial spoken Tamil (Tamil script only), like Tamil comedy dubbing on YouTube "
        "Shorts. Tanglish is natural: everyday English words written in Tamil script are welcome "
        "(செகண்ட், டேஞ்சரஸ், சிம்பிள், அசால்ட்டா), and so are spoken endings like டா, டி, ப்பா."
    ),
    "en": "Write in casual everyday English, the way a real person talks to themselves, with contractions (I'm, don't, that's).",
}

# Extra direction per (language, style); learned from the owner's own comedy dubbing.
STYLE_RULES = {
    ("ta", "comedy"): (
        "Style: Tamil movie mass-hero punch-dialogue parody. The cat talks big and dramatic, "
        "boasts about its 'weapons' and power, builds up, and lands a huge over-the-top "
        "exaggeration as the punchline (like 'even ten thousand lions, one second, assault-a')."
    ),
}
ADDRESS = {"ta": "மம்மி", "en": "Mom"}
# eleven_v3 delivery tags; "none" when the voice engine has no tag support
TAGS = ["none", "excited", "dramatic", "proud", "sarcastic", "laughs", "whispers", "shouting", "curious"]

SCHEMA = {
    "name": "cat_script",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["lines"],
        "properties": {
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["at", "text", "tag"],
                    "properties": {
                        "at": {"type": "number", "description": "start time in seconds"},
                        "text": {"type": "string"},
                        "tag": {"type": "string", "enum": TAGS, "description": "delivery tag"},
                    },
                },
            }
        },
    },
}


def describe_timeline(tl: dict, meta: dict | None) -> str:
    rows = [f"- {e['start']:.1f}s to {e['end']:.1f}s: {e['action']}, looks {e['mood']}, "
            f"motion {e['avg_motion']:.2f} (peak {e['peak_motion']:.2f} at {e['peak_t']}s)"
            for e in tl["events"]]
    title = ""
    if meta:
        title = meta["page"].rstrip("/").split("/")[-1].rsplit("-", 1)[0].replace("-", " ")
    return (f"Clip: {title or tl['video']} ({tl['duration']:.1f}s, no dialogue).\n"
            "Frame-study timeline (automatic, may be imperfect):\n" + "\n".join(rows))


def build_prompt(tl: dict, meta: dict | None, lang: str, style: str, budget: int, name: str,
                 tags: bool) -> str:
    tag_rule = ("- Give each line a delivery tag that fits the moment (vary them, use 'dramatic' or "
                "'proud' for build-ups and 'excited' or 'shouting' for the attack/punchline)."
                if tags else "- Set every tag to none.")
    return f"""{describe_timeline(tl, meta)}

Write a voice-over where the cat, named {name}, talks out loud to its owner (calls them {ADDRESS[lang]}) about what is happening. Style: {style}.
{LANG_RULES[lang]}
{STYLE_RULES.get((lang, style), "")}

Rules:
- About {budget} characters in total (spaces included), split into 6 to 10 sentences.
- Exactly one sentence per line, {MIN_LINE_CHARS} to {MAX_LINE_CHARS} characters, ending with . ! or ? (no extra . ! ? inside a line, no semicolons).
- Numbers written as words, no digits, no emojis, no quotes, no stage directions.
- Give each line an "at" time (seconds) so jokes land on the matching moment in the timeline; lines in time order, first line near 0, last line starting before {tl['duration'] - 4:.0f}s.
- It must sound natural when spoken aloud: simple everyday words and normal phrasing, like a real thought. No strange metaphors, made-up phrases or forced wordplay; the humour comes from the cat's attitude and what is really happening.
- A clear mini story with escalation and a funny ending line.
{tag_rule}"""


def validate(lines: list[dict], lang: str, budget: int) -> list[str]:
    problems = []
    total = sum(len(l["text"]) for l in lines)
    if abs(total - budget) > budget * BUDGET_TOLERANCE:
        problems.append(f"total is {total} characters, needs about {budget}")
    for i, l in enumerate(lines, 1):
        t = l["text"].strip()
        if not re.search(r"[.!?।]$", t):
            problems.append(f"line {i} must end with . ! or ?")
        if re.search(r"[.!?।;]", t[:-1]):
            problems.append(f"line {i} must be one sentence (no . ! ? ; before the end)")
        if not MIN_LINE_CHARS <= len(t) <= MAX_LINE_CHARS:
            problems.append(f"line {i} is {len(t)} characters, needs {MIN_LINE_CHARS} to {MAX_LINE_CHARS}")
        if re.search(r"\d", t):
            problems.append(f"line {i} contains digits")
        if lang == "ta" and re.search(r"[A-Za-z]", t):
            problems.append(f"line {i} contains Latin letters")
    return problems


def ask(key: str, messages: list[dict]) -> list[dict]:
    resp = requests.post(
        CHAT_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": MODEL, "messages": messages,
              "response_format": {"type": "json_schema", "json_schema": SCHEMA}},
        timeout=180,
    )
    if not resp.ok:
        raise RuntimeError(f"OpenAI {resp.status_code}: {resp.text[:300]}")
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)["lines"]


def pick_window(tl: dict, length: float) -> tuple[float, dict]:
    """Choose the liveliest `length`-second stretch; returns (start, timeline cut to it)."""
    if length <= 0 or length >= tl["duration"] - 1:
        return 0.0, tl
    frames = tl["frames"]
    best_start, best_score = 0.0, -1.0
    for f in frames:
        if f["t"] + length > tl["duration"]:
            break
        inside = [g["motion"] for g in frames if f["t"] <= g["t"] < f["t"] + length]
        score = sum(inside) / len(inside)
        if score > best_score:
            best_start, best_score = f["t"], score
    start = round(best_start, 1)
    end = start + length
    events = []
    for e in tl["events"]:
        if e["end"] <= start or e["start"] >= end:
            continue
        events.append({**e, "start": round(max(e["start"], start) - start, 2),
                       "end": round(min(e["end"], end) - start, 2),
                       "peak_t": round(min(max(e["peak_t"], start), end) - start, 2)})
    return start, {**tl, "duration": length, "events": events}


def write_script(timeline: Path, lang: str, style: str, name: str, window: float,
                 tags: bool = False) -> tuple[float, float, list[dict]]:
    full = json.loads(timeline.read_text(encoding="utf-8"))
    start, tl = pick_window(full, window)
    meta_path = timeline.parents[2] / "inbox" / f"{tl['video'].rsplit('.', 1)[0]}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    budget = round(tl["duration"] * CHARS_PER_SECOND[lang])
    log.info("window %.1fs-%.1fs, budget %d chars", start, start + tl["duration"], budget)

    key = load_key("OPEN_AI_API")
    messages = [{"role": "user", "content": build_prompt(tl, meta, lang, style, budget, name, tags)}]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        lines = ask(key, messages)
        problems = validate(lines, lang, budget)
        log.info("attempt %d (%s): %d lines, %d chars, problems: %s", attempt, MODEL,
                 len(lines), sum(len(l["text"]) for l in lines), problems or "none")
        if not problems:
            return start, tl["duration"], lines
        messages += [{"role": "assistant", "content": json.dumps({"lines": lines}, ensure_ascii=False)},
                     {"role": "user", "content": "Fix these problems and return the full script again:\n- "
                      + "\n- ".join(problems)}]
    raise SystemExit(f"script still invalid after {MAX_ATTEMPTS} attempts: {problems}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline", type=Path)
    ap.add_argument("--lang", default="ta", choices=sorted(LANG_RULES))
    ap.add_argument("--style", default="comedy")
    ap.add_argument("--name", default="Karees", help="the cat's name")
    ap.add_argument("--window", type=float, default=0, help="seconds of clip to use; 0 = all")
    ap.add_argument("--tags", action="store_true", help="add eleven_v3 delivery tags (written to <out>.tagged)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    start, length, lines = write_script(args.timeline, args.lang, args.style, args.name,
                                        args.window, args.tags)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(l["text"].strip() for l in lines) + "\n", encoding="utf-8", newline="\n")
    tagged = [(f"[{l['tag']}] " if l["tag"] != "none" else "") + l["text"].strip() for l in lines]
    args.out.with_suffix(args.out.suffix + ".tagged").write_text(
        "\n".join(tagged) + "\n", encoding="utf-8", newline="\n")
    args.out.with_suffix(args.out.suffix + ".json").write_text(
        json.dumps({"model": MODEL, "start": start, "length": length, "lines": lines},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    for l in lines:
        log.info("%5.1fs  %-10s %s", l["at"], l["tag"], l["text"])


if __name__ == "__main__":
    main()
