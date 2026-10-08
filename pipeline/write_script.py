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
# Measured on the cloned Tamil voice: ~13.6 chars/s speech + ~0.5 s between sentences.
CHARS_PER_SECOND = {"ta": 11.0, "en": 13.0}  # overall rate incl. pauses
BUDGET_TOLERANCE = 0.12
MIN_LINE_CHARS = 20  # very short sentences come out garbled in the cloned voice
MAX_LINE_CHARS = 70
MAX_ATTEMPTS = 3

LANG_RULES = {
    "ta": (
        "Write in natural spoken Tamil (Tamil script only), like a funny Tamil YouTube voiceover, "
        "but keep words clear and easy for a text-to-speech model: avoid heavy slang contractions "
        "and English loanwords written in Tamil (no ஹலோ, அட்டாக், சிப்ஸ் etc.)."
    ),
    "en": "Write in casual, punchy English.",
}

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
                    "required": ["at", "text"],
                    "properties": {
                        "at": {"type": "number", "description": "start time in seconds"},
                        "text": {"type": "string"},
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


def build_prompt(tl: dict, meta: dict | None, lang: str, style: str, budget: int) -> str:
    return f"""{describe_timeline(tl, meta)}

Write the cat's secret inner voice (first person, the cat talking in its head about its owner and what is happening), style: {style}.
{LANG_RULES[lang]}

Rules:
- About {budget} characters in total (spaces included), split into 9 to 13 short sentences.
- Each line {MIN_LINE_CHARS} to {MAX_LINE_CHARS} characters (no tiny one-word lines) and ends with . ! or ?
- Numbers written as words, no digits, no emojis, no quotes, no stage directions.
- Give each line an "at" time (seconds) so jokes land on the matching moment in the timeline; lines in time order, first line near 0, last line starting before {tl['duration'] - 4:.0f}s.
- Make it genuinely funny: a clear mini story with escalation and a punchline at the end."""


def validate(lines: list[dict], lang: str, budget: int) -> list[str]:
    problems = []
    total = sum(len(l["text"]) for l in lines)
    if abs(total - budget) > budget * BUDGET_TOLERANCE:
        problems.append(f"total is {total} characters, needs about {budget}")
    for i, l in enumerate(lines, 1):
        t = l["text"].strip()
        if not re.search(r"[.!?।]$", t):
            problems.append(f"line {i} must end with . ! or ?")
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


def write_script(timeline: Path, lang: str, style: str) -> list[dict]:
    tl = json.loads(timeline.read_text(encoding="utf-8"))
    meta_path = timeline.parents[2] / "inbox" / f"{tl['video'].rsplit('.', 1)[0]}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    budget = round(tl["duration"] * CHARS_PER_SECOND[lang])

    key = load_key("OPEN_AI_API")
    messages = [{"role": "user", "content": build_prompt(tl, meta, lang, style, budget)}]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        lines = ask(key, messages)
        problems = validate(lines, lang, budget)
        log.info("attempt %d (%s): %d lines, %d chars, problems: %s", attempt, MODEL,
                 len(lines), sum(len(l["text"]) for l in lines), problems or "none")
        if not problems:
            return lines
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
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lines = write_script(args.timeline, args.lang, args.style)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(l["text"].strip() for l in lines) + "\n", encoding="utf-8", newline="\n")
    args.out.with_suffix(args.out.suffix + ".json").write_text(
        json.dumps({"model": MODEL, "lines": lines}, ensure_ascii=False, indent=2), encoding="utf-8")
    for l in lines:
        log.info("%5.1fs  %s", l["at"], l["text"])


if __name__ == "__main__":
    main()
