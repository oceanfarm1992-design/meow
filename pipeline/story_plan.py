"""Plan a told-story Short: scenes with narration + a stock-footage search for each.

Usage:
    python pipeline/story_plan.py --theme "Karees steals fried fish" --lang ta \
        --seconds 45 --out story/story.json
Without --theme the model invents one. Key: OPEN_AI_API.
"""
import argparse
import json
import logging
import re
from pathlib import Path

from keys import load_key
from write_script import ADDRESS, LANG_RULES, MODEL, TAGS, ask_schema

log = logging.getLogger("story_plan")

CHARS_PER_SECOND = {"ta": 12.0, "en": 11.0}   # narration budget incl. scene gaps
BUDGET_TOLERANCE = 0.15
SECONDS_PER_SCENE = (10, 6)   # scene count scales with length: 45 s -> 4-7, 5 min -> 30-50
SCENE_CHARS = (40, 140)
MAX_ATTEMPTS = 3

STYLE = {
    "ta": ("Tell it like a Tamil comedy storyteller on YouTube Shorts: a dramatic 'ஒரு ஊர்ல' style "
           "narrator with mass-hero build-ups, cheeky asides and a big over-the-top punchline. "
           "The narrator may quote the cat's lines (the cat calls its owner {address})."),
    "en": ("Tell it like a funny YouTube Shorts storyteller: dramatic build-ups, cheeky asides and "
           "a big over-the-top punchline. The narrator may quote the cat's lines (calls owner {address})."),
}

SCHEMA = {
    "name": "story",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "scenes"],
        "properties": {
            "title": {"type": "string", "description": "short banner title, English capitals"},
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["query", "text", "tag"],
                    "properties": {
                        "query": {"type": "string", "description": "stock video search, English"},
                        "text": {"type": "string", "description": "narration for this scene"},
                        "tag": {"type": "string", "enum": TAGS},
                    },
                },
            },
        },
    },
}


def scene_range(seconds: float) -> tuple[int, int]:
    return max(4, round(seconds / SECONDS_PER_SCENE[0])), max(7, round(seconds / SECONDS_PER_SCENE[1]))


def build_prompt(theme: str, lang: str, name: str, budget: int, scenes: tuple[int, int]) -> str:
    about = f"Story idea: {theme}" if theme else "Invent a funny everyday story idea."
    if scenes[1] > 12:
        about += (" This is a long story: give it a clear beginning, several escalating mini-adventures"
                  " with running jokes and callbacks, and a big finale.")
    return f"""Write a short told story for a vertical animal video. The hero is a black cat named {name}.
{about}
{STYLE[lang].format(address=ADDRESS[lang])}
{LANG_RULES[lang]}

Rules:
- {scenes[0]} to {scenes[1]} scenes, about {budget} narration characters in total.
- Each scene's text is 1 or 2 sentences, {SCENE_CHARS[0]} to {SCENE_CHARS[1]} characters, each sentence ending with . ! or ?
- Numbers written as words, no digits, no emojis, no quotes marks, no stage directions.
- Each scene has a stock footage search query in simple English (2 to 5 words) that a free stock
  site will surely have, e.g. "black cat walking", "fish frying in pan", "cat looking up". Keep the
  cat shots to black cats and keep queries visually concrete.
- Give each scene a delivery tag (vary them; 'excited' for the punchline).
- title: two or three English words in capitals for the banner, e.g. "KAREES STORIES"."""


def validate(plan: dict, lang: str, budget: int, limits: tuple[int, int]) -> list[str]:
    problems = []
    scenes = plan["scenes"]
    if not limits[0] <= len(scenes) <= limits[1]:
        problems.append(f"{len(scenes)} scenes, needs {limits[0]} to {limits[1]}")
    total = sum(len(s["text"]) for s in scenes)
    if abs(total - budget) > budget * BUDGET_TOLERANCE:
        problems.append(f"narration is {total} characters, needs about {budget}")
    for i, s in enumerate(scenes, 1):
        t = s["text"].strip()
        if not SCENE_CHARS[0] <= len(t) <= SCENE_CHARS[1]:
            problems.append(f"scene {i} text is {len(t)} characters, needs {SCENE_CHARS[0]} to {SCENE_CHARS[1]}")
        if not re.search(r"[.!?]$", t):
            problems.append(f"scene {i} text must end with . ! or ?")
        if re.search(r"\d", t):
            problems.append(f"scene {i} contains digits")
        if lang == "ta" and re.search(r"[A-Za-z]", t):
            problems.append(f"scene {i} text contains Latin letters")
        if not re.fullmatch(r"[A-Za-z ]{3,40}", s["query"].strip()):
            problems.append(f"scene {i} query must be 2 to 5 simple English words")
    return problems


def plan_story(theme: str, lang: str, name: str, seconds: float) -> dict:
    budget = round(seconds * CHARS_PER_SECOND[lang])
    limits = scene_range(seconds)
    key = load_key("OPEN_AI_API")
    messages = [{"role": "user", "content": build_prompt(theme, lang, name, budget, limits)}]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        plan = ask_schema(key, messages, SCHEMA)
        problems = validate(plan, lang, budget, limits)
        log.info("attempt %d (%s): %d scenes, %d chars, problems: %s", attempt, MODEL,
                 len(plan["scenes"]), sum(len(s["text"]) for s in plan["scenes"]), problems or "none")
        if not problems:
            return plan
        messages += [{"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
                     {"role": "user", "content": "Fix these problems and return the full story again:\n- "
                      + "\n- ".join(problems)}]
    raise SystemExit(f"story still invalid after {MAX_ATTEMPTS} attempts: {problems}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="")
    ap.add_argument("--lang", default="ta", choices=sorted(LANG_RULES))
    ap.add_argument("--name", default="Karees")
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plan = plan_story(args.theme, args.lang, args.name, args.seconds)
    plan = {**plan, "lang": args.lang, "model": MODEL}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("title: %s", plan["title"])
    for i, s in enumerate(plan["scenes"], 1):
        log.info("%d. [%s] %s  | %s", i, s["tag"], s["text"], s["query"])


if __name__ == "__main__":
    main()
