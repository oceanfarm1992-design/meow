"""Search Pexels for short cat videos and download one into inbox/.

Usage:
    python pipeline/fetch_pexels.py "angry cat"                 # list candidates
    python pipeline/fetch_pexels.py --pick 123456                # download by id

The API key is read from the PEXELS_API env var (GitHub Actions secret)
or from a local .api.txt file containing PEXELS_API=<key>.
"""
import argparse
import json
from pathlib import Path

import requests

from keys import load_key

ROOT = Path(__file__).resolve().parent.parent
SEARCH_URL = "https://api.pexels.com/videos/search"
VIDEO_URL = "https://api.pexels.com/videos/videos/{id}"
MIN_SECONDS = 5
MAX_SECONDS = 60
TARGET_WIDTH = 960  # small enough for fast CPU study
DOWNLOAD_TRIES = 3


def search(key: str, query: str, per_page: int = 30) -> list[dict]:
    resp = requests.get(
        SEARCH_URL,
        headers={"Authorization": key},
        params={"query": query, "per_page": per_page, "size": "small"},
        timeout=30,
    )
    resp.raise_for_status()
    videos = resp.json().get("videos", [])
    return [v for v in videos if MIN_SECONDS <= v["duration"] <= MAX_SECONDS]


def get_video(key: str, video_id: int) -> dict:
    resp = requests.get(VIDEO_URL.format(id=video_id), headers={"Authorization": key}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def best_file(video: dict) -> dict:
    """Pick the mp4 whose width is closest to TARGET_WIDTH."""
    files = [f for f in video["video_files"] if f.get("file_type") == "video/mp4" and f.get("width")]
    return min(files, key=lambda f: abs(f["width"] - TARGET_WIDTH))


def download(video: dict, dest_dir: Path) -> Path:
    f = best_file(video)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"pexels_{video['id']}.mp4"
    for attempt in range(1, DOWNLOAD_TRIES + 1):
        try:
            with requests.get(f["link"], stream=True, timeout=120) as r:
                r.raise_for_status()
                with out.open("wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        fh.write(chunk)
            break
        except requests.RequestException as err:  # dropped connections happen on big files
            if attempt == DOWNLOAD_TRIES:
                raise
            print(f"download of {video['id']} failed ({err.__class__.__name__}), retrying")
    meta = {
        "id": video["id"],
        "page": video["url"],
        "author": video["user"]["name"],
        "author_url": video["user"]["url"],
        "duration": video["duration"],
        "width": f["width"],
        "height": f["height"],
        "credit": f"Video by {video['user']['name']} on Pexels",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("--pick", type=int, help="Pexels video id to download")
    args = ap.parse_args()

    key = load_key("PEXELS_API")
    if args.pick:
        path = download(get_video(key, args.pick), ROOT / "inbox")
        print(f"saved {path.relative_to(ROOT)}")
        return

    for v in search(key, args.query):
        slug = v["url"].rstrip("/").split("/")[-1]
        print(f"{v['id']:>10}  {v['duration']:>3}s  {v['width']}x{v['height']}  {slug}")


if __name__ == "__main__":
    main()
