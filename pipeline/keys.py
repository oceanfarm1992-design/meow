"""Load API keys from env vars (GitHub Actions secrets) or the local .api.txt."""
import os
import sys
from pathlib import Path

KEY_FILE = Path(__file__).resolve().parent.parent / ".api.txt"


def load_key(name: str) -> str:
    key = os.environ.get(name, "").strip()
    if key:
        return key
    if KEY_FILE.exists():
        for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
            k, _, value = line.partition("=")
            if k.strip() == name and value.strip():
                return value.strip()
    sys.exit(f"{name} not found in env or .api.txt")
