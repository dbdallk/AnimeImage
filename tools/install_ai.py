from __future__ import annotations

"""One-command AI environment bootstrap for AnimeImage.

This script installs the project's AI Python dependencies from
requirements-ai.txt. It does not copy a multi-gigabyte model into Git.
Hugging Face stores the model in a local cache and reuses it on later runs.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements-ai.txt"


def main():
    if not REQ.exists():
        raise SystemExit(f"Missing requirements file: {REQ}")

    print("Installing AnimeImage AI dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(REQ)]
    )
    print()
    print("AI environment is ready.")
    print("Run:")
    print("  python tools/anime_mt02.py")


if __name__ == "__main__":
    main()
