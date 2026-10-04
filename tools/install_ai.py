from __future__ import annotations

"""Bootstrap AnimeImage AI dependencies and optionally prefetch the HF model."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements-ai.txt"
MODEL_ID = os.getenv("ANIME_MODEL", "cagliostrolab/animagine-xl-3.1")
HF_HOME = Path(os.getenv("HF_HOME", str(ROOT / ".hf-cache"))).resolve()


def main():
    if not REQ.exists():
        raise SystemExit(f"Missing requirements file: {REQ}")

    print("1/2 Installing AnimeImage AI dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(REQ)]
    )

    print()
    print(f"2/2 Preparing Hugging Face model: {MODEL_ID}")
    print(f"    Cache: {HF_HOME}")

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=str(HF_HOME / "hub"),
    )

    print()
    print("AnimeImage AI environment is ready.")
    print("Run:")
    print("  python tools/anime_mt02.py")


if __name__ == "__main__":
    main()
