from __future__ import annotations

"""AnimeImage MT01 flagship controller.

Input : 01/*.png
Output: MT01/*.png

The Hugging Face / Diffusers engine is intentionally separated into
tools/hf_anime_engine.py. This file is the only public MT01 entry point.
"""

import time
from pathlib import Path

import cv2

from hf_anime_engine import SEED, process

SRC = Path("01")
DST = Path("MT01")


def read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    if image.shape[2] == 4:
        return image[:, :, :3], image[:, :, 3]
    return image[:, :, :3], None


def save_png(path: Path, bgr, alpha):
    path.parent.mkdir(parents=True, exist_ok=True)
    output = cv2.dstack((bgr, alpha)) if alpha is not None else bgr
    if not cv2.imwrite(str(path), output, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"Cannot write: {path}")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.png"))
    if not files:
        raise SystemExit("No PNG files found in 01/.")

    print("=" * 64)
    print("AnimeImage MT01 — Hugging Face AI Image-to-Image")
    print(f"Input : {SRC}/")
    print(f"Output: {DST}/")
    print(f"Count : {len(files)}")
    print("=" * 64)

    started = time.perf_counter()

    for index, path in enumerate(files, 1):
        print(f"[{index:02d}/{len(files):02d}] {path.name}")
        original, alpha = read_image(path)
        original_size = (original.shape[1], original.shape[0])

        result = process(original, SEED + index)

        if result.shape[1] != original_size[0] or result.shape[0] != original_size[1]:
            result = cv2.resize(
                result,
                original_size,
                interpolation=cv2.INTER_LANCZOS4,
            )

        save_png(DST / path.name, result, alpha)
        print(f"       -> {DST / path.name}")

    elapsed = time.perf_counter() - started
    print("=" * 64)
    print(f"DONE: {len(files)} images in {elapsed:.1f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
