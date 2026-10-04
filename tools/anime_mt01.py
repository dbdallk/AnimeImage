from __future__ import annotations

"""AnimeImage MT01 — flagship AI Image-to-Image anime renderer.

Primary path:
    Hugging Face Diffusers + an anime-capable Stable Diffusion XL checkpoint.

Fallback path:
    deterministic OpenCV multi-stage anime rendering when AI dependencies/model
    are unavailable.

The AI model is intentionally NOT stored in Git. Set ANIME_MODEL to a model
you are licensed to use. The default is Animagine XL 3.1.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np

SRC = Path("01")
DST = Path("MT01")
MODEL_ID = os.getenv("ANIME_MODEL", "cagliostrolab/animagine-xl-3.1")
DEVICE = os.getenv("ANIME_DEVICE", "cuda")
STRENGTH = float(os.getenv("ANIME_STRENGTH", "0.48"))
GUIDANCE = float(os.getenv("ANIME_GUIDANCE", "6.5"))
STEPS = int(os.getenv("ANIME_STEPS", "28"))
SEED = int(os.getenv("ANIME_SEED", "1337"))
MAX_SIDE = int(os.getenv("ANIME_MAX_SIDE", "1024"))
USE_AI = os.getenv("ANIME_USE_AI", "1").lower() not in {"0", "false", "no"}

PROMPT = os.getenv(
    "ANIME_PROMPT",
    "masterpiece, best quality, anime illustration, natural anime face, "
    "clean lineart, expressive eyes, detailed hair, beautiful lighting, "
    "soft cinematic colors, coherent anatomy, highly detailed background",
)
NEGATIVE_PROMPT = os.getenv(
    "ANIME_NEGATIVE",
    "low quality, worst quality, blurry, jpeg artifacts, deformed, bad anatomy, "
    "extra fingers, missing fingers, extra limbs, text, watermark, logo, "
    "oversaturated, plastic skin",
)


def log(stage: str, message: str = "") -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {stage:<16} {message}")


def resize_for_model(image: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image
    nw = max(64, int(w * scale))
    nh = max(64, int(h * scale))
    # SD pipelines work best when dimensions are multiples of 8.
    nw -= nw % 8
    nh -= nh % 8
    nw = max(64, nw)
    nh = max(64, nh)
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def read_image(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img[:, :, :3], img[:, :, 3]


def save_png(path: Path, bgr: np.ndarray, alpha: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.dstack((bgr, alpha)) if alpha is not None else bgr
    cv2.imwrite(str(path), out, [cv2.IMWRITE_PNG_COMPRESSION, 3])


# --------------------------- AI Image-to-Image ---------------------------

_PIPELINE = None


def load_ai_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    import torch
    from diffusers import AutoPipelineForImage2Image

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA GPU is available.")

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    log("AI MODEL", f"loading {MODEL_ID} on {DEVICE}")

    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(DEVICE)

    # xFormers is optional. Do not make it a hard dependency.
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log("AI MEMORY", "xFormers attention enabled")
    except Exception:
        pass

    _PIPELINE = pipe
    return pipe


def ai_img2img(bgr: np.ndarray, seed: int) -> np.ndarray:
    from PIL import Image
    import torch

    pipe = load_ai_pipeline()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    log("AI RENDER", f"steps={STEPS} strength={STRENGTH:.2f} seed={seed}")
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=pil,
        strength=STRENGTH,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        generator=generator,
    ).images[0]

    return cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)


# ------------------------ High-quality OpenCV fallback ---------------------

def smooth(bgr: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(bgr, 9, 55, 55)


def quantize(bgr: np.ndarray, k: int = 12) -> np.ndarray:
    data = np.float32(bgr.reshape((-1, 3)))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        24,
        0.8,
    )
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    return centers[labels.flatten()].reshape(bgr.shape).astype(np.uint8)


def edges(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    e = cv2.Canny(gray, 45, 125)
    e = cv2.GaussianBlur(e, (3, 3), 0)
    return e


def line_render(color: np.ndarray, e: np.ndarray, opacity: float = 0.55) -> np.ndarray:
    mask = (e.astype(np.float32) / 255.0 * opacity)[..., None]
    return np.clip(color.astype(np.float32) * (1.0 - mask), 0, 255).astype(np.uint8)


def enhance(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.08, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.03, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def opencv_anime(bgr: np.ndarray) -> np.ndarray:
    base = smooth(bgr)
    q = quantize(base, 12)
    out = line_render(q, edges(base), 0.52)
    return enhance(out)


# ----------------------------- Batch pipeline ------------------------------

def process_one(path: Path, index: int, total: int) -> None:
    log("INPUT", f"{index}/{total} {path.name}")
    bgr, alpha = read_image(path)
    original_size = (bgr.shape[1], bgr.shape[0])

    log("PREPROCESS", f"resize <= {MAX_SIDE}px")
    model_input = resize_for_model(bgr)

    result = None
    if USE_AI:
        try:
            result = ai_img2img(model_input, SEED + index)
            log("AI OUTPUT", "image-to-image complete")
        except Exception as exc:
            log("AI FALLBACK", f"{type(exc).__name__}: {exc}")

    if result is None:
        log("OPENCV", "running deterministic anime renderer")
        result = opencv_anime(model_input)

    if result.shape[1] != original_size[0] or result.shape[0] != original_size[1]:
        result = cv2.resize(result, original_size, interpolation=cv2.INTER_LANCZOS4)

    save_png(DST / path.name, result, alpha)
    log("SAVE", str(DST / path.name))


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.png"))
    if not files:
        raise SystemExit("No PNG files found in 01/.")

    log("MT01 START", f"{len(files)} images | AI={USE_AI} | model={MODEL_ID}")
    started = time.perf_counter()

    for index, path in enumerate(files, 1):
        process_one(path, index, len(files))

    elapsed = time.perf_counter() - started
    log("MT01 DONE", f"{len(files)} images in {elapsed:.1f}s")
    print()
    print("MT01 = AI Image-to-Image flagship renderer")
    print(f"Output: {DST}/")
    print("If AI is unavailable, the script safely falls back to OpenCV.")


if __name__ == "__main__":
    main()
