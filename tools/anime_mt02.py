from __future__ import annotations

"""AnimeImage MT01 engine — implemented in anime_mt02.py.

There is intentionally NO separate anime_mt01.py file.
This single script is the flagship MT01 processor.

Pipeline
--------
01/*.png
   -> high-quality preprocessing
   -> optional Python AI Image-to-Image (Diffusers/SDXL anime model)
   -> optional second AI refinement pass
   -> OpenCV anime finishing / color cleanup
   -> original-size PNG with alpha
   -> MT01/*.png

The AI model is downloaded by Diffusers at runtime and is never committed to
this repository. Set ANIME_MODEL to a model you are licensed to use.

Useful environment variables:
    ANIME_USE_AI=1/0
    ANIME_MODEL=cagliostrolab/animagine-xl-3.1
    ANIME_DEVICE=auto/cuda/cpu
    ANIME_STRENGTH=0.45
    ANIME_GUIDANCE=6.5
    ANIME_STEPS=28
    ANIME_REFINEMENT=1
    ANIME_REFINE_STRENGTH=0.18
    ANIME_MAX_SIDE=1024
    ANIME_SEED=1337
    ANIME_PROMPT="..."
    ANIME_NEGATIVE="..."
    ANIME_CUDA_OFFLOAD=1
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np

SRC = Path("01")
DST = Path("MT01")

MODEL_ID = os.getenv("ANIME_MODEL", "cagliostrolab/animagine-xl-3.1")
DEVICE_REQUEST = os.getenv("ANIME_DEVICE", "auto").lower()
USE_AI = os.getenv("ANIME_USE_AI", "1").lower() not in {"0", "false", "no"}
AI_STRENGTH = float(os.getenv("ANIME_STRENGTH", "0.45"))
GUIDANCE = float(os.getenv("ANIME_GUIDANCE", "6.5"))
STEPS = int(os.getenv("ANIME_STEPS", "28"))
REFINEMENT = os.getenv("ANIME_REFINEMENT", "1").lower() not in {"0", "false", "no"}
REFINE_STRENGTH = float(os.getenv("ANIME_REFINE_STRENGTH", "0.18"))
SEED = int(os.getenv("ANIME_SEED", "1337"))
MAX_SIDE = int(os.getenv("ANIME_MAX_SIDE", "1024"))
CUDA_OFFLOAD = os.getenv("ANIME_CUDA_OFFLOAD", "1").lower() not in {"0", "false", "no"}

PROMPT = os.getenv(
    "ANIME_PROMPT",
    "masterpiece, best quality, anime illustration, high quality anime art, "
    "clean precise lineart, expressive eyes, detailed hair, natural face, "
    "coherent anatomy, beautiful lighting, cinematic composition, "
    "rich but natural colors, detailed background, polished digital illustration",
)

NEGATIVE_PROMPT = os.getenv(
    "ANIME_NEGATIVE",
    "low quality, worst quality, blurry, jpeg artifacts, deformed, "
    "bad anatomy, bad hands, extra fingers, missing fingers, extra limbs, "
    "duplicate, distorted face, poorly drawn face, text, watermark, logo, "
    "oversaturated, plastic skin, noisy, cropped",
)

_PIPELINE = None
_RUNTIME_DEVICE = None


def log(stage: str, message: str = "") -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {stage:<18} {message}")


def choose_device() -> str:
    if DEVICE_REQUEST in {"cuda", "cpu", "mps"}:
        if DEVICE_REQUEST == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    log("DEVICE", "CUDA requested but unavailable; using CPU")
                    return "cpu"
            except Exception:
                return "cpu"
        return DEVICE_REQUEST

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def read_image(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    if img.ndim == 2:
        gray = img
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        alpha = None
    elif img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img[:, :, :3]
        alpha = None

    return bgr, alpha


def resize_for_model(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale >= 1.0:
        # Diffusion models prefer dimensions divisible by 8.
        nw = max(64, w - (w % 8))
        nh = max(64, h - (h % 8))
        if (nw, nh) != (w, h):
            return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        return bgr

    nw = max(64, int(w * scale))
    nh = max(64, int(h * scale))
    nw = max(64, nw - (nw % 8))
    nh = max(64, nh - (nh % 8))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def save_png(path: Path, bgr: np.ndarray, alpha: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.dstack((bgr, alpha)) if alpha is not None else bgr
    ok = cv2.imwrite(str(path), out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"Cannot write image: {path}")


def load_ai_pipeline():
    global _PIPELINE, _RUNTIME_DEVICE

    if _PIPELINE is not None:
        return _PIPELINE

    import torch
    from diffusers import AutoPipelineForImage2Image

    _RUNTIME_DEVICE = choose_device()
    log("AI MODEL", f"{MODEL_ID}")
    log("AI DEVICE", _RUNTIME_DEVICE)

    if _RUNTIME_DEVICE == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    # Offload is useful on smaller NVIDIA cards. For CPU, normal .to("cpu")
    # is simpler and avoids unnecessary offload hooks.
    if _RUNTIME_DEVICE == "cuda" and CUDA_OFFLOAD:
        try:
            pipe.enable_model_cpu_offload()
            log("AI MEMORY", "model CPU offload enabled")
        except Exception:
            pipe = pipe.to("cuda")
    else:
        pipe = pipe.to(_RUNTIME_DEVICE)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        log("AI MEMORY", "xFormers attention enabled")
    except Exception:
        # PyTorch 2.x has native efficient attention, so this is optional.
        pass

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    _PIPELINE = pipe
    return pipe


def ai_img2img(bgr: np.ndarray, seed: int, strength: float) -> np.ndarray:
    from PIL import Image
    import torch

    pipe = load_ai_pipeline()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    device_for_generator = "cuda" if _RUNTIME_DEVICE == "cuda" else "cpu"
    generator = torch.Generator(device=device_for_generator).manual_seed(seed)

    log(
        "AI RENDER",
        f"steps={STEPS} strength={strength:.2f} guidance={GUIDANCE:.2f} seed={seed}",
    )

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=pil,
        strength=max(0.05, min(0.95, strength)),
        guidance_scale=max(1.0, GUIDANCE),
        num_inference_steps=max(1, STEPS),
        generator=generator,
    ).images[0]

    return cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# OpenCV finishing engine — always available and used as a safe fallback.
# ---------------------------------------------------------------------------

def smooth_preserve_edges(bgr: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(bgr, 9, 55, 55)


def color_quantize(bgr: np.ndarray, k: int = 12) -> np.ndarray:
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


def anime_edges(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(gray, 40, 125)
    edge = cv2.dilate(edge, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.GaussianBlur(edge, (3, 3), 0)


def apply_lines(color: np.ndarray, edge: np.ndarray, opacity: float = 0.34) -> np.ndarray:
    mask = np.clip(edge.astype(np.float32) / 255.0 * opacity, 0.0, 1.0)[..., None]
    return np.clip(color.astype(np.float32) * (1.0 - mask), 0, 255).astype(np.uint8)


def color_finish(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.15, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.06, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.025, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def opencv_anime_finish(bgr: np.ndarray) -> np.ndarray:
    base = smooth_preserve_edges(bgr)
    quantized = color_quantize(base, 12)
    outlined = apply_lines(quantized, anime_edges(base), 0.32)
    return color_finish(outlined)


def blend_ai_with_original(original: np.ndarray, generated: np.ndarray) -> np.ndarray:
    # A small structural blend keeps the source recognizable while allowing
    # the diffusion model to change the artistic style.
    original_f = original.astype(np.float32)
    generated_f = generated.astype(np.float32)
    out = generated_f * 0.90 + original_f * 0.10
    return np.clip(out, 0, 255).astype(np.uint8)


def process_one(path: Path, index: int, total: int) -> None:
    log("INPUT", f"{index}/{total} {path.name}")

    original, alpha = read_image(path)
    original_size = (original.shape[1], original.shape[0])

    model_input = resize_for_model(original)
    log("PREPROCESS", f"{model_input.shape[1]}x{model_input.shape[0]}")

    result = None

    if USE_AI:
        try:
            result = ai_img2img(model_input, SEED + index, AI_STRENGTH)
            log("AI OUTPUT", "first image-to-image pass complete")

            if REFINEMENT:
                result = ai_img2img(
                    result,
                    SEED + index + 100000,
                    REFINE_STRENGTH,
                )
                log("AI REFINE", "second detail/style pass complete")

            result = blend_ai_with_original(model_input, result)
            result = opencv_anime_finish(result)
            log("FINISH", "AI + OpenCV anime finishing complete")

        except Exception as exc:
            log("AI FALLBACK", f"{type(exc).__name__}: {exc}")
            result = None

    if result is None:
        log("OPENCV", "AI unavailable; deterministic renderer activated")
        result = opencv_anime_finish(model_input)

    if result.shape[1] != original_size[0] or result.shape[0] != original_size[1]:
        result = cv2.resize(
            result,
            original_size,
            interpolation=cv2.INTER_LANCZOS4,
        )

    save_png(DST / path.name, result, alpha)
    log("SAVE", f"{DST / path.name}")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC.glob("*.png"))
    if not files:
        raise SystemExit("No PNG files found in 01/.")

    log("MT01 START", f"{len(files)} images | AI={USE_AI}")
    log("OUTPUT", str(DST))
    started = time.perf_counter()

    for index, path in enumerate(files, 1):
        process_one(path, index, len(files))

    elapsed = time.perf_counter() - started
    log("MT01 DONE", f"{len(files)} images in {elapsed:.1f}s")

    print()
    print("=" * 64)
    print("AnimeImage MT01 — flagship AI Image-to-Image")
    print("=" * 64)
    print(f"Input : {SRC}/")
    print(f"Output: {DST}/")
    print(f"Images: {len(files)}")
    print(f"AI    : {'ON' if USE_AI else 'OFF'}")
    print(f"Model : {MODEL_ID}")
    print("=" * 64)


if __name__ == "__main__":
    main()
