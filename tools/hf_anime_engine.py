from __future__ import annotations

"""Hugging Face / Diffusers engine used by AnimeImage MT01.

The engine is deliberately separated from anime_mt02.py so the project
controller stays small. It supports automatic device selection, CPU/GPU
memory saving, Hugging Face cache placement, and a deterministic OpenCV
fallback for weak/offline machines.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

MODEL_ID = os.getenv("ANIME_MODEL", "cagliostrolab/animagine-xl-3.1")
DEVICE_REQUEST = os.getenv("ANIME_DEVICE", "auto").lower()
USE_AI = os.getenv("ANIME_USE_AI", "1").lower() not in {"0", "false", "no"}
STEPS = int(os.getenv("ANIME_STEPS", "18"))
STRENGTH = float(os.getenv("ANIME_STRENGTH", "0.42"))
GUIDANCE = float(os.getenv("ANIME_GUIDANCE", "6.0"))
MAX_SIDE = int(os.getenv("ANIME_MAX_SIDE", "768"))
SEED = int(os.getenv("ANIME_SEED", "1337"))
SEQUENTIAL_OFFLOAD = os.getenv("ANIME_SEQUENTIAL_OFFLOAD", "auto").lower()
CACHE_DIR = Path(os.getenv("HF_HOME", ".hf-cache"))

PROMPT = os.getenv(
    "ANIME_PROMPT",
    "masterpiece, best quality, anime illustration, clean lineart, "
    "beautiful lighting, detailed hair, natural face, coherent anatomy, "
    "rich natural colors, polished digital illustration",
)
NEGATIVE_PROMPT = os.getenv(
    "ANIME_NEGATIVE",
    "low quality, worst quality, blurry, jpeg artifacts, deformed, "
    "bad anatomy, bad hands, extra fingers, extra limbs, duplicate, "
    "distorted face, text, watermark, logo, oversaturated, noisy",
)

_PIPE = None
_DEVICE = None


def choose_device() -> str:
    if DEVICE_REQUEST in {"cpu", "cuda", "mps"}:
        if DEVICE_REQUEST == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
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


def resize_for_ai(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    nw = max(64, int(w * scale))
    nh = max(64, int(h * scale))
    nw -= nw % 8
    nh -= nh % 8
    nw = max(64, nw)
    nh = max(64, nh)
    if (nw, nh) == (w, h):
        return bgr
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def load_pipeline():
    global _PIPE, _DEVICE
    if _PIPE is not None:
        return _PIPE

    import torch
    from diffusers import AutoPipelineForImage2Image

    _DEVICE = choose_device()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_DIR.resolve()))
    os.environ.setdefault("HF_HUB_CACHE", str((CACHE_DIR / "hub").resolve()))

    dtype = torch.float16 if _DEVICE == "cuda" else torch.float32
    print(f"[HF] model={MODEL_ID}")
    print(f"[HF] device={_DEVICE}")
    print(f"[HF] cache={CACHE_DIR.resolve()}")

    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=str((CACHE_DIR / "hub").resolve()),
    )

    if _DEVICE == "cuda":
        # model offload is the preferred balance for small NVIDIA GPUs.
        if SEQUENTIAL_OFFLOAD in {"1", "true", "yes"}:
            pipe.enable_sequential_cpu_offload()
            print("[HF] sequential CPU offload enabled")
        elif SEQUENTIAL_OFFLOAD == "auto":
            try:
                pipe.enable_model_cpu_offload()
                print("[HF] model CPU offload enabled")
            except Exception:
                pipe.to("cuda")
        else:
            pipe.to("cuda")
    else:
        pipe.to(_DEVICE)

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    # PyTorch 2.x already provides efficient SDPA; xFormers is optional.
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[HF] xFormers enabled")
    except Exception:
        pass

    _PIPE = pipe
    return pipe


def ai_process(bgr: np.ndarray, seed: int, strength: Optional[float] = None) -> np.ndarray:
    from PIL import Image
    import torch

    pipe = load_pipeline()
    work = resize_for_ai(bgr)
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    gen_device = "cuda" if _DEVICE == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=image,
        strength=max(0.05, min(0.95, strength if strength is not None else STRENGTH)),
        guidance_scale=max(1.0, GUIDANCE),
        num_inference_steps=max(1, STEPS),
        generator=generator,
    ).images[0]

    out = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
    if out.shape[:2] != bgr.shape[:2]:
        out = cv2.resize(out, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return out


def opencv_fallback(bgr: np.ndarray) -> np.ndarray:
    smooth = cv2.bilateralFilter(bgr, 9, 55, 55)

    data = np.float32(smooth.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.8)
    _, labels, centers = cv2.kmeans(data, 10, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    quant = centers[labels.flatten()].reshape(smooth.shape).astype(np.uint8)

    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 125)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), 1)
    mask = (edges.astype(np.float32) / 255.0 * 0.30)[..., None]
    out = np.clip(quant.astype(np.float32) * (1.0 - mask), 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return out


def process(bgr: np.ndarray, seed: int) -> np.ndarray:
    if not USE_AI:
        return opencv_fallback(bgr)

    try:
        generated = ai_process(bgr, seed)
        # Keep 10% of the source structure for stable image-to-image output.
        work = resize_for_ai(bgr)
        generated = cv2.addWeighted(generated, 0.90, work, 0.10, 0)
        return generated
    except Exception as exc:
        print(f"[HF FALLBACK] {type(exc).__name__}: {exc}")
        return opencv_fallback(bgr)
