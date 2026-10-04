from pathlib import Path
import cv2
import numpy as np

"""AnimeImage MT02 — multi-style, natural anime renderer.

This is a deterministic OpenCV renderer. It is designed to create several
clean anime-like render styles without claiming to be a generative AI model.
"""

SRC = Path("01")
DST = Path("MT02")
DST.mkdir(parents=True, exist_ok=True)


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def resize_for_render(bgra, max_side=1800):
    h, w = bgra.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return bgra
    return cv2.resize(bgra, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def smooth_preserve_edges(bgr):
    # Strong enough to remove sensor/compression noise, but keeps facial/hair edges.
    return cv2.bilateralFilter(bgr, 9, 55, 55)


def color_quantize(bgr, k=10):
    data = np.float32(bgr.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.8)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    return centers[labels.flatten()].reshape(bgr.shape).astype(np.uint8)


def anime_edges(bgr, strength=1):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 120)
    if strength > 1:
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    # Keep only meaningful dark contours.
    return cv2.GaussianBlur(edges, (3, 3), 0)


def apply_lines(color, edges, opacity=0.72):
    line_mask = edges.astype(np.float32) / 255.0
    line_mask = np.clip(line_mask * opacity, 0.0, 1.0)[..., None]
    ink = np.zeros_like(color)
    result = color.astype(np.float32) * (1.0 - line_mask) + ink * line_mask
    return np.clip(result, 0, 255).astype(np.uint8)


def tone(bgr, saturation=1.10, brightness=1.02, contrast=1.02):
    x = bgr.astype(np.float32) * contrast
    x = np.clip(x, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def soft_highlights(bgr):
    # Gentle local contrast creates a more photographic/anime illustration feel.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def render(bgra, style):
    bgr = bgra[:, :, :3]
    base = smooth_preserve_edges(bgr)

    if style == "natural":
        q = color_quantize(base, 12)
        q = soft_highlights(q)
        e = anime_edges(base, 1)
        out = apply_lines(q, e, 0.55)
        return tone(out, 1.08, 1.03, 1.01)

    if style == "cinematic":
        q = color_quantize(base, 16)
        e = anime_edges(base, 1)
        out = apply_lines(q, e, 0.42)
        return tone(out, 1.12, 1.00, 1.04)

    if style == "cel":
        q = color_quantize(base, 7)
        e = anime_edges(base, 2)
        out = apply_lines(q, e, 0.82)
        return tone(out, 1.16, 1.04, 1.02)

    if style == "soft":
        q = color_quantize(base, 18)
        q = cv2.GaussianBlur(q, (3, 3), 0)
        e = anime_edges(base, 1)
        out = apply_lines(q, e, 0.30)
        return tone(out, 1.05, 1.05, 1.00)

    if style == "manga":
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        ink = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 7
        )
        return cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unknown style: {style}")


STYLES = {
    "natural": "Natural Anime",
    "cinematic": "Cinematic Anime",
    "cel": "Strong Cel-Shading",
    "soft": "Soft Anime",
    "manga": "Manga Ink",
}


def save_png(path, bgr, alpha):
    if alpha is not None:
        out = np.dstack((bgr, alpha))
    else:
        out = bgr
    cv2.imwrite(str(path), out, [cv2.IMWRITE_PNG_COMPRESSION, 3])


for src in sorted(SRC.glob("*.png")):
    original = read_image(src)
    original = resize_for_render(original)
    alpha = original[:, :, 3]
    bgr = original[:, :, :3]

    stem = src.stem
    for key in STYLES:
        result = render(original, key)
        out_dir = DST / key
        out_dir.mkdir(parents=True, exist_ok=True)
        save_png(out_dir / f"{stem}.png", result, alpha)

    print(f"processed: {src.name} -> {len(STYLES)} anime renders")

print("MT02 complete: Natural / Cinematic / Cel / Soft / Manga")