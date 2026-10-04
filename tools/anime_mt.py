from pathlib import Path
import cv2
import numpy as np

SRC = Path("01")
DST = Path("MT")
DST.mkdir(parents=True, exist_ok=True)

def anime_filter(path: Path, out: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")

    alpha = None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3]
    elif img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img

    # Smooth flat areas while keeping important edges.
    smooth = cv2.bilateralFilter(bgr, 9, 70, 70)

    # Gentle color quantization for a cel/anime-paint look.
    data = np.float32(smooth.reshape((-1, 3)))
    k = 12 if data.shape[0] > 3000 else 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    quant = centers[labels.flatten()].reshape(smooth.shape).astype(np.uint8)

    # Clean, dark line work.
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 4
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    result = cv2.bitwise_and(quant, edges)

    # Slightly richer anime-style color without destroying the original layout.
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.18 + 3, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.04, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if alpha is not None:
        result = np.dstack((result, alpha))

    cv2.imwrite(str(out), result, [cv2.IMWRITE_PNG_COMPRESSION, 3])

for src in sorted(SRC.glob("*.png")):
    dst = DST / src.name
    anime_filter(src, dst)
    print(f"processed: {src} -> {dst}")

# MT batch trigger
