# 🎨 AnimeImage — آموزش تبدیل تصویر به استایل Anime با Python

> یک پروژه ساده و قابل فهم برای تبدیل گروهی تصاویر PNG به ظاهر نزدیک به **Anime / Cel-Shading** با استفاده از Python، OpenCV و NumPy.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Automation-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)

---

## ⭐ این پروژه دقیقاً چه کاری انجام می‌دهد؟

AnimeImage تصویر اصلی را می‌گیرد و با چند مرحله پردازش تصویر، ظاهر آن را به یک سبک کارتونی/انیمه‌ای نزدیک می‌کند.

**نکته مهم:** نسخه فعلی یک روش پردازش تصویر با OpenCV است و از مدل Generative AI مانند Stable Diffusion استفاده نمی‌کند. مزیت آن این است که سبک، قابل فهم و مناسب پردازش تعداد زیادی تصویر است.

### مسیر کلی پردازش

```text
PNG → Smooth → Color Quantization → Edge Detection → Color Enhancement → PNG
```

---

# 🐍 مهم‌ترین بخش: کد Python

هسته پروژه در فایل tools/anime_mt.py قرار دارد.

این قسمت، منطق اصلی تبدیل تصویر را نشان می‌دهد:

```python
from pathlib import Path
import cv2
import numpy as np

SRC = Path("01")
DST = Path("MT")

def anime_filter(path: Path, out: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    # 1) نرم کردن نواحی تصویر، بدون حذف کامل لبه‌ها
    smooth = cv2.bilateralFilter(img[:, :, :3], 9, 70, 70)

    # 2) کاهش تعداد رنگ‌ها برای ظاهر Cel-Shading
    data = np.float32(smooth.reshape((-1, 3)))
    k = 12 if data.shape[0] > 3000 else 8

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0
    )

    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 2,
        cv2.KMEANS_PP_CENTERS
    )

    quant = centers[labels.flatten()]
    quant = quant.reshape(smooth.shape).astype(np.uint8)

    # 3) استخراج خطوط
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9, 4
    )

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 4) ترکیب رنگ‌های ساده‌شده با خطوط
    result = cv2.bitwise_and(quant, edges)

    # 5) تقویت کنترل‌شده رنگ
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.18 + 3, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.04, 0, 255)

    result = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR
    )

    cv2.imwrite(str(out), result)

for src in sorted(SRC.glob("*.png")):
    anime_filter(src, DST / src.name)
```

### 🔍 در این کد چه اتفاقی می‌افتد؟

| مرحله | Python / OpenCV | نتیجه |
|---|---|---|
| 1 | cv2.imread | خواندن تصویر |
| 2 | cv2.bilateralFilter | نرم کردن با حفظ لبه‌ها |
| 3 | cv2.kmeans | کاهش و دسته‌بندی رنگ‌ها |
| 4 | cv2.cvtColor | تبدیل فضای رنگ |
| 5 | cv2.adaptiveThreshold | ساخت خطوط کارتونی |
| 6 | cv2.bitwise_and | ترکیب رنگ و خطوط |
| 7 | HSV + np.clip | تقویت کنترل‌شده رنگ |
| 8 | cv2.imwrite | ذخیره خروجی PNG |

---

# 🧠 روش‌های آسان برای Anime کردن تصویر

### روش 1 — Cartoon / Anime Filter

**آسان‌ترین روش**

1. Smooth کردن تصویر
2. کاهش تعداد رنگ‌ها
3. استخراج خطوط
4. افزایش Saturation
5. ذخیره PNG

این روش سریع است و برای Batch Processing مناسب است.

### روش 2 — Edge + Color Quantization

اگر خطوط واضح‌تر می‌خواهید:

```text
Original
   ↓
Bilateral Filter
   ↓
K-Means
   ↓
Edge Detection
   ↓
Blend
   ↓
Anime-like Image
```

این همان ایده اصلی نسخه فعلی پروژه است.

### روش 3 — Generative AI

برای تغییر بسیار شدیدتر و تولید ظاهر جدید می‌توان از مدل‌های **Stable Diffusion / ControlNet / Image-to-Image** استفاده کرد.

این روش‌ها معمولاً سنگین‌تر هستند، تنظیمات بیشتری می‌خواهند و ممکن است ترکیب‌بندی تصویر اصلی را تغییر دهند. بنابراین نسخه فعلی AnimeImage عمداً با OpenCV شروع کرده است.

---

# ⚙️ چگونه نتیجه را بهتر کنیم؟

پارامترهای اصلی در Python قابل تغییر هستند.

### نرم‌تر کردن تصویر

```python
smooth = cv2.bilateralFilter(bgr, 9, 70, 70)
```

- مقدار بیشتر → نرم‌تر
- مقدار کمتر → جزئیات بیشتر

### تعداد رنگ‌ها

```python
k = 12
```

- k = 4 → رنگ‌های بسیار ساده
- k = 8 → کارتونی
- k = 12 → جزئیات رنگی بیشتر
- k = 16 یا بیشتر → نزدیک‌تر به تصویر اصلی

### شدت رنگ

```python
hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.18 + 3, 0, 255)
```

افزایش Saturation رنگ‌ها را زنده‌تر می‌کند؛ کاهش آن نتیجه طبیعی‌تر می‌دهد.

### خطوط

برای ظاهر Manga/Anime، Edge Detection مهم است. با تغییر Median Blur و Adaptive Threshold می‌توان شدت و جزئیات خطوط را کنترل کرد.

---

# 🖼️ نمونه‌های پروژه

ورودی‌ها در 01/ و خروجی‌های پردازش‌شده در MT/ قرار دارند.

| ورودی | خروجی |
|---|---|
| 01/01.png | MT/01.png |
| 01/02.png | MT/02.png |
| 01/03.png | MT/03.png |
| 01/04.png | MT/04.png |
| 01/05.png | MT/05.png |
| 01/06.png | MT/06.png |
| 01/07.png | MT/07.png |
| 01/08.png | MT/08.png |
| 01/09.png | MT/09.png |
| 01/10.png | MT/10.png |
| 01/11.png | MT/11.png |
| 01/12.png | MT/12.png |
| 01/13.png | MT/13.png |

---

# 🚀 اجرای پروژه با Python

```bash
pip install opencv-python numpy
python tools/anime_mt.py
```

اسکریپت همه PNGهای پوشه 01 را پیدا می‌کند و خروجی را در MT می‌سازد.

---

# 🤖 پردازش خودکار با GitHub Actions

پروژه Workflow دارد و می‌تواند پردازش تصاویر را خودکار کند:

1. آماده‌سازی Python
2. نصب OpenCV و NumPy
3. اجرای tools/anime_mt.py
4. تولید خروجی‌های MT/
5. Commit کردن تغییرات تولیدشده

---

# 💡 ایده‌های نسخه‌های آینده

- 🎛️ کنترل شدت Anime
- 🎨 چند Preset مختلف
- ✏️ حالت Manga
- 🌸 حالت Soft Anime
- 🔥 حالت Strong Anime
- 🖌️ حالت Cel-Shading
- 🤖 Image-to-Image با مدل AI
- 🧠 Stable Diffusion / ControlNet
- 🖼️ پشتیبانی JPG / WEBP
- 🎞️ پردازش ویدئو
- 🖥️ رابط گرافیکی Python
- 📱 نسخه Android
- 🌐 Web API
- 📦 خروجی EXE
- ⚡ پردازش موازی تعداد زیاد تصویر

---

# 📁 ساختار پروژه

```text
AnimeImage/
├── 01/
│   ├── 01.png
│   ├── 02.png
│   └── ... 13.png
├── MT/
│   ├── 01.png
│   ├── 02.png
│   └── ... 13.png
├── tools/
│   └── anime_mt.py
├── .github/
│   └── workflows/
│       └── anime-mt.yml
├── README.md
└── AnimeImage.md
```

---

# ❤️ اگر این پروژه برای شما مفید بود

⭐ به Repository ستاره بدهید  
🍴 پروژه را Fork کنید  
👀 پروژه را دنبال کنید  
💬 پیشنهاد یا Issue ثبت کنید  
🔧 Pull Request ارسال کنید  

هر ⭐ به دیده‌شدن بیشتر پروژه کمک می‌کند.

---

# 🔥 هشتگ‌های پیشنهادی

**#AnimeImage #Python #PythonProgramming #OpenCV #NumPy #ImageProcessing #ComputerVision #Anime #AnimeArt #AnimeFilter #ImageToAnime #PhotoToAnime #AIArt #DigitalArt #MachineLearning #DeepLearning #ComputerVisionAI #PythonProject #OpenSource #GitHub #GitHubProjects #Coding #Programming #OpenCVPython #ImageEditing #PhotoEditing #AnimeStyle #CelShading #Manga #Developer #SoftwareDevelopment #ArtificialIntelligence #AI #Tech**

---

## 🌟 هدف AnimeImage

این پروژه نشان می‌دهد که با چند تکنیک قابل فهم Python و OpenCV می‌توان بدون مدل‌های سنگین، تصویر معمولی را به ظاهری نزدیک به **Anime / Cartoon** تبدیل کرد.

مسیر پیشنهادی توسعه:

**OpenCV → Presets → GUI → Batch Processing → AI Image-to-Image**

> ساخته‌شده با ❤️ و Python برای یادگیری، آزمایش و توسعه پردازش تصویر.
