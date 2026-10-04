# AnimeImage AI dependencies

## Architecture

`tools/anime_mt02.py` is the only MT01 entry point.

It sends each image to:

`tools/hf_anime_engine.py`

That module connects Python to Hugging Face Diffusers and performs
Image-to-Image generation. If the model cannot be loaded (weak/offline
machine, missing GPU memory, or another runtime error), the engine
automatically falls back to a lightweight OpenCV anime renderer.

## Files

- `requirements-ai.txt`: complete Python AI dependency list.
- `tools/install_ai.py`: installs the dependencies using the active Python.
- `tools/hf_anime_engine.py`: Hugging Face / Diffusers engine.
- `tools/anime_mt02.py`: MT01 controller.
- `01/`: exactly the source PNG set.
- `MT01/`: exactly the generated PNG set.

## Install

From the repository root:

```bat
python -m pip install -r requirements-ai.txt
```

Or:

```bat
python tools/install_ai.py
```

Then:

```bat
python tools/anime_mt02.py
```

## Hugging Face model storage

The model is **not committed into Git** because diffusion checkpoints are
large and would make the repository unnecessarily huge.

The engine uses `HF_HOME=.hf-cache` by default, so the local cache is kept
under the project folder. Hugging Face reuses already downloaded files on
later runs instead of downloading them again.

You can move the cache to another drive:

```bat
set HF_HOME=D:\HuggingFaceCache
set HF_HUB_CACHE=D:\HuggingFaceCache\hub
```

## Weak systems

The engine automatically chooses CPU when CUDA is unavailable.

For a low-memory NVIDIA GPU:

```bat
set ANIME_DEVICE=cuda
set ANIME_MAX_SIDE=512
set ANIME_STEPS=10
set ANIME_SEQUENTIAL_OFFLOAD=1
python tools/anime_mt02.py
```

For a CPU-only machine, AI diffusion can be very slow. The project still
works because the engine has a deterministic OpenCV fallback:

```bat
set ANIME_DEVICE=cpu
set ANIME_USE_AI=0
python tools/anime_mt02.py
```

## Output contract

The program processes every PNG in `01/` independently and writes:

```text
MT01/
├── 01.png
├── 02.png
├── 03.png
├── 04.png
├── 05.png
├── 06.png
├── 07.png
├── 08.png
├── 09.png
├── 10.png
├── 11.png
├── 12.png
└── 13.png
```

Original dimensions and PNG alpha are preserved.
