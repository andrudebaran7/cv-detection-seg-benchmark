from __future__ import annotations

import pathlib

from PIL import Image

RESOLUTIONS = [320, 640, 960, 1280]


def _dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "data" / "benchmark_images"


def load_images() -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in sorted(_dir().glob("*.jpg"))]


def resize(img: Image.Image, size: int) -> Image.Image:
    return img.convert("RGB").resize((size, size))
