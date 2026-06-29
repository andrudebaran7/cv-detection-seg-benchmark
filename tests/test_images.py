from PIL import Image

from benchmark import images


def test_resolutions_constant():
    assert images.RESOLUTIONS == [320, 640, 960, 1280]


def test_resize_returns_square_target():
    src = Image.new("RGB", (1000, 500))
    out = images.resize(src, 640)
    assert out.size == (640, 640)
    assert out.mode == "RGB"
