import numpy as np
from PIL import Image

from models.base import Box, Prediction
from app.components.visualization import draw_prediction


def test_draw_prediction_returns_new_image_same_size():
    img = Image.new("RGB", (100, 80), "white")
    pred = Prediction(
        boxes=[Box(10, 10, 50, 50)],
        labels=["cat"],
        scores=[0.88],
        latency_ms=5.0,
    )
    out = draw_prediction(img, pred)

    assert isinstance(out, Image.Image)
    assert out.size == (100, 80)
    assert out is not img  # must not mutate the original
    # something was drawn: output differs from a blank white image
    assert not np.array_equal(np.array(out), np.array(img))
